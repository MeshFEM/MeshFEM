////////////////////////////////////////////////////////////////////////////////
// FEMView.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      OpenGL-based viewer for the MeshlessFEM/CSG code.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//
//  Created:  01/28/2013 15:22:32
//  Revision History:
//      01/28/2013  Julian Panetta    Initial Revision
////////////////////////////////////////////////////////////////////////////////
#include "FEMView.hh"
#include <QtGui>
#include <QGLWidget>
#include <QColor>
#include <cassert>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <limits>
#include <boost/format.hpp>

#include "MeshlessFEM.hh"
#include "ShaderCompiler.hh"
// #include "timing.h"

#define MAX_NODES 128
#define MAX_PRIMITIVES 64
typedef struct _CSGPrimitiveData {
    cl_float2 center;
    union {
        struct {
            cl_float2 half_dim;
            cl_float2 rotationCosSin;
        } rect;
        struct {
            cl_float2 focus;
            float double_majorRadius;
        } ellipse;
    };
} CSGPrimitiveData;


FEMView2D::FEMView2D(MeshlessFEM_t &fem, const ViewSettings &vs,
                     QWidget *parent)
    : QGLWidget(parent),
      m_font("/Users/jpanetta/Research/CSGFEM/fonts/Arial.ttf"),
      m_frameDim(4, 3), m_frameCenter(0, 0),
      m_fem(fem), m_result(NULL),
      m_pressurePaintValue(0.1),
      m_guiState(STATE_MODEL), m_gesture(GESTURE_NONE),
      m_displacementPhase(0.0), m_viewSettings(vs),
      m_scalarColorMap(COLORMAP_JET),
      m_modelTexBuf(NULL), m_overlayTexBuf(NULL)
{
    if (m_font.Error())
        throw std::runtime_error("Failed to load font!");
    m_font.FaceSize(12);
    setFormat(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer));
    setFocusPolicy(Qt::StrongFocus);
    viewSettingsUpdated();
}

void FEMView2D::csgNodesSelected(const NodeList &nList)
{
    m_selectedObjects = nList;
    m_rerenderOverlay();
    update();
}

void FEMView2D::viewSettingsUpdated()
{
    m_scalarColorMap.selectMap(m_viewSettings.colormap);

    if (isVibrating())
        m_timer.start(1000.0 / 60, this);
    else
        m_timer.stop();

    update();
}
    
void FEMView2D::initializeGL()
{
    // Create OpenCL context sharing with the OpenGL context
    CGLContextObj kCGLContext = CGLGetCurrentContext();                   
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);      

    cl_context_properties props[] = {                                     
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,                     
        (cl_context_properties) kCGLShareGroup,                           
        0                                                                 
    };                                                                    

    cl_int status;                                                        
    m_clContext = clCreateContext(props, 0, 0, NULL, 0, &status);               
    // compute the number of devices                                      
    cl_int err;                                                           
    size_t ret_size;                                                      
    err = clGetContextInfo(m_clContext, CL_CONTEXT_DEVICES, 0, NULL, &ret_size);
    CHECK_CL_ERROR(err, "clGetContextInfo");                              
    cl_int numDevices = ret_size / sizeof(cl_device_id);                  

    // Get the device list                                                
    cl_device_id devices[numDevices];                                     
    err = clGetContextInfo(m_clContext, CL_CONTEXT_DEVICES, ret_size, devices,  
            &ret_size);                                    
    CHECK_CL_ERROR(err, "clGetContextInfo");                              

    // Get the GPU device and queue                                       
    for(int i = 0; i < numDevices; ++i) {                                 
        cl_int deviceType, error;                                         
        err = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE,                 
                sizeof(cl_device_type), &deviceType, &ret_size);          
        CHECK_CL_ERROR(err, "clGetDeviceInfo");                           

        if (deviceType == CL_DEVICE_TYPE_GPU) {                           
            cl_device_id dev = devices[i];                                
            m_clQueue = clCreateCommandQueue(m_clContext, dev, 0, &error);        
            CHECK_CL_ERROR(error, "clCreateCommandQueue");                

            break;                                                        
        }                                                                 
    }                                                                     

    // Allocate CSG Tree host/device buffers
    m_nodeBuf = clCreateBuffer(m_clContext, CL_MEM_READ_ONLY,
                               MAX_NODES * sizeof(CSGNodeType),
                               0, &err);
    CHECK_CL_ERROR(err, "Creating node buffer");
    m_primBuf = clCreateBuffer(m_clContext, CL_MEM_READ_ONLY,
                               MAX_PRIMITIVES * sizeof(CSGPrimitiveData),
                               0, &err);

    CHECK_CL_ERROR(err, "Creating primitive buffer");
    m_nodeHostBuf = clCreateBuffer(m_clContext,
                               CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               MAX_NODES * sizeof(CSGNodeType),
                               NULL, &err);
    m_primHostBuf = clCreateBuffer(m_clContext,
                               CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               MAX_PRIMITIVES * sizeof(CSGPrimitiveData),
                               NULL, &err);

    // Load and compile OpenCL Kernels
    char *knl_text = read_file("/Users/jpanetta/Research/CSGFEM/Kernels/RenderCSG.cl");
    assert(knl_text != NULL);
    m_renderKernel = kernel_from_string(m_clContext, knl_text,
                                        "RenderCSG", NULL);
    free(knl_text);
    knl_text = read_file("/Users/jpanetta/Research/CSGFEM/Kernels/ClearTexture.cl");
    assert(knl_text != NULL);
    m_clearKernel = kernel_from_string(m_clContext, knl_text,
                                       "ClearTexture", NULL);
    free(knl_text);

    glClearColor(0.8, 0.8, 0.8, 1.0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    
    glGenTextures(1, &m_modelTex);
    glBindTexture(GL_TEXTURE_2D, m_modelTex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glGenTextures(1, &m_overlayTex);
    glBindTexture(GL_TEXTURE_2D, m_overlayTex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // TODO: make these paths relative to the application bundle and copy the
    // shaders in the build rules
    readShader("/Users/jpanetta/Research/CSGFEM/Shaders/BilinearShader.vert",
               "/Users/jpanetta/Research/CSGFEM/Shaders/BilinearShader.frag",
               m_bilinearShader);
    glUseProgram(m_bilinearShader);
    m_vCoordLoc[0] = glGetAttribLocation(m_bilinearShader, "point0");
    m_vCoordLoc[1] = glGetAttribLocation(m_bilinearShader, "point1");
    m_vCoordLoc[2] = glGetAttribLocation(m_bilinearShader, "point2");
    m_vCoordLoc[3] = glGetAttribLocation(m_bilinearShader, "point3");

    m_tCoordLoc[0] = glGetAttribLocation(m_bilinearShader, "texCoord0");
    m_tCoordLoc[1] = glGetAttribLocation(m_bilinearShader, "texCoord1");
    m_tCoordLoc[2] = glGetAttribLocation(m_bilinearShader, "texCoord2");
    m_tCoordLoc[3] = glGetAttribLocation(m_bilinearShader, "texCoord3");

    m_objectTexLoc = glGetUniformLocation(m_bilinearShader, "objectTex");
    glUseProgram(0);
}

void FEMView2D::resizeGL(int width, int height)
{
    // Largest possible viewing rectangle with the frame's aspect ratio
    // aspect = view width / height
    float aspect = m_frameDim[0] / m_frameDim[1];
    float proportionalWidth = aspect * height;
    if (proportionalWidth < width) {
        m_width  = proportionalWidth;
        m_height = height;
    }
    else {
        m_width = width;
        m_height = width / aspect;
    }

    // Allocate empty textures
    glBindTexture(GL_TEXTURE_2D, m_modelTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, m_overlayTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, NULL);

    cl_int err;

    if (m_modelTexBuf)
        CALL_CL_GUARDED(clReleaseMemObject, (m_modelTexBuf));
    m_modelTexBuf = clCreateFromGLTexture(m_clContext, CL_MEM_WRITE_ONLY,
                                            GL_TEXTURE_2D, 0, m_modelTex, &err);
    CHECK_CL_ERROR(err, "clCreateFromGLTexture");

    if (m_overlayTexBuf)
        CALL_CL_GUARDED(clReleaseMemObject, (m_overlayTexBuf));
    m_overlayTexBuf = clCreateFromGLTexture(m_clContext, CL_MEM_WRITE_ONLY,
                                            GL_TEXTURE_2D, 0, m_overlayTex, &err);

    m_rerenderObject();
    m_rerenderOverlay();

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Center m_width, m_height box
    int hmargin = (height - m_height) / 2;
    int wmargin = (width  -  m_width) / 2;
    glOrtho(-wmargin, width - wmargin, -hmargin, height - hmargin, -1, 1);
    m_screenTop = height - hmargin;
    m_screenLeft = -wmargin;
    glMatrixMode(GL_MODELVIEW);
}

void drawQuad(float minx, float miny, float maxx, float maxy) {
    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex2f(minx, miny);
    glTexCoord2f(1, 0);
    glVertex2f(maxx, miny);
    glTexCoord2f(1, 1);
    glVertex2f(maxx, maxy);
    glTexCoord2f(0, 1);
    glVertex2f(minx, maxy);

    glEnd();
}

struct CSGTreeFlattener {
    CSGTreeFlattener(CSGNodeType *ntype, CSGPrimitiveData *pdata)
        : nodeTypes(ntype), primitiveData(pdata),
          numNodes(0), numPrimitives(0) { }
    void preVisit(CSGNode *) { } 
    void postVisit(CSGNode *node) {
        CSGNodeType type = node->nodeType();
        assert(numNodes < MAX_NODES);
        nodeTypes[numNodes++] = type;

        if (type == CSG_NODE_RECT) {
            assert(numPrimitives < MAX_PRIMITIVES);
            CSGPrimitiveData &p = primitiveData[numPrimitives++];
            CSGRectangleNode *r = dynamic_cast<CSGRectangleNode *>(node);
            assert(r);
            Vector dim = .5 * r->getDimensions();
            p.center.s[0]        = r->getCenter()[0];
            p.center.s[1]        = r->getCenter()[1];
            p.rect.half_dim.s[0] = dim[0];
            p.rect.half_dim.s[1] = dim[1];
            p.rect.rotationCosSin.s[0] = cos(-r->getRotationRad());
            p.rect.rotationCosSin.s[1] = sin(-r->getRotationRad());
        }

        else if (type == CSG_NODE_ELLIPSE) {
            assert(numPrimitives < MAX_PRIMITIVES);
            CSGPrimitiveData &p = primitiveData[numPrimitives++];
            CSGEllipseNode *e = dynamic_cast<CSGEllipseNode *>(node);
            assert(e);
            Vector focus = e->getFocus();
            p.center.s[0]                   = e->getCenter()[0];
            p.center.s[1]                   = e->getCenter()[1];
            p.ellipse.focus.s[0]            = focus[0];
            p.ellipse.focus.s[1]            = focus[1];
            p.ellipse.double_majorRadius = 2.0 * e->getMajorRadius();
        }
    }

    CSGNodeType      *nodeTypes;
    CSGPrimitiveData *primitiveData;
    int numNodes, numPrimitives;
};

void FEMView2D::m_clClearCSGRender(cl_mem texBuf)
{
    CALL_CL_GUARDED(clEnqueueAcquireGLObjects,
            (m_clQueue, 1, &texBuf, 0, NULL, NULL));
    size_t ldim[] = {128, 1};
    size_t gdim[] = {(((size_t) m_height + ldim[0]) / ldim[0]) * ldim[0],
                       (size_t) m_width};
    SET_3_KERNEL_ARGS(m_clearKernel, texBuf, m_width, m_height);
    CALL_CL_GUARDED(clEnqueueNDRangeKernel, (m_clQueue, m_clearKernel,
                /* Dimensions */ 2, NULL, gdim, ldim, 0, NULL, NULL));

    CALL_CL_GUARDED(clEnqueueReleaseGLObjects, (m_clQueue, 1,
                &texBuf, 0, NULL, NULL));
    CALL_CL_GUARDED(clFinish, (m_clQueue));
}

void FEMView2D::m_clRenderCSGNode(CSGNode *node, cl_mem texBuf,
                                  const QColor &fg)
{
    cl_int err;
    // timestamp_type start, end;
    CALL_CL_GUARDED(clEnqueueAcquireGLObjects,
            (m_clQueue, 1, &texBuf, 0, NULL, NULL));

    CSGNodeType *nodes = (CSGNodeType *) clEnqueueMapBuffer(m_clQueue,
                        m_nodeHostBuf, CL_TRUE, CL_MAP_WRITE, 0,
                        MAX_NODES * sizeof(CSGNodeType), 0, NULL, NULL, &err);
    CHECK_CL_ERROR(err, "map node buffer");
    CSGPrimitiveData *pdata = (CSGPrimitiveData *) clEnqueueMapBuffer(m_clQueue,
                        m_primHostBuf, CL_TRUE, CL_MAP_WRITE, 0,
                        MAX_PRIMITIVES * sizeof(CSGPrimitiveData),
                        0, NULL, NULL, &err);
    CHECK_CL_ERROR(err, "map prim buffer");
                 
    CSGTreeFlattener flatTree = m_fem.model().dfs(CSGTreeFlattener(nodes, pdata), node);
    int numNodes      = flatTree.numNodes;
    int numPrimitives = flatTree.numPrimitives;

    if ((numNodes > 0) && (numPrimitives > 0)) {
        CALL_CL_GUARDED(clEnqueueWriteBuffer,
                ( m_clQueue, m_nodeBuf, /* Blocking */ CL_FALSE, 0,
                  numNodes * sizeof(CSGNodeType), nodes, 0, NULL, NULL ));
        CALL_CL_GUARDED(clEnqueueWriteBuffer,
                ( m_clQueue, m_primBuf, /* Blocking */ CL_FALSE, 0,
                  numPrimitives * sizeof(CSGPrimitiveData), pdata, 0, NULL, NULL ));

        clEnqueueUnmapMemObject(m_clQueue, m_nodeHostBuf, nodes, 0, NULL, NULL);
        clEnqueueUnmapMemObject(m_clQueue, m_primHostBuf, pdata, 0, NULL, NULL);

        size_t ldim[] = {128, 1};
        size_t gdim[] = {(((size_t) m_height + ldim[0]) / ldim[0]) * ldim[0],
                           (size_t) m_width};

        float minX = m_frameCenter[0] - .5f * m_frameDim[0],
              maxX = m_frameCenter[0] + .5f * m_frameDim[0],
              minY = m_frameCenter[1] - .5f * m_frameDim[1],
              maxY = m_frameCenter[1] + .5f * m_frameDim[1];
        cl_float4 fgColor = {{fg.red() / 255.0f, fg.green() / 255.0f,
                              fg.blue() / 255.0f, fg.alpha() / 255.0f}};

        SET_12_KERNEL_ARGS(m_renderKernel, texBuf, m_width, m_height,
                minX, maxX, minY, maxY, numNodes, numPrimitives,
                m_nodeBuf, m_primBuf, fgColor);

        // get_timestamp(&start);
        CALL_CL_GUARDED(clEnqueueNDRangeKernel, (m_clQueue, m_renderKernel,
                    /* Dimensions */ 2, NULL, gdim, ldim, 0, NULL, NULL));
        CALL_CL_GUARDED(clEnqueueReleaseGLObjects, (m_clQueue, 1,
                    &texBuf, 0, NULL, NULL));

        CALL_CL_GUARDED(clFinish, (m_clQueue));
    }
    // get_timestamp(&end);
    // std::cout << "Kernel ran in " << timestamp_diff_in_seconds(start, end)
    //           << std::endl;
}

void FEMView2D::m_drawObject()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_modelTex);
    drawQuad(0, 0, m_width, m_height);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void FEMView2D::m_drawWorldBox(const BBox_t &b)
{
    Scalar minx, miny, maxx, maxy;
    getScreenCoords(b.minCorner[0], b.minCorner[1], minx, miny);
    getScreenCoords(b.maxCorner[0], b.maxCorner[1], maxx, maxy);
    drawQuad(minx, miny, maxx, maxy);
}

void FEMView2D::m_drawWorldArrow(const Vector &p, const Vector &n,
                                 Scalar length, bool rescale)
{
    // Draw unit vectors "length" pixels long
    Scalar scale = rescale ? length * getPixelSize() : 1.0;
    Vector tip = p + scale * n;

    glBegin(GL_LINES);
        Scalar x, y;
        getScreenCoords(p[0], p[1], x, y);
        glVertex2f(x, y);
        getScreenCoords(tip[0], tip[1], x, y);
        glVertex2f(x, y);
    glEnd();
}

void FEMView2D::m_drawSelectedObjects()
{
    QColor selectedObjectColor(128, 128, 128, 128);

    glBindTexture(GL_TEXTURE_2D, m_overlayTex);
    glEnable(GL_TEXTURE_2D);
    drawQuad(0, 0, m_width, m_height);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    // Draw the bounding boxes for selected objects
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3ub(selectedObjectColor.red(), selectedObjectColor.green(),
               selectedObjectColor.blue());
    for (NodeList::iterator it = m_selectedObjects.begin();
                            it != m_selectedObjects.end(); ++it) {
        m_drawWorldBox((*it)->boundingBox());
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void FEMView2D::m_drawWorldVertex(const Vector &v)
{
    Scalar x, y;
    getScreenCoords(v[0], v[1], x, y);
    glVertex2f(x, y);
}

////////////////////////////////////////////////////////////////////////////////
/*! Draw all elements textured with a rendering of the element's part of the
//  model. Also, color with the passed scalar field, if available.
//  @param[in]  deformation     optional per-node displacement
//  @param[in]  elemScalarField optional per-elem scalar field for shading
//  @return     true if shading was done (and color legend is needed)
*///////////////////////////////////////////////////////////////////////////////
bool FEMView2D::drawObjectTextureCells(const VField &deformation,
                         const SField &elemScalarField)
{
    ElementGrid2D_t &grid = m_fem.elementGrid();
    bool hasDeformation = deformation.domainSize() == grid.numNodes();
    bool hasEScalarField = elemScalarField.domainSize() == grid.numElements();

    if (hasEScalarField) {
        if (m_viewSettings.colormapRangeAuto) {
            m_scalarColorMap.setRange(elemScalarField.min(),
                    elemScalarField.max());
        }
        else {
            m_scalarColorMap.setRange(m_viewSettings.colormapRangeMin,
                    m_viewSettings.colormapRangeMax);
        }
    }
    glUseProgram(m_bilinearShader);
    
    // Set the object texture sampler to be texture unit 0
    glBindTexture(GL_TEXTURE_2D, m_modelTex);
    glUniform1i(m_objectTexLoc, 0);

    glBegin(GL_QUADS);
    for (size_t i = 0; i < grid.numElements(); ++i) {
        ElementGrid2D_t::AdjacencyVec corners;
        grid.elementCorners(i, corners);
        Vector minCorner, maxCorner;
        for (size_t c = 0; c < (size_t) corners.rows(); ++c) {
            Vector p = grid.nodePosition(corners[c]);
            // Map world coordinates to texture coordinates
            Scalar s, t;
            getTextureCoordinates(p[0], p[1], s, t);
            glVertexAttrib2f(m_tCoordLoc[c], s, t);
            
            if (hasDeformation)
                p += deformation(corners[c]);
            
            Vector v;
            getScreenCoords(p[0], p[1], v[0], v[1]);

            glVertexAttrib2f(m_vCoordLoc[c], v[0], v[1]);
            
            // Get bounding box
            if (c == 0) {
                minCorner = maxCorner = v;
            }
            else {
                minCorner = minCorner.cwiseMin(v);
                maxCorner = maxCorner.cwiseMax(v);
            }
        }

        if (hasEScalarField)
            glColor4fv(m_scalarColorMap(elemScalarField[i]));
        else
            glColor3f(0.25f, 0.25f, 0.25f);

        // Draw quad's bounding box
        glVertex2f(minCorner[0], minCorner[1]);
        glVertex2f(maxCorner[0], minCorner[1]);
        glVertex2f(maxCorner[0], maxCorner[1]);
        glVertex2f(minCorner[0], maxCorner[1]);
    }
    glEnd();

    glUseProgram(0);

    // glDisable(GL_TEXTURE_2D);
    return hasEScalarField;
}


void FEMView2D::drawGrid(DrawOp op, const VField &deformation,
                         const SField &elemScalarField)
{
    ElementGrid2D_t &grid = m_fem.elementGrid();
    bool hasDeformation = deformation.domainSize() == grid.numNodes();
    bool hasEScalarField = elemScalarField.domainSize() == grid.numElements();
    ElementGrid2D_t::AdjacencyVec corners;

    if (hasEScalarField) {
        if (m_viewSettings.colormapRangeAuto) {
            m_scalarColorMap.setRange(elemScalarField.min(),
                    elemScalarField.max());
        }
        else {
            m_scalarColorMap.setRange(m_viewSettings.colormapRangeMin,
                    m_viewSettings.colormapRangeMax);
        }
    }

    glColor3f(0, 0, 0);
    if ((op == DRAW_CELLS) || (op == DRAW_EDGES)) {
        if (op == DRAW_EDGES)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBegin(GL_QUADS);
        for (size_t i = 0; i < grid.numElements(); ++i) {
            if (op == DRAW_CELLS) {
                if (hasEScalarField) {
                    glColor4fv(m_scalarColorMap(elemScalarField[i]));
                }
                else {
                    glColor4f(.8f, .8f, .8f, .5f);
                    if (!grid.elementIsFull(i)) 
                        glColor4f(.8f, 0.0f, 0.0f, .5f);
                }
            }
            grid.elementCorners(i, corners);
            for (size_t c = 0; c < (size_t) corners.rows(); ++c) {
                Vector p = grid.nodePosition(corners[c]);
                if (hasDeformation) p += deformation(corners[c]);
                m_drawWorldVertex(p);
            }
        }
        glEnd();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    else if (op == DRAW_NODES) {
        glPointSize(3.0f);
        glBegin(GL_POINTS);
        for (unsigned int i = 0; i < grid.numNodes(); ++i) {
            Vector p = grid.nodePosition(i);
            if (hasDeformation)
                p += deformation(i);
            m_drawWorldVertex(p);
        }
        glEnd();
    }
}

void FEMView2D::drawBoundary(bool pressureField,
        const QColor &color, const QColor &selColor)
{
    // Draw boundary points
    glPointSize(5.0);

    glBegin(GL_POINTS);
    const std::vector<BoundaryPoint_t> &bndPts = m_fem.boundaryPoints();
    glColor3ub(color.red(), color.green(), color.blue());
    for (size_t i = 0; i < bndPts.size(); ++i)
        m_drawWorldVertex(bndPts[i].p);
    // Draw selected boundary point

    bool hasSelect = (m_select.type() == SelectionTool::BOUNDARY) &&
                     (m_select.index() < bndPts.size());
    size_t selIndex = m_select.index();
    if (hasSelect) {
        glColor3ub(selColor.red(), selColor.green(), selColor.blue());
        m_drawWorldVertex(bndPts[selIndex].p);
    }

    glEnd();

    glColor3ub(color.red(), color.green(), color.blue());
    for (size_t i = 0; i < bndPts.size(); ++i) {
        Scalar scale = pressureField ? 800 * m_fem.pressure(i) : 15.0;
        m_drawWorldArrow(bndPts[i].p, scale * bndPts[i].n, 1.0, true);
    }
    
    if (hasSelect) {
        glColor3ub(selColor.red(), selColor.green(), selColor.blue());
        Scalar scale = pressureField ? 800 * m_fem.pressure(selIndex) : 15.0;
        m_drawWorldArrow(bndPts[selIndex].p, scale * bndPts[selIndex].n,
                         1.0, true);
    }
}

void FEMView2D::drawSelection(const VField &deformation)
{
    ElementGrid2D_t &grid = m_fem.elementGrid();
    ElementGrid2D_t::AdjacencyVec corners;
    bool hasDeformation = deformation.domainSize() == grid.numNodes();

    glColor3ub(255, 160, 0);
    if ((m_select.type() == SelectionTool::NODE) &&
        (m_select.index() < grid.numNodes())) {
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        Vector p = grid.nodePosition(m_select.index());
        if (hasDeformation) p += deformation(m_select.index());
        m_drawWorldVertex(p);
        glEnd();
    }

    else if ((m_select.type() == SelectionTool::ELEM) &&
        (m_select.index() < grid.numElements())) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(3.0);
        glBegin(GL_QUADS);
        grid.elementCorners(m_select.index(), corners);
        for (size_t c = 0; c < (size_t) corners.rows(); ++c) {
            Vector p = grid.nodePosition(corners[c]);
            if (hasDeformation) p += deformation(corners[c]);
            m_drawWorldVertex(p);
        }
        glEnd();

        glColor3f(0.0, 0.0, 0.0);
        glLineWidth(1.0);
        glBegin(GL_QUADS);
        grid.elementCorners(m_select.index(), corners);
        for (size_t c = 0; c < (size_t) corners.rows(); ++c) {
            Vector p = grid.nodePosition(corners[c]);
            if (hasDeformation) p += deformation(corners[c]);
            m_drawWorldVertex(p);
        }

        glEnd();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // Draw boundary point selection if there is no deformation (we don't know
    // how to displace boundary points).
    else if ((m_select.type() == SelectionTool::BOUNDARY) &&
             (m_select.index() < m_fem.numBoundaryPoints()) &&
             !hasDeformation) {
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        m_drawWorldVertex(m_fem.boundaryPoints()[m_select.index()].p);
        glEnd();
    }
}

void FEMView2D::m_rerenderObject()
{
    QColor modelColor(128, 192, 255, 255);
    glFinish();
    m_clClearCSGRender(m_modelTexBuf);
    m_clRenderCSGNode(NULL, m_modelTexBuf, modelColor);
}

void FEMView2D::m_rerenderOverlay()
{
    glFinish();
    CSGNode *overlayRoot = NULL;
    std::vector<CSGNode *> unionGlueNodes;
    for (NodeList::iterator it = m_selectedObjects.begin();
                            it != m_selectedObjects.end(); ++it) {
        if (overlayRoot == NULL) {
            overlayRoot = *it;
        }
        else {
            overlayRoot = new CSGGlueNode(overlayRoot, *it);
            unionGlueNodes.push_back(overlayRoot);
        }
    }
    QColor selectedObjectColor(128, 128, 128, 128);

    m_clClearCSGRender(m_overlayTexBuf);
    if (overlayRoot)
        m_clRenderCSGNode(overlayRoot, m_overlayTexBuf, selectedObjectColor);

    for (size_t i = 0; i < unionGlueNodes.size(); ++i) {
        delete unionGlueNodes[i];
    }
}

void FEMView2D::draw()
{
    glColor3f(1, 1, 1);
    drawQuad(0, 0, m_width, m_height);
    
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_POINT_SMOOTH);
    bool usedColormap = false;

    m_activeDeformation.clear();

    ElementGrid2D_t &grid = m_fem.elementGrid();

    if (m_guiState == STATE_MODEL) {
        m_drawObject();
        m_drawSelectedObjects();
    }

    else if (m_guiState == STATE_RESULT) {
        usedColormap |= m_drawResult();
    }
    else if (m_guiState == STATE_ELEMENTS) {
        usedColormap |= m_drawElements();
    }
    else if (m_guiState == STATE_PRESSURE_DRAW) {
        drawObjectTextureCells();
        // drawGrid(DRAW_CELLS);
        drawGrid(DRAW_EDGES);

        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (unsigned int i = 0; i < grid.numNodes(); ++i) {
            glColor3f(0.0f, 0.0f, 0.0f);

            Vector p = grid.nodePosition(i);
            m_drawWorldVertex(p);
        }
        glEnd();

        drawBoundary(true, QColor(255, 160, 0));
    }

    if (usedColormap && m_viewSettings.showColorbar) {
        float colorBarWidth = 300;
        // Horizontally center colorbar
        float colorbarX = .5 * (m_width - colorBarWidth);

        m_drawColorbar(colorbarX, 5, colorBarWidth, 35);
    }
}

bool FEMView2D::m_drawResult()
{
    assert(m_result);

    ElementGrid2D_t &grid = m_fem.elementGrid();

    const SField &sfield = m_result->getScalarField(Result::PER_ELEM);

    size_t numNodes = grid.numNodes();
    size_t numElems = grid.numElements();

    VField vfield = m_result->getVectorField(Result::PER_NODE);
    SField bsfield = m_result->getScalarField(Result::PER_BDRY);

    // The result deformation is recorded in m_activeDeformation
    // so node offsets can be used elsewhere (e.g. selection).
    VField &deformation = m_activeDeformation;

    Scalar vecScale = 1.0;

    Scalar objectSize = grid.getBoudingBox().dimensions().sum() / 2.0;
    if (m_result->hasNodeVField() && m_viewSettings.autofitVectorField) {
        // Scale vector field so that the maximum magnitude is a specified
        // fraction of the object size
        Scalar maxNorm = 0.0;
        assert((size_t) vfield.domainSize() == numNodes);
        for (size_t i = 0; i < numNodes; ++i)
            maxNorm = std::max(vfield(i).norm(), maxNorm);

        vecScale = (maxNorm > 1e-9) ?
            m_viewSettings.autofitMagnitude * (objectSize / maxNorm) : 1.0;
    }

    if (m_viewSettings.vfDisplayStyle == ViewSettings::VFIELD_VIBRATE)
        vfield *= vecScale * sin(m_displacementPhase);
    else
        vfield *= vecScale;

    if (m_viewSettings.vfDisplayStyle == ViewSettings::VFIELD_DEFORM ||
        m_viewSettings.vfDisplayStyle == ViewSettings::VFIELD_VIBRATE) {
        deformation = vfield;
    }

    // Draw the object, posisbly shaded with an element scalar field and
    // deformed by a nodal vector field.
    m_scalarColorMap.setAlpha(0.5f);
    bool usedColormap = drawObjectTextureCells(deformation, sfield);

    // Visualize boundary scalar fields as scaled arrows.
    if (m_result->hasBdrySField()) {
        const std::vector<BoundaryPoint_t> &bndPts = m_fem.boundaryPoints();
        assert((size_t) bsfield.domainSize() == bndPts.size());

        Scalar maxMag = std::max(std::abs(bsfield.max()),
                                 std::abs(bsfield.min()));
        Scalar scale =
            (m_viewSettings.autofitMagnitude * objectSize) / maxMag;

        glColor3ub(255, 160, 0);
        for (size_t i = 0; i < bndPts.size(); ++i)
            m_drawWorldArrow(bndPts[i].p, scale * bsfield[i] * bndPts[i].n);
    }

    // Draw the per-node vector field arrows, if necessary
    if ((m_viewSettings.vfDisplayStyle == ViewSettings::VFIELD_ARROW) &&
        (vfield.domainSize() == numNodes)) {
        glLineWidth(3.0);
        glColor3f(1.0, 1.0, 1.0);
        for (size_t i = 0; i < numNodes; ++i) {
            Vector p = grid.nodePosition(i);
            Vector v = vfield(i);
            m_drawWorldArrow(p, v);
        }

        glLineWidth(1.0);
        glColor3f(0.0, 0.0, 0.0);
        for (size_t i = 0; i < numNodes; ++i) {
            Vector p = grid.nodePosition(i);
            Vector v = vfield(i);
            m_drawWorldArrow(p, v);
        }
    }

    if (m_viewSettings.showGridOverResults) {
        drawGrid(DRAW_EDGES, deformation);
        drawGrid(DRAW_NODES, deformation);
    }

    drawSelection(deformation);

    ////////////////////////////////////////////////////////////////////////////
    // Display selected results.
    ////////////////////////////////////////////////////////////////////////////
    using boost::format;
    std::string resultString;
    if (m_result->hasNodeVField() && (m_select.type() == SelectionTool::NODE)) {
        size_t i = m_select.index();
        assert(i < vfield.domainSize());
        Vector v = vfield(i);
        resultString = boost::str(format(
            "Node %i vector: [%lf, %lf] (mag: %lf)") %
            (int) i % v[0] % v[1] % v.norm());
    }

    bool hasElemScalarField = sfield.domainSize() == numElems;
    if (hasElemScalarField && (m_select.type() == SelectionTool::ELEM)) {
        size_t i = m_select.index();
        assert(i < sfield.domainSize());
        resultString = boost::str(format("Elem %i scalar: %lf") 
                % (int) i % sfield[i]);
    }

    if (m_result->hasBdrySField() &&
        (m_select.type() == SelectionTool::BOUNDARY)) {
        size_t i = m_select.index();
        assert(i < bsfield.domainSize());
        resultString = boost::str(format("Boundary point %i scalar: %lf")
                % (int) i % bsfield[i]);
    }

    if (resultString.length()) {
        glColor3f(0.0f, 0.0f, 0.0f);
        glRasterPos2i(5, 5);
        m_font.Render(resultString.c_str());
    }

    return usedColormap;
}

bool FEMView2D::m_drawElements()
{
    ElementGrid2D_t &grid = m_fem.elementGrid();

    drawObjectTextureCells();
    drawGrid(DRAW_CELLS);
    drawGrid(DRAW_EDGES);
    drawGrid(DRAW_NODES);

    // // Draw non-element cells
    // glColor4f(0.0, 0.0, 0.0, .25f);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // for (size_t i = 0; i < grid.numCells(); ++i) {
    //     m_drawWorldBox(grid.cellBoundingBox(i));
    // }
    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (m_viewSettings.showQuadraturePoints) {
        // Draw quadrature points
        glPointSize(1.0f);
        glColor3f(1.0, 1.0, 0);
        glBegin(GL_POINTS);

        std::vector<Vector> qPoints;
        for (unsigned int i = 0; i < grid.numElements(); ++i) {
            BBox_t b = grid.elementBoundingBox(i);
            m_fem.quadrature().quadraturePoints(b, qPoints);
            for (unsigned int p = 0; p < qPoints.size(); ++p) {
                m_drawWorldVertex(qPoints[p]);
            }
        }
        glEnd();
    }

    drawBoundary();

    // Visualize cubic kernel around selected point
    if  ((m_select.type() == SelectionTool::BOUNDARY) &&
        (m_select.index() < m_fem.boundaryPoints().size())) {
        const MeshlessFEM_t::BoundaryFunction &phi =
            m_fem.boundaryFunction(m_select.index());
        Scalar radius = phi.supportRadius();
        // Highlight all elements overlapping the basis function's support
        std::vector<size_t> elems;
        grid.elementsAroundPoint(phi.center(), radius, elems);
        glColor4f(1.0f, 1.0f, 1.0f, 0.5f);
        for (size_t i = 0; i < elems.size(); ++i)
            m_drawWorldBox(grid.elementBoundingBox(elems[i]));
        
        // Draw a sub-grid of quads around the point, spanning
        // the full basis function's support
        glBegin(GL_QUADS);
            #define KERNEL_VIS_SUBDIV 10
            Scalar subdivWidth = 2 * radius / KERNEL_VIS_SUBDIV;
            Scalar scale = phi.maxNormalizationFactor();
            for (int i = 0; i < KERNEL_VIS_SUBDIV; ++i) {
                Scalar minY = phi.center()[1] - radius +
                              subdivWidth * i;
                Scalar maxY = minY + subdivWidth;
                for (int j = 0; j < KERNEL_VIS_SUBDIV; ++j) {
                    Scalar minX = phi.center()[0] - radius +
                                  subdivWidth * j;
                    Scalar maxX = minX + subdivWidth;

                    Scalar x, y;
                    glColor4f(0.0f, 1.0f, 0.0f,
                              scale * phi(Vector(minX, minY)));
                    getScreenCoords(minX, minY, x, y);
                    glVertex2f(x, y);
                    glColor4f(0.0f, 1.0f, 0.0f,
                              scale * phi(Vector(maxX, minY)));
                    getScreenCoords(maxX, minY, x, y);
                    glVertex2f(x, y);
                    glColor4f(0.0f, 1.0f, 0.0f,
                              scale * phi(Vector(maxX, maxY)));
                    getScreenCoords(maxX, maxY, x, y);
                    glVertex2f(x, y);
                    glColor4f(0.0f, 1.0f, 0.0f,
                              scale * phi(Vector(minX, maxY)));
                    getScreenCoords(minX, maxY, x, y);
                    glVertex2f(x, y);
                }
            }
        glEnd();
    }

    drawSelection();

    return false;
}

void FEMView2D::m_drawColorbar(float x, float y, float width, float height)
{
    // Draw background box
    glColor4f(1.0f, 1.0f, 1.0f, .5f);
    glBegin(GL_QUADS);
        glVertex2f(x, y);
        glVertex2f(x + width, y);
        glVertex2f(x + width, y + height);
        glVertex2f(x, y + height);
    glEnd();
    
    std::stringstream ss;
    ss << m_scalarColorMap.getRangeMin();
    std::string rangeMin = ss.str();
    ss.str("");
    ss.clear();
    ss << m_scalarColorMap.getRangeMax();
    std::string rangeMax = ss.str();

    FTBBox bbox = m_font.BBox(rangeMin.c_str());
    float lowTextWidth  = bbox.Upper().X() - bbox.Lower().X();
    float textHeight = bbox.Upper().Y() - bbox.Lower().Y();

    bbox = m_font.BBox(rangeMax.c_str());
    float highTextWidth = bbox.Upper().X() - bbox.Lower().X();

    // Vertically center text within height.
    // Horizontal margins on text, with colorbar filling the rest
    float textMargin = 5;
    float barWidth = width - 4 * textMargin - lowTextWidth - highTextWidth;
    float barVMargin = 5;
    float barHeight = height - 2 * barVMargin;
    float textY = y + .5 * (height - textHeight);

    // Note: glRasterPos2i must be used to apply glColor3;
    glColor3f(0.0f, 0.0f, 0.0f);
    glRasterPos2i(x + textMargin, textY);
    m_font.Render(rangeMin.c_str());
    glRasterPos2i(x + 3 * textMargin + lowTextWidth + barWidth, textY);
    m_font.Render(rangeMax.c_str());
    
    float barX = x + 2 * textMargin + lowTextWidth;
    float barY = y + barVMargin;
    int numSegments = 100;
    float segmentWidth = barWidth / numSegments;
    glBegin(GL_QUADS);
        for (int i = 0; i < numSegments; ++i) {
            float segmentStart = barX + segmentWidth * i;
            float segmentEnd = barX + segmentWidth * (i + 1);
            float normalizedValue = i / ((float) numSegments);
            glColor3fv(m_scalarColorMap.normalizedValueColor(normalizedValue));
            glVertex2f(segmentStart, barY + barHeight);
            glVertex2f(segmentStart, barY);
            glVertex2f(segmentEnd  , barY);
            glVertex2f(segmentEnd  , barY + barHeight);
        }
    glEnd();
}


void FEMView2D::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw();
}

////////////////////////////////////////////////////////////////////////////////
// Selection/painting tools
////////////////////////////////////////////////////////////////////////////////
// Pixel-space distance threshold for selection/painting
#define SELECT_DIST_THRESHOLD 30.0

// Gets the closest point in a collection and its distance in screen coordinates
template<typename Collection>
void FEMView2D::getClosest(const Collection &points, const Vector &pt,
                           size_t &closest, Scalar &sDist) {
    closest = 0;
    sDist = std::numeric_limits<Scalar>::max();
    for (size_t i = 0; i < points.size(); ++i) {
        Vector candidate;
        getScreenCoords(points(i), candidate);
        Scalar dist = (pt - candidate).norm();
        if (dist < sDist) {
            closest = i;
            sDist = dist;
        }
    }
}

// Selection points are boundary points 
struct BPCollect {
    BPCollect(const std::vector<BoundaryPoint_t> &bp) : m_bp(bp) { }
    size_t size() const { return m_bp.size(); }
    Vector operator()(size_t i) const { assert(i < size()); return m_bp[i].p; }
private:
    const std::vector<BoundaryPoint_t> &m_bp;
};

// Selection points are nodes 
struct NDCollect {
    typedef MeshlessFEM_t::VField VField;
    NDCollect(const ElementGrid2D_t &grid,
              const VField &deformation = VField())
        : m_grid(grid), m_deformation(deformation) { }
    size_t size() const { return m_grid.numNodes(); }

    Vector operator()(size_t i) const {
        return (m_deformation.domainSize() == size())
            ? m_grid.nodePosition(i) + m_deformation(i)
            : m_grid.nodePosition(i);
    }

private:
    const ElementGrid2D_t &m_grid;
    const VField &m_deformation;
};

// Selection points are element barycenters 
struct ELCollect {
    typedef MeshlessFEM_t::VField VField;
    ELCollect(const ElementGrid2D_t &grid,
              const VField &deformation = VField())
        : m_grid(grid), m_deformation(deformation) { }
    size_t size() const { return m_grid.numElements(); }
    Vector operator()(size_t i) const {
        ElementGrid2D_t::AdjacencyVec corners;
        m_grid.elementCorners(i, corners);

        Vector center(Vector::Zero());
        Scalar weight = 1.0 / corners.rows();
        bool hasDeformation = m_deformation.domainSize() == m_grid.numNodes();
        for (size_t c = 0; c < (size_t) corners.rows(); ++c) {
            center += weight * m_grid.nodePosition(corners[c]);
            if (hasDeformation) center += weight * m_deformation(corners[c]);
        }

        return center;
    }

private:
    const ElementGrid2D_t &m_grid;
    const VField &m_deformation;
};

void FEMView2D::paintPressure(const Vector &screenPt, bool erase)
{
    size_t closest = 0;
    Scalar closestDist;
    getClosest(BPCollect(m_fem.boundaryPoints()), screenPt,
              closest, closestDist);

    if (closestDist < SELECT_DIST_THRESHOLD) {
        m_fem.pressure(closest) = erase ? 0.0 : m_pressurePaintValue;
        update();
    }
}

void FEMView2D::performSelection(const Vector &screenPt)
{
    size_t closest = 0;
    Scalar closestDist = std::numeric_limits<Scalar>::max();

    if (m_select.mode() == SelectionTool::NODE) {
        getClosest(NDCollect(m_fem.elementGrid(), m_activeDeformation),
                   screenPt, closest, closestDist);
    }
    else if (m_select.mode() == SelectionTool::ELEM) {
        getClosest(ELCollect(m_fem.elementGrid(), m_activeDeformation),
                   screenPt, closest, closestDist);
    }
    else if (m_select.mode() == SelectionTool::BOUNDARY) {
        getClosest(BPCollect(m_fem.boundaryPoints()), screenPt,
                   closest, closestDist);
    }
    if (closestDist < SELECT_DIST_THRESHOLD)
        m_select.select(closest);
    else
        m_select.clear();

    update();
}

void FEMView2D::wheelEvent(QWheelEvent *event)
{
    // Expand around the mouse position (keep it fixed)
    Vector pos, wPos;
    qtToScreenCoords(event->pos(), pos);
    getWorldCoords(pos[1], pos[0], wPos[0], wPos[1]);

    // delta() returns eiths of degrees,
    // scroll wheels are usually quantized into 15-degree increments.
    float degrees = event->delta() / 8;
    float scale = pow(1.25, degrees / 15);
    // Adjust center to keep "wPos" at the same screen coordinates
    m_frameCenter = wPos + scale * (m_frameCenter - wPos);
    m_frameDim *= scale;

    m_rerenderObject();
    if (m_guiState == STATE_MODEL)
        m_rerenderOverlay();
    update();

    event->accept();
}

void FEMView2D::mouseReleaseEvent(QMouseEvent *event)
{
    qtToScreenCoords(event->pos(), m_prevMouseLoc);
    m_gesture = GESTURE_NONE;
}

void FEMView2D::mousePressEvent(QMouseEvent *event)
{
    if (m_guiState == STATE_MODEL) {
        if (event->button() == Qt::LeftButton) {
            qtToScreenCoords(event->pos(), m_prevMouseLoc);
            m_gesture = GESTURE_DRAG;
        }
    }
    else if (m_guiState == STATE_PRESSURE_DRAW) {
        if (event->button() == Qt::LeftButton) {
            bool erase = event->modifiers() & Qt::AltModifier;
            paintPressure(qtToScreenCoords(event->pos()), erase);
        }
    }
    else if ((m_guiState == STATE_ELEMENTS) || (m_guiState == STATE_RESULT)) {
        if (event->button() == Qt::LeftButton)
            performSelection(qtToScreenCoords(event->pos()));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Navigation gestures should work in all modes
    ////////////////////////////////////////////////////////////////////////////
    if (event->button() == Qt::MiddleButton) {
        qtToScreenCoords(event->pos(), m_prevMouseLoc);
        m_gesture = GESTURE_ZOOM;
    }
    if (event->button() == Qt::RightButton) {
        qtToScreenCoords(event->pos(), m_prevMouseLoc);
        m_gesture = GESTURE_PAN;
    }
}

void FEMView2D::mouseMoveEvent(QMouseEvent *event)
{
    Vector start, end;
    getWorldCoords(m_prevMouseLoc[1], m_prevMouseLoc[0], start[0], start[1]);
    Vector endScreen;
    qtToScreenCoords(event->pos(), endScreen);
    getWorldCoords(endScreen[1], endScreen[0], end[0], end[1]);
    bool handled = false;

    ////////////////////////////////////////////////////////////////////////////
    // Navigation gestures should work in all modes
    ////////////////////////////////////////////////////////////////////////////
    if (m_gesture == GESTURE_PAN) {
        // Coordinate system translates with inverse of pan
        m_frameCenter += start - end;
        m_rerenderObject();
        if (m_guiState == STATE_MODEL)
            m_rerenderOverlay();
        update();
        handled = true;
    }
    if (m_gesture == GESTURE_ZOOM) {
        float deltaYpx = endScreen[1] - m_prevMouseLoc[1];
        m_frameDim *= pow(1.01, deltaYpx);
        m_rerenderObject();
        if (m_guiState == STATE_MODEL)
            m_rerenderOverlay();
        update();
        handled = true;
    }

    if (!handled) {
        bool leftButton  = event->buttons() & Qt::LeftButton;

        if ((m_guiState == STATE_MODEL) && m_gesture == GESTURE_DRAG) {
            for (NodeList::iterator it = m_selectedObjects.begin();
                                    it != m_selectedObjects.end(); ++it) {
                (*it)->applyTranslation(end - start);
            }
            m_rerenderObject();
            m_rerenderOverlay();
            update();
        }
        else if (m_guiState == STATE_PRESSURE_DRAW) {
            if (leftButton) {
                bool erase = event->modifiers() & Qt::AltModifier;
                paintPressure(qtToScreenCoords(event->pos()), erase);
            }
        }
        else if ((m_guiState == STATE_ELEMENTS) || (m_guiState == STATE_RESULT))
        {
            if (leftButton)
                performSelection(qtToScreenCoords(event->pos()));
        }
    }

    m_prevMouseLoc = endScreen;
}

void FEMView2D::mouseDoubleClickEvent(QMouseEvent *event)
{
    Q_UNUSED(event)
}
