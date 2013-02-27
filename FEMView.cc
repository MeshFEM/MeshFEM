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
#include <algorithm>
#include <cstdlib>

#include "MeshlessFEM.hh"
#include "ShaderCompiler.hh"
#include "timing.h"

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


FEMView2D::FEMView2D(MeshlessFEM_t &fem, QWidget *parent)
    : QGLWidget(parent), m_frameMin(-2, -1.5), m_frameMax(2, 1.5),
      m_fem(fem), m_guiState(MODEL_STATE), m_gesture(NONE),
      m_displacementPhase(0.0), m_showGridWhileDeforming(true),
      m_modelTexBuf(NULL), m_overlayTexBuf(NULL)
{
    setFormat(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer));
    setFocusPolicy(Qt::StrongFocus);
}

void FEMView2D::csgNodesSelected(const NodeList &nList)
{
    m_selectedObjects = nList;
    m_rerenderOverlay();
    update();
}

void FEMView2D::showGridDuringDeformationToggled(bool showGrid)
{
    m_showGridWhileDeforming = showGrid;
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
    float aspect = (m_frameMax[0] - m_frameMin[0]) /
                   (m_frameMax[1] - m_frameMin[1]);
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
    void preVisit(CSGNode *node) { } 
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
            p.center.x        = r->getCenter()[0];
            p.center.y        = r->getCenter()[1];
            p.rect.half_dim.x = dim[0];
            p.rect.half_dim.y = dim[1];
            p.rect.rotationCosSin.x = cos(-r->getRotationRad());
            p.rect.rotationCosSin.y = sin(-r->getRotationRad());
        }

        else if (type == CSG_NODE_ELLIPSE) {
            assert(numPrimitives < MAX_PRIMITIVES);
            CSGPrimitiveData &p = primitiveData[numPrimitives++];
            CSGEllipseNode *e = dynamic_cast<CSGEllipseNode *>(node);
            assert(e);
            Vector focus = e->getFocus();
            p.center.x                   = e->getCenter()[0];
            p.center.y                   = e->getCenter()[1];
            p.ellipse.focus.x            = focus[0];
            p.ellipse.focus.y            = focus[1];
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
    size_t gdim[] = {((m_height + ldim[0]) / ldim[0]) * ldim[0],
        m_width};
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

    CALL_CL_GUARDED(clEnqueueWriteBuffer,
            ( m_clQueue, m_nodeBuf, /* Blocking */ CL_FALSE, 0,
              numNodes * sizeof(CSGNodeType), nodes, 0, NULL, NULL ));
    CALL_CL_GUARDED(clEnqueueWriteBuffer,
            ( m_clQueue, m_primBuf, /* Blocking */ CL_FALSE, 0,
              numPrimitives * sizeof(CSGPrimitiveData), pdata, 0, NULL, NULL ));

    clEnqueueUnmapMemObject(m_clQueue, m_nodeHostBuf, nodes, 0, NULL, NULL);
    clEnqueueUnmapMemObject(m_clQueue, m_primHostBuf, pdata, 0, NULL, NULL);

    size_t ldim[] = {128, 1};
    size_t gdim[] = {((m_height + ldim[0]) / ldim[0]) * ldim[0], m_width};

    float minX = m_frameMin[0], maxX = m_frameMax[0],
          minY = m_frameMin[1], maxY = m_frameMax[1];
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
    // get_timestamp(&end);
    // std::cout << "Kernel ran in " << timestamp_diff_in_seconds(start, end) << std::endl;
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
    glColor3i(selectedObjectColor.red(), selectedObjectColor.green(),
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

void FEMView2D::drawObjectTextureCells(const VField &deformation)
{
    ElementGrid2D_t &grid = m_fem.elementGrid();
    bool hasDeformation = deformation.domainSize() == grid.numNodes();
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

        // Draw quad's bounding box
        glVertex2f(minCorner[0], minCorner[1]);
        glVertex2f(maxCorner[0], minCorner[1]);
        glVertex2f(maxCorner[0], maxCorner[1]);
        glVertex2f(minCorner[0], maxCorner[1]);
    }
    glEnd();

    glUseProgram(0);

    // glDisable(GL_TEXTURE_2D);
}

void FEMView2D::drawGrid(DrawOp op, const VField &deformation)
{
    ElementGrid2D_t &grid = m_fem.elementGrid();
    bool hasDeformation = deformation.domainSize() == grid.numNodes();
    ElementGrid2D_t::AdjacencyVec corners;

    glColor3f(0, 0, 0);
    if ((op == DRAW_CELLS) || (op == DRAW_EDGES)) {
        if (op == DRAW_EDGES)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBegin(GL_QUADS);
        for (size_t i = 0; i < grid.numElements(); ++i) {
            if (op == DRAW_CELLS) {
                glColor4f(.8f, .8f, .8f, .5f);
                if (!grid.elementIsFull(i)) 
                    glColor4f(.8f, 0.0f, 0.0f, .5f);
            }
            grid.elementCorners(i, corners);
            for (size_t c = 0; c < (size_t) corners.rows(); ++c) {
                Vector p = grid.nodePosition(corners[c]);
                if (hasDeformation)
                    p += deformation(corners[c]);
                m_drawWorldVertex(p);
            }
        }
        glEnd();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    else if (op == DRAW_NODES) {
        glPointSize(5.0f);
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

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_POINT_SMOOTH);

    if (m_guiState == MODEL_STATE) {
        m_drawObject();
        m_drawSelectedObjects();
    }
    else if (m_guiState == DISPLACEMENTS_STATE) {
        VField deformation;
        if (m_selectedDeformation < m_fem.numModes()) {
            deformation = m_fem.mode(m_selectedDeformation);
            // Scale deformation so that the maximum displacement doesn't exceed
            // a fraction of the window size
            Scalar relMag = .125;
            Scalar maxX = 0.0, maxY = 0.0;
            size_t numNodes = m_fem.elementGrid().numNodes();
            assert((size_t) deformation.domainSize() == numNodes);
            for (size_t i = 0; i < numNodes; ++i) {
                maxX = std::max((Scalar) std::abs(deformation(i)[0]), maxX);
                maxY = std::max((Scalar) std::abs(deformation(i)[1]), maxY);
            }

            Vector frameDim = m_frameMax - m_frameMin;
            Scalar xMag = (maxX > 1e-6) ? relMag * (frameDim[0] / maxX) : 1.0;
            Scalar yMag = (maxY > 1e-6) ? relMag * (frameDim[1] / maxY) : 1.0;

            Scalar magnitude = std::min(xMag, yMag);

            deformation *= magnitude * sin(m_displacementPhase);
        }
        // if (m_shadeWithStress) {
        //     const SField &stressNorms =
        //             m_fem.modalStressNorms(m_selectedDeformation);
        //     drawGrid(DRAW_CELLS, deformation);
        // }
        drawObjectTextureCells(deformation);
        if (m_showGridWhileDeforming) {
            drawGrid(DRAW_EDGES, deformation);
            drawGrid(DRAW_NODES, deformation);
        }
    }
    else if (m_guiState == ELEMENTS_STATE) {
        drawObjectTextureCells();
        drawGrid(DRAW_CELLS);
        drawGrid(DRAW_EDGES);
        drawGrid(DRAW_NODES);

        // Draw quadrature points
        glPointSize(2.0f);
        glColor3f(1.0, 1.0, 0);
        glBegin(GL_POINTS);

        ElementGrid2D_t &grid = m_fem.elementGrid();
        for (unsigned int i = 0; i < grid.numElements(); ++i) {
            BBox_t b = grid.elementBoundingBox(i);
            std::vector<Vector> qpoints =
                m_fem.quadrature().quadraturePoints(b);
            for (unsigned int p = 0; p < qpoints.size(); ++p) {
                m_drawWorldVertex(qpoints[p]);
            }
        }
        glEnd();
    }
}

void FEMView2D::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw();
}

void FEMView2D::mouseReleaseEvent(QMouseEvent *event)
{
    m_prevMouseLoc = event->pos();
    m_gesture = NONE;
}

void FEMView2D::mousePressEvent(QMouseEvent *event)
{
    if (m_guiState == MODEL_STATE) {
        m_prevMouseLoc = event->pos();
        m_gesture = DRAGGING;
    }
}

void FEMView2D::mouseMoveEvent(QMouseEvent *event)
{
    Vector start, end;
    getWorldCoords(-m_prevMouseLoc.y(), m_prevMouseLoc.x(), start[0], start[1]);
    getWorldCoords(-event->pos().y(), event->pos().x(), end[0], end[1]);
    if ((m_guiState == MODEL_STATE) && m_gesture == DRAGGING) {
        for (NodeList::iterator it = m_selectedObjects.begin();
                                it != m_selectedObjects.end(); ++it) {
            (*it)->applyTranslation(end - start);
        }
        m_rerenderObject();
        m_rerenderOverlay();
        update();
    }
    m_prevMouseLoc = event->pos();
}

void FEMView2D::mouseDoubleClickEvent(QMouseEvent *event)
{
    
}
