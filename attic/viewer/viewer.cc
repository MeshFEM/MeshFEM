////////////////////////////////////////////////////////////////////////////////
// viewer.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      2D OpenGL viewer for FEM2D.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/04/2012 01:16:50
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <limits>
#include <fstream>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include <Eigen/Dense>

#include "viewer.hh"
#include "UIController.hh"
#include "MeshIO.hh"
#include "draw.hh"
#include "Geometry.hh"
#include "ShaderCompiler.hh"
// #define CG_ANALYSIS
#include "LaplaceSmoother.hh"
#include "pnmutil.h"
#include "util.h"

using namespace std;
using namespace MESH_IO;

typedef Point2D::Scalar Real;

vector<TriangleIndex> triangles;
vector<Point2D>       vertices;
// The vertices at which the exact scalar field was originally interpolated
vector<Point2D>       interp_vertices;

// The old mesh so we can revert...
vector<TriangleIndex> orig_triangles;
vector<Point2D>       orig_vertices;
// Whether a vertex is fixed
vector<bool>          isFixedVertex;

LMesh *mesh = NULL;

/** Selection-related data (Also accessed by SettingsBar.cc) */
UIController g_uiController; 

/** Shader programs for piecewise polynomial interpolation and an exact scalar
 * field HSV shading */
GLuint lagrangeShader = 0, exactShader = 0;
GLuint ls_cornerValuesLoc, ls_edgeValuesLoc, ls_colorScaleLoc, ls_baryCoordLoc,
       ls_degreeLoc;
GLuint es_freqLoc, es_colorScaleLoc;

void drawTriangle2D(const TriangleIndex &tri)
{
    glVertex2f(vertices[tri[0]][0], vertices[tri[0]][1]);
    glVertex2f(vertices[tri[1]][0], vertices[tri[1]][1]);
    glVertex2f(vertices[tri[2]][0], vertices[tri[2]][1]);
}

void drawSelectedTriangle()
{
    if (g_uiController.selectedTriangle < 0)
        return;

    glBegin(GL_TRIANGLES);
    TriangleIndex tri = triangles[g_uiController.selectedTriangle];
    glColor3ub(242, 135, 5);
    drawTriangle2D(tri);
    glEnd();

    // Draw incircle and circumscribed circle
    Point2D center;
    Real r;
    Incircle(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]],
             center, r);
    glColor3f(1.0f, 1.0f, 1.0f);
    drawCircle(center[0], center[1], r, 200);

    Real R;
    Circumcircle(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]],
             center, R);
    glColor3f(1.0f, 0.0f, 0.0f);
    drawCircle(center[0], center[1], R, 200);

    Real cond = .5 * R / r;
    if (!g_uiController.hideText) {
        stringstream ss;
        ss << "Triangle " << g_uiController.selectedTriangle
            << " condition number: " << cond;
        glColor3f(0.0f, 0.0f, 0.0f);
        drawString(1, 1, ss.str().c_str());
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! Called by GLUT when redisplay needed
*///////////////////////////////////////////////////////////////////////////////
void Display()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);

    glDisable(GL_LIGHTING);

    // Use antialiasing
    glEnable(GL_BLEND);

    // glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POINT_SMOOTH);

    // Apply rotation/zoom
    glPushMatrix();
    g_uiController.applyViewTransforms();

    // Clear drawing buffers
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (g_uiController.shadeStyle != UIController::SHADE_NONE) {
        if (g_uiController.shadeStyle == UIController::SHADE_WEIGHTS) {
            glUseProgram(lagrangeShader);
            glUniform1i(ls_degreeLoc, g_uiController.degree);
            glUniform2f(ls_colorScaleLoc, 0.0f, 1.0f);
        } else if (g_uiController.shadeStyle == UIController::SHADE_EXACT) {
            glUseProgram(exactShader);
            // Based off sin(kx) * sin(ky)
            glUniform2f(es_colorScaleLoc, -1.0f, 1.0f);
            glUniform1f(es_freqLoc, g_uiController.freq);
        } else if (g_uiController.shadeStyle == UIController::SHADE_INTERP) {
            glUseProgram(lagrangeShader);
            glUniform1i(ls_degreeLoc, g_uiController.degree);
            glUniform2f(ls_colorScaleLoc, -1.0f, 1.0f);
        }
        
        glBegin(GL_TRIANGLES);

        Point3D cornerValues, edgeValues;
        for (unsigned int i = 0; i < triangles.size(); ++i) {
            TriangleIndex tri = triangles[i];
            if (g_uiController.shadeStyle == UIController::SHADE_WEIGHTS) {
                for (unsigned int j = 0; j < 3; ++j) {
                    cornerValues[j] =
                            (g_uiController.selectedVertex == (int) tri[j]);
                    int hj   = mesh->halfedge_index(tri[(j + 1) % 3],
                                                    tri[(j + 2) % 3]);
                    int hjOp = mesh->halfedge_index(tri[(j + 2) % 3],
                                                    tri[(j + 1) % 3]);
                    edgeValues[j] = ((size_t) g_uiController.selectedHalfedge < mesh->halfedge_size()) &&
                                    ((g_uiController.selectedHalfedge == hj)
                                  || (g_uiController.selectedHalfedge == hjOp));
                    glVertexAttrib3f(ls_edgeValuesLoc, edgeValues[0], edgeValues[1],
                            edgeValues[2]);
                    glVertexAttrib3f(ls_cornerValuesLoc, cornerValues[0],
                            cornerValues[1], cornerValues[2]);
                }
            }
            else if (g_uiController.shadeStyle == UIController::SHADE_INTERP) {
                mesh->quadraticNodeValues(i, cornerValues, edgeValues);
                glVertexAttrib3f(ls_edgeValuesLoc, edgeValues[0], edgeValues[1],
                        edgeValues[2]);
                glVertexAttrib3f(ls_cornerValuesLoc, cornerValues[0],
                        cornerValues[1], cornerValues[2]);
            }

            glVertexAttrib3f(ls_baryCoordLoc, 1.0f, 0.0f, 0.0f);
            glVertex2f(vertices[tri[0]][0], vertices[tri[0]][1]);
            glVertexAttrib3f(ls_baryCoordLoc, 0.0f, 1.0f, 0.0f);
            glVertex2f(vertices[tri[1]][0], vertices[tri[1]][1]);
            glVertexAttrib3f(ls_baryCoordLoc, 0.0f, 0.0f, 1.0f);
            glVertex2f(vertices[tri[2]][0], vertices[tri[2]][1]);
        }
        glEnd();
        glUseProgram(0);
    }

    drawSelectedTriangle();

    glColor3f(0.0f, 0.0f, 0.0f);
    // Overlay wireframe
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBegin(GL_TRIANGLES);
    for (unsigned int i = 0; i < triangles.size(); ++i)
        drawTriangle2D(triangles[i]);
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glPointSize(10.0f);
    glColor3ub(242, 135, 5);
    glBegin(GL_POINTS);
    // Highlight fixed vertices
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        if (isFixedVertex[i])
            glVertex2f(vertices[i][0], vertices[i][1]);
    }

    // Highlight selected vertex
    glColor4ub(147, 166, 5, 128);
    if (g_uiController.selectedVertex > -1) {
        glVertex2f(vertices[g_uiController.selectedVertex][0],
                   vertices[g_uiController.selectedVertex][1]);
    }
    glEnd();

    // Display conditioning Statistics
    double totalCond = 0, minCond, maxCond;
    maxCond = minCond = cond(triangles[0], vertices);
    for (unsigned int i = 0; i < triangles.size(); ++i) {
        double cn = cond(triangles[i], vertices);
        totalCond += cn;
        minCond = min(minCond, cn);
        maxCond = max(maxCond, cn);
    }
    if (!g_uiController.hideText) {
        stringstream ss;
        ss << "Average, Max, Min: " << totalCond / triangles.size() << ", "
           << minCond << ", " << maxCond;
        glColor3f(0, 0, 0);
        drawString(1, g_uiController.height - 10, ss.str().c_str());
    }

    glPopMatrix();
    if (g_uiController.screenshotRequested) {
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        unsigned int width  = viewport[2];
        unsigned int height = viewport[3];
        unsigned char *pixelBuffer = (unsigned char *) malloc(width * height * 4);
        // Read RGBA for proper mod 4 alignment...
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                     pixelBuffer);
        // Strip alpha component
        unsigned char *pixelBufferRGB = (unsigned char *) malloc(width * height * 3);
        for (unsigned int i = 0; i < width * height; ++i)
            for (unsigned int c = 0; c < 3; ++c)
                pixelBufferRGB[3 * i + c] = pixelBuffer[4 * i + c];
        ppmFlipVertical(pixelBufferRGB, width, height);
        ppmWrite(nextNewFile("screenshot_", ".ppm").c_str(), pixelBufferRGB, width,
                 height);
        free(pixelBuffer);
        free(pixelBufferRGB);
        g_uiController.screenshotRequested = false;
    }

    glutSwapBuffers();
}

////////////////////////////////////////////////////////////////////////////////
/*! Repositions a particular vertex
*///////////////////////////////////////////////////////////////////////////////
void positionVertex(size_t vidx, const Point2D &pt)
{
    vertices[vidx] = Point2D(pt[0], pt[1]);
    mesh->updateBarycentricCoordinates(triangles, vertices);
}

////////////////////////////////////////////////////////////////////////////////
/*! Repositions all vertices (by default, to their original positions)
*///////////////////////////////////////////////////////////////////////////////
void positionVertices(const std::vector<Point2D> &reset_verts)
{
    vertices = reset_verts;
    mesh->updateBarycentricCoordinates(triangles, vertices);
}

////////////////////////////////////////////////////////////////////////////////
/*! Toggles whether a particular vertex is fixed
*///////////////////////////////////////////////////////////////////////////////
void toggleFixed(size_t vidx)
{
    assert (vidx < isFixedVertex.size());
    isFixedVertex[vidx] = (isFixedVertex[vidx] == false);
}

////////////////////////////////////////////////////////////////////////////////
/*! Toggles whether the boundary vertices are fixed.
*///////////////////////////////////////////////////////////////////////////////
void toggleFixedBoundary()
{
    bool firstOne = true;
    bool value = true;
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        if (mesh->vertex(i)->isBoundary()) {
            if (firstOne) {
                value = !isFixedVertex[i];
                firstOne = false;
            }
            isFixedVertex[i] = value;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! Runs the laplacian smoother on the mesh
*///////////////////////////////////////////////////////////////////////////////
void smooth()
{
    std::vector<Point2D> solution(vertices.size());
    LaplacianSmoother(triangles, vertices, *mesh, isFixedVertex, solution);
    positionVertices(solution);
}

////////////////////////////////////////////////////////////////////////////////
/*! Runs one step of CG iteration of the laplacian smoother on the mesh
*///////////////////////////////////////////////////////////////////////////////
void cgSmooth()
{
    std::vector<Point2D> solution(vertices.size());
    LaplacianSmoother(triangles, vertices, *mesh, isFixedVertex, solution, 1);
    positionVertices(solution);
}

////////////////////////////////////////////////////////////////////////////////
/*! Samples the function:
//      sin(2 * pi * freq * x) * sin(2 * pi * freq * y)
//  at the lagrange interpolation nodes.
*///////////////////////////////////////////////////////////////////////////////
void interpolateScalarField()
{
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        double x = vertices[i][0], y = vertices[i][1];
        mesh->vertex(i)->c[0] = sin(2.0 * M_PI * g_uiController.freq * x) *
                                sin(2.0 * M_PI * g_uiController.freq * y);
    }
    for (unsigned int i = 0; i < mesh->halfedge_size(); ++i) {
        LMesh::Halfedge *h = mesh->halfedge(i);
        // Sample at the edge midpoint
        Point2D p = .5 * (vertices[mesh->vertex_index(h->tip())] + 
                          vertices[mesh->vertex_index(h->opposite()->tip())]);
        h->edgeData().c[0] = sin(2.0 * M_PI * g_uiController.freq * p[0]) *
                             sin(2.0 * M_PI * g_uiController.freq * p[1]);
    }
    interp_vertices = vertices;
}

////////////////////////////////////////////////////////////////////////////////
/*! Samples the old piecewise quadratic interpolated function at the new
 *  current interpolation nodes.
*///////////////////////////////////////////////////////////////////////////////
void transferScalarField()
{
    // Only transfer if there is an interpolated solution to transfer
    if (interp_vertices.size() != vertices.size())
        return;

    // Use basis functions from the original interpolation time.
    mesh->updateBarycentricCoordinates(triangles, interp_vertices);

    // Sample the original solution at the current mesh nodes
    vector<LMesh::FieldType> newVertexValues(vertices.size());
    // Storing per-halfedge is of course redundant for internal edges
    vector<LMesh::FieldType> newHalfedgeValues(mesh->halfedge_size());
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        if (isFixedVertex[i])
            newVertexValues[i] = mesh->vertex(i)->c[0];
        newVertexValues[i] = mesh->sample(vertices[i]);
    }
    for (unsigned int i = 0; i < mesh->halfedge_size(); ++i) {
        LMesh::Halfedge *h = mesh->halfedge(i);
        size_t v1 = mesh->vertex_index(h->tip()),
               v2 = mesh->vertex_index(h->opposite()->tip());
        if (isFixedVertex[v1] && isFixedVertex[v2])
            newHalfedgeValues[i] = mesh->halfedge(i)->edgeData().c[0];
        // Sample at the edge midpoint
        newHalfedgeValues[i] = mesh->sample(.5 * (vertices[v1] + vertices[v2]));
    }

    // Copy samples into the mesh
    for (unsigned int i = 0; i < vertices.size(); ++i)
        mesh->vertex(i)->c[0] = newVertexValues[i];
    for (unsigned int i = 0; i < mesh->halfedge_size(); ++i)
        mesh->halfedge(i)->edgeData().c[0] = newHalfedgeValues[i];

    // The scalar field is now on the current mesh
    interp_vertices = vertices;

    // Return to our new basis functions/coordinates
    mesh->updateBarycentricCoordinates(triangles, vertices);
}

////////////////////////////////////////////////////////////////////////////////
/*! Perturbs all the vertices a random fraction of the edge length to a random
//  neighbor
*///////////////////////////////////////////////////////////////////////////////
void perturbVertices()
{
    std::vector<Point2D> perturbed(vertices);

    for (unsigned int i = 0; i < vertices.size(); ++i) {
        long neighbor = random() % (mesh->vertex(i)->valence());
        LMesh::Halfedge *circ = mesh->vertex(i)->halfedge();
        for (unsigned int j = 0; j < neighbor; ++j, circ = circ->cw())
        { }
        Point2D neighborPt =
            vertices[mesh->vertex_index(circ->opposite()->tip())];
        double alpha = .5 * (random() / ((double) INT_MAX));
        if (!isFixedVertex[i])
            perturbed[i] = (1 - alpha) * vertices[i] + alpha * neighborPt;
    }

    positionVertices(perturbed);
}

////////////////////////////////////////////////////////////////////////////////
/*! Maps all the boundary vertices to a unit circle.
//  (Assumes disk topology)
*///////////////////////////////////////////////////////////////////////////////
void mapBoundary()
{
    vector<size_t> bnd = mesh->boundary_vertices();
    static enum {CIRCLE, SQUARE} shape = CIRCLE;
    // Map boundary vertices to a (counterclockwise) circle
    Real deltaTheta = 2 * M_PI / bnd.size();
    std::vector<Point2D> positions(vertices);
    size_t numBnd = bnd.size();
    if (shape == CIRCLE || (numBnd <= 3)) {
        for (size_t i = 0; i < bnd.size(); ++i) {
            positions[bnd[i]] = Point2D(.5 * cos(deltaTheta * i) + .5,
                                        .5 * sin(deltaTheta * i) + .5);
        }
        shape = SQUARE;
    }
    else {
        // Partition the boundary vertices into four sets that will map to the
        // four sides.
        shape = CIRCLE;
        size_t first  = numBnd / 4;
        size_t second = first * 2;
        size_t third  = first * 3;

        for (size_t i = 0; i < first; ++i)
            positions[bnd[i]] = Point2D(1.0, (1.0 / first) * i);
        for (size_t i = first; i < second; ++i)
            positions[bnd[i]] = Point2D(1.0 - (1.0 / (second - first)) * (i - first), 1.0);
        for (size_t i = second; i < third; ++i)
            positions[bnd[i]] = Point2D(0.0, 1.0 - (1.0 / (third - second)) * (i - second));
        for (size_t i = third; i < numBnd; ++i)
            positions[bnd[i]] = Point2D((1.0 / (numBnd - third)) * (i - third), 0.0);
    }

    positionVertices(positions);
}

////////////////////////////////////////////////////////////////////////////////
// Callback Dispatch
// Forward all GLUT callbacks to the UIController object
////////////////////////////////////////////////////////////////////////////////
void Reshape(             int width, int height) { g_uiController.Reshape(      width, height); }
void KeyboardFunc(unsigned char k, int x, int y) { g_uiController.KeyboardFunc(       k, x, y); }
void SpecialKeyboardFunc(   int k, int x, int y) { g_uiController.SpecialKeyboardFunc(k, x, y); }
void MotionFunc(                   int x, int y) { g_uiController.MotionFunc(            x, y); }
void PassiveMotionFunc(            int x, int y) { g_uiController.PassiveMotionFunc(     x, y); }
void MouseFunc(int but, int state, int x, int y) { g_uiController.MouseFunc( but, state, x, y); }

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on sucess)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    // Initialize GLUT
    glutInit(&argc, argv);

    char *meshPath;
    int width = 640;
    int height = 640;

    if (argc == 2) {
        meshPath = argv[1];
    }
    else if (argc == 4) {
        meshPath = argv[1];
        width  = atoi(argv[2]);
        height = atoi(argv[3]);
    }
    else {
        cout << "usage: viewer mesh.off [width height]" << endl;
        exit(-1);
    }

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(width, height);

    char *name = strrchr(meshPath, '/');
    if (name)
        ++name;
    else
        name = meshPath;

    char title[512];
    snprintf(title, 512, "FEM2D: %s", name);
    glutCreateWindow(title);

    cout << "Reading mesh... "; cout.flush();
    vector<IOVertex<Point3D> > in_vertices;
    vector<IOElement> in_triangles;
    load(meshPath, in_vertices, in_triangles);
    
    // translate and scale to fit bounding box inside [0, 1]x[0, 1]
    Point2D minCoords(in_vertices[0][0], in_vertices[0][1]);
    Point2D maxCoords(minCoords);
    for (unsigned int i = 0; i < in_vertices.size(); ++i) {
        minCoords = Point2D(min(minCoords[0], in_vertices[i][0]),
                            min(minCoords[1], in_vertices[i][1]));
        maxCoords = Point2D(max(maxCoords[0], in_vertices[i][0]),
                            max(maxCoords[1], in_vertices[i][1]));
    }
    Point2D dimensions = maxCoords - minCoords;
    double maxDim = max(dimensions[0], dimensions[1]); 

    for (unsigned int i = 0; i < in_vertices.size(); ++i) {
        Point2D p(in_vertices[i][0], in_vertices[i][1]);
        vertices.push_back((p - minCoords) / maxDim);
    }
    for (unsigned int i = 0; i < in_triangles.size(); ++i) {
        triangles.push_back(TriangleIndex(in_triangles[i][0],
                                          in_triangles[i][1],
                                          in_triangles[i][2]));
    }

    orig_vertices  = vertices;
    orig_triangles = triangles;
    isFixedVertex.resize(vertices.size());
    cout << "done." << endl;

    cout << "Constructing lagrange mesh... "; cout.flush();
    mesh = new LMesh(triangles, vertices);
    cout << "done." << endl;

    readShader("LagrangeShader.vert", "LagrangeShader.frag", lagrangeShader);
    glUseProgram(lagrangeShader);
    ls_cornerValuesLoc = glGetAttribLocation(lagrangeShader, "cornerValues");
    ls_edgeValuesLoc   = glGetAttribLocation(lagrangeShader, "edgeValues");
    ls_colorScaleLoc   = glGetUniformLocation(lagrangeShader, "colorScale");
    ls_baryCoordLoc    = glGetAttribLocation( lagrangeShader, "baryCoord");
    ls_degreeLoc       = glGetUniformLocation(lagrangeShader, "degree");
    glUseProgram(0);
    readShader("ExactShader.vert", "ExactShader.frag", exactShader);
    glUseProgram(exactShader);
    es_colorScaleLoc   = glGetUniformLocation(exactShader, "colorScale");
    es_freqLoc         = glGetUniformLocation(exactShader, "freq");
    glUseProgram(0);

    // Set GLUT event callbacks
    glutMouseFunc(MouseFunc);
    glutMotionFunc(MotionFunc);
    glutPassiveMotionFunc(PassiveMotionFunc);
    glutKeyboardFunc(KeyboardFunc);
    glutSpecialFunc(SpecialKeyboardFunc);

    // Set GLUT view callbacks
    glutDisplayFunc(Display);
    glutReshapeFunc(Reshape);

    // Call the GLUT main loop
    glutMainLoop();

    return 0;
}
