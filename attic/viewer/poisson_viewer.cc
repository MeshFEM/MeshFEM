#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <limits>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>

#include <Eigen/Dense>

#include "../MeshIO.hh"
#include "../Geometry.hh"

#include "../Poisson.hh"
#include "colors.hh"
#include "draw.hh"

using namespace std;
using namespace MeshIO;

vector<TriangleIndex> triangles;
vector<Point2D>       vertices;
vector<Point2D>       gradU;
vector<double>        soln;

size_t g_selectedNode;

#ifndef DEGREE
#define DEGREE 2
#endif

typedef PoissonMesh<2, DEGREE, Point2D> PMesh;
PMesh *mesh;

void drawTriangle2D(const TriangleIndex &tri)
{
    glVertex2f(vertices[tri[0]][0], vertices[tri[0]][1]);
    glVertex2f(vertices[tri[1]][0], vertices[tri[1]][1]);
    glVertex2f(vertices[tri[2]][0], vertices[tri[2]][1]);
}

////////////////////////////////////////////////////////////////////////////
/*! The window size changed.
//  @param[in]  width   new width  (in pixels)
//  @param[in]  height  new height (in pixels)
*///////////////////////////////////////////////////////////////////////////
void Reshape(int width, int height)
{
    // Set OpenGL viewport and camera
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double minDimension = std::min(width, height);
    gluOrtho2D(0, width / minDimension, 0, height / minDimension);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glutPostRedisplay();
}

Point3D translation, dragTranslation;
bool translating = false;
float zoom = 1.0;
void applyViewTransforms(bool includeInteractive = true)
{
    glMatrixMode(GL_MODELVIEW);
    glTranslated(.5, .5, 0);
    glScalef(zoom, zoom, 1.0);
    glTranslated(-.5, -.5, 0);

    // Apply interactive translation
    glTranslated(translation[0], translation[1], translation[2]);

    // Apply temporary transformation
    if (translating && includeInteractive)  {
        glTranslated(dragTranslation[0], dragTranslation[1],
                     dragTranslation[2]);
    }
}

void drawArrow(const Point2D &tail, const Point2D &tip)
{
    glBegin(GL_LINES);
    Point2D vec = tip - tail;

    glVertex2f(tail[0], tail[1]);
    glVertex2f(tip[0],  tip[1]);

    float cospi_4 = cos(M_PI / 4.0), sinpi_4 = sin(M_PI / 4.0);
    // Draw arrow head (scaled vec, rotated by +/- Pi/4, subtracted from tip)
    for (int j = 0; j < 2; ++j) {
        Point2D head;
        head[0] = cospi_4 * vec[0] - sinpi_4 * vec[1];
        head[1] = sinpi_4 * vec[0] + cospi_4 * vec[1];
        head = tip - .25 * head;
        glVertex2f(tip[0], tip[1]);
        glVertex2f(head[0], head[1]);
        sinpi_4 = -sinpi_4; // Draw the -pi/4 rotation next
    }
    glEnd();
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
    applyViewTransforms();

    // Clear drawing buffers
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw solution
    double smax = -1e16, smin = 1e16;
    for (size_t i = 0; i < soln.size(); ++i) {
        smax = max(soln[i], smax);
        smin = min(soln[i], smin);
    }
    ColorMap<RGBColorf, double> colorMap(COLORMAP_FIREPRINT, smax, smin);

    glBegin(GL_TRIANGLES);
    for (unsigned int i = 0; i < triangles.size(); ++i) {
        const TriangleIndex &tri = triangles[i];

        glColor4fv(colorMap(soln[tri[0]]));
        glVertex2f(vertices[tri[0]][0], vertices[tri[0]][1]);
        glColor4fv(colorMap(soln[tri[1]]));
        glVertex2f(vertices[tri[1]][0], vertices[tri[1]][1]);
        glColor4fv(colorMap(soln[tri[2]]));
        glVertex2f(vertices[tri[2]][0], vertices[tri[2]][1]);
    }
    glEnd();


    // Draw gradient at Barycenter, scaled based on shortest edge in entire mesh
    double shortest = 1e16;
    double maxNorm = 0;
    for (unsigned int i = 0; i < triangles.size(); ++i) {
        const TriangleIndex &tri = triangles[i];
        shortest = min((vertices[tri[1]] - vertices[tri[0]]).norm(), shortest);
        shortest = min((vertices[tri[2]] - vertices[tri[1]]).norm(), shortest);
        shortest = min((vertices[tri[0]] - vertices[tri[2]]).norm(), shortest);
        maxNorm  = max(gradU[i].norm(), maxNorm);
    }
    float scale = shortest / maxNorm;
    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(2.0f);
    for (unsigned int i = 0; i < triangles.size(); ++i) {
        const TriangleIndex &tri = triangles[i];
        Point2D c = (1.0/3.0) * (vertices[tri[0]] + vertices[tri[1]]
                             + vertices[tri[2]]);
        Point2D vec = gradU[i] * scale;;
        Point2D tip  = c + .5 * vec;
        Point2D tail = c - .5 * vec;
        drawArrow(tail, tip);
    }

    glColor3f(0.0f, 0.0f, 0.0f);
    glLineWidth(1.0f);
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
    for (unsigned int i = 0; i < mesh->numBoundaryNodes(); ++i) {
        auto bn = mesh->boundaryNode(i);
        if (bn->constraintType == CONSTRAINT_DIRICHLET)
            glVertex2f(bn.volumeNode()->p[0], bn.volumeNode()->p[1]);
    }
    glEnd();

    if (g_selectedNode < mesh->numNodes()) {
        glBegin(GL_POINTS);
            glColor3f(1.0f, 1.0f, 1.0f);
            auto n = mesh->node(g_selectedNode);
            glVertex2f(n->p[0], n->p[1]);
        glEnd();
        glColor3f(0.0f, 0.0f, 0.0f);
        string str("Val for node " + to_string(g_selectedNode) + ": "
                 + to_string(soln.at(g_selectedNode)));
        drawString(1, 1, str.c_str());
    }
    // if (g_selectedVertex < vertices.size()) {
    //     glLineWidth(2.0f);

    //     int numHalfedges = 2 * mesh->vertex(g_selectedVertex)->valence();
    //     HSVColorf edgeColor(0.0, 1.0, 0.75);
    //     int textHeight = 1;

    //     size_t hi = mesh->halfedge_index(
    //             mesh->vertex(g_selectedVertex)->halfedge());
    //     size_t he = hi;
    //     do {
    //         size_t h = hi;
    //         for (size_t i = 0; i < 2; ++i) {
    //             if (mesh->halfedge(h)->facet()) {
    //                 Point2D ev = mesh->outwardEdgeVector(h);
    //                 Point2D mp = mesh->midpoint(h);
    //                 glColor3fv(RGBColorf(edgeColor));
    //                 drawArrow(mp, mp + .25 * ev);
    //                 
    //                 size_t fi = mesh->facet_index(mesh->halfedge(h)->facet());
    //                 assert(fi < gradU.size());
    //                 string str("Flux for ");
    //                 if (mesh->halfedge(h)->isBoundaryEdge())
    //                     str += "boundary ";
    //                 str += "halfedge ";
    //                 str += to_string(h) + ": ";
    //                 str += to_string(gradU[fi].dot(ev));
    //                 double flux = gradU[fi].dot(ev);
    //                 drawString(1, textHeight, str.c_str());
    //                 textHeight += 10;
    //             }
    //             h = mesh->halfedge_index(mesh->halfedge(h)->opposite());
    //             edgeColor.h += 1.0 / numHalfedges;
    //         }
    //     } while ((hi = mesh->halfedge_index(mesh->halfedge(hi)->cw())) != he);

    //     glBegin(GL_POINTS);
    //         glColor3f(1.0f, 1.0f, 1.0f);
    //         glVertex2f(vertices[g_selectedVertex][0],
    //                    vertices[g_selectedVertex][1]);
    //     glEnd();
    // }

    glutSwapBuffers();
}

////////////////////////////////////////////////////////////////////////////
/*! Retrieves the worldspace coordinates of a click location
//  @param[in]  x, y, z     window coordinates (y = 0 means top of screen)
//  @param[out] wx, wy, wz  worldspace coordinates
*///////////////////////////////////////////////////////////////////////////
void getWorldCoords(GLdouble   x, GLdouble   y, GLdouble   z,
                    GLdouble &wx, GLdouble &wy, GLdouble &wz)
{
    GLdouble model[16], proj[16];
    GLint viewport[4];

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    applyViewTransforms(false);
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glPopMatrix();

    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, viewport);
    // OpenGL has 0 = bottom of screen
    y = (viewport[3] - 1) - y;

    gluUnProject(x, y, z, model, proj, viewport, &wx, &wy, &wz);
}

void selectVertex(int x, int y)
{
    GLdouble wx, wy, wz;
    getWorldCoords(x, y, 0, wx, wy, wz);
    Point2D p(wx, wy);
    double closestDist = .05;
    g_selectedNode = mesh->numNodes();
    // Would be better to do threshold check in screen coords...
    for (size_t i = 0; i < mesh->numNodes(); ++i) {
        float dist = (mesh->node(i)->p - p).norm();
         if (dist < closestDist) {
             closestDist = dist;
             g_selectedNode = i;
         }
    }
}

////////////////////////////////////////////////////////////////////////////
/*! Called when a mouse button event occurs
//  @param[in]  button  GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON,
//                      GLUT_RIGHT_BUTTON
//  @param[in]  state   GLUT_UP or GLUT_DOWN
//  @param[in]  x       Mouse x location (window coordinates)
//  @param[in]  y       Mouse y location (window coordinates)
*///////////////////////////////////////////////////////////////////////////
void MouseFunc(int button, int state, int x, int y)
{
    selectVertex(x, y);

    glutPostRedisplay();
}

void MotionFunc(int x, int y) {
    selectVertex(x, y);
    glutPostRedisplay();
}


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
        cout << "usage: poisson_viewer mesh.off [width height]" << endl;
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
    snprintf(title, 512, "Poisson: %s", name);
    glutCreateWindow(title);

    vector<IOVertex>  in_vertices;
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
        in_vertices[i][0] = vertices.back()[0];
        in_vertices[i][1] = vertices.back()[1];
    }
    for (unsigned int i = 0; i < in_triangles.size(); ++i) {
        triangles.push_back(TriangleIndex(in_triangles[i][0],
                                          in_triangles[i][1],
                                          in_triangles[i][2]));
    }

    mesh = new PMesh(in_triangles, in_vertices);
    cout << "Vertices: " << mesh->numVertices() << std::endl;
    cout << "Elements: " << mesh->numElements() << std::endl;
    cout << "Nodes: " << mesh->numNodes() << std::endl;
    cout << "BoundaryVertices: " << mesh->numBoundaryVertices() << std::endl;
    cout << "Elements: " << mesh->numBoundaryElements() << std::endl;
    cout << "Boundary Nodes: " << mesh->numBoundaryNodes() << std::endl;
    size_t numConstrained = 0;
    Real minVal = numeric_limits<Real>::max(), maxVal = numeric_limits<Real>::min();
    for (size_t i = 0; i < mesh->numBoundaryNodes(); ++i) {
        auto bn = mesh->boundaryNode(i);
        Point2D origP = bn.volumeNode()->p * maxDim + minCoords;
        if (origP.norm() > 2.0) {
            bn->constraintType = CONSTRAINT_DIRICHLET;
            bn->constraintData = sin(.5 * M_PI * origP[0]);
            ++numConstrained;

            minVal = min(minVal, bn->constraintData);
            maxVal = max(maxVal, bn->constraintData);
        }
    }

    cout << "minVal = " << minVal << ", maxVal = " << maxVal << endl;


    mesh->solve(soln);
    gradU = mesh->gradUAverage(soln);

    double smax = -1e16, smin = 1e16;
    for (size_t i = 0; i < soln.size(); ++i) {
        smax = max(soln[i], smax);
        smin = min(soln[i], smin);
    }
    std::cout << "smin, smax: " << smin << ", " << smax << std::endl;

    g_selectedNode = mesh->numNodes();

    // Set GLUT view callbacks
    glutDisplayFunc(Display);
    glutReshapeFunc(Reshape);
    glutMouseFunc(MouseFunc);
    glutMotionFunc(MotionFunc);

    // Call the GLUT main loop
    glutMainLoop();

    return 0;
}
