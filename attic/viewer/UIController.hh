////////////////////////////////////////////////////////////////////////////////
// UIController.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements the UIController singleton that tracks UI and gesture-related
//      state.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/04/2012 01:26:51
////////////////////////////////////////////////////////////////////////////////
#ifndef UICONTROLLER_HH
#define UICONTROLLER_HH

#include <iostream>
#include <GLUT/glut.h>
#include "Geometry.hh"
#include "LagrangeMesh.hh"

// Link against viewer's mesh.

struct UIController
{
    /** The window dimensions */
    int width, height;

    /** View zoom factor */
    GLdouble zoom;
    /** View translation */
    GLdouble translation[3];

    /** The index of the currently selected vertex (-1 for none) */
    int selectedVertex;
    /** The index of the currently selected triangle (-1 for none) */
    int selectedTriangle;
    /** The index of the currently selected halfedge (-1 for none) */
    int selectedHalfedge;

    bool selectingGeometry;

    /** The start point of a drag (world space) */
    GLdouble    dragStart[3];
    /** The current dragging translation */
    GLdouble    dragTranslation[3];
    /** Whether we are currently translating  */
    bool  translating;
    bool  draggingVertex;

    bool  screenshotRequested;
    bool  hideText;

    typedef enum  { SHADE_NONE    = 0,
                    SHADE_WEIGHTS = 1,
                    SHADE_INTERP  = 2,
                    SHADE_EXACT   = 3 } ShadingStyle;
    ShadingStyle shadeStyle;
    /** The frequency of the exact scalar field */
    float freq;
    /** The polynomical interpolation degree */
    int degree;

    UIController()
        : selectedVertex(-1), selectedTriangle(-1), selectedHalfedge(-1),
          selectingGeometry(false), translating(false), draggingVertex(false),
          screenshotRequested(false), hideText(false),
          shadeStyle(SHADE_WEIGHTS), freq(1.0), degree(2)
    {
        translation[0] = translation[1] = translation[2] = 0.0;
        zoom = 1.0;
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! GLUT keyboard callback
    //  @param[in]  k   Key pressed
    //  @param[in]  x   Mouse x location (window coordinates)
    //  @param[in]  y   Mouse y location (window coordinates)
    *///////////////////////////////////////////////////////////////////////////
    void KeyboardFunc(unsigned char k, int x, int y)
    {
        if ((k == 'q') || (k == 27 /* ESC */)) {
            exit(0);
        }
        else if (k == 'r') {
            // Reset vertex locations
            positionVertices();
        }
        else if (k == 'f') {
            // Fix selected vertex
            if (selectedVertex != -1)
                toggleFixed(selectedVertex);
        }
        else if (k == 'l') {
            // Do laplacian smoothing
            smooth();
        }
        else if (k == 'b') {
            toggleFixedBoundary();
        }
        else if (k == 'm') {
            mapBoundary();
        }
        else if (k == 'p') {
            screenshotRequested = true;
        }
        else if (k == 's') {
            // cycle shading mode
            shadeStyle = (ShadingStyle) ((shadeStyle + 1) % 4);
        }
        else if (k == 'S') {
            // cycle shading mode backward
            shadeStyle = (ShadingStyle) ((shadeStyle + 3) % 4);
        }
        else if (k == '+' || k == '=')
            zoom *= 1.25;
        else if (k == '-' || k == '_')
            zoom /= 1.25;
        else if (k == 'j')
            freq = std::max(freq / 1.25f, 0.0f);
        else if (k == 'k')
            freq *= 1.25;
        else if (k == 'i')
            interpolateScalarField();
        else if (k == ' ')
            perturbVertices();
        else if (k == 'd')
            degree = (degree == 1) ? 2 : 1;
        else if (k == 'c') {
            // Do one CG step of Laplacian smoothing
            cgSmooth();
        }
        else if (k == 't')
            transferScalarField();
        else if (k == 'h')
            hideText = !hideText;
        glutPostRedisplay();
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! GLUT keyboard special key callback
    //  @param[in]  k   Key pressed
    //  @param[in]  x   Mouse x location (window coordinates)
    //  @param[in]  y   Mouse y location (window coordinates)
    *///////////////////////////////////////////////////////////////////////////
    void SpecialKeyboardFunc(int k, int x, int y)
    {
        glutPostRedisplay();
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Called when the mouse moves while a button is pressed
    //  @param[in]  x   Mouse x location (window coordinates)
    //  @param[in]  y   Mouse y location (window coordinates)
    *///////////////////////////////////////////////////////////////////////////
    void MotionFunc(int x, int y)
    {
        if (translating) {
            GLdouble dragx, dragy, dragz;
            getWorldCoords(x, y, 0, dragx, dragy, dragz);

            // Compute the transform that will move the clicked point under the
            // cursor
            dragTranslation[0] = dragx - dragStart[0];
            dragTranslation[1] = dragy - dragStart[1];
            dragTranslation[2] = dragz - dragStart[2];
        }
        else if (draggingVertex) {
            GLdouble dragx, dragy, dragz;
            getWorldCoords(x, y, 0, dragx, dragy, dragz);
            positionVertex(selectedVertex, Point2D(dragx, dragy));
        }
        else if (selectingGeometry) {
            selectGeometry(x, y);
        }

        glutPostRedisplay();
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Called when the mouse moves while a button is NOT pressed
    //  @param[in]  x   Mouse x location (window coordinates)
    //  @param[in]  y   Mouse y location (window coordinates)
    *///////////////////////////////////////////////////////////////////////////
    void PassiveMotionFunc(int x, int y)
    {
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
        if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
            if (glutGetModifiers() & GLUT_ACTIVE_CTRL) {
                translating = true;
                getWorldCoords(x, y, 0, dragStart[0], dragStart[1],
                                        dragStart[2]);
                dragTranslation[0] = dragTranslation[1]
                                   = dragTranslation[2] = 0;
            }
            else if (glutGetModifiers() & GLUT_ACTIVE_ALT) {
                size_t tidx;
                GLdouble wx, wy, wz;
                getWorldCoords(x, y, 0, wx, wy, wz);
                Point2D p(wx, wy);
                Point3D baryCoords;
                if (mesh->barycentricCoords(p, tidx, baryCoords)) {
                    selectedVertex = triangles[tidx][0];
                    double closestBaryCoord = baryCoords[0];
                    for (unsigned int i = 0; i < 3; ++i) {
                        if (baryCoords[i] > closestBaryCoord) {
                            selectedVertex = triangles[tidx][i];
                            closestBaryCoord = baryCoords[i];
                        }
                    }
                    if (selectedVertex != -1)
                        draggingVertex = true;
                }
            }
            else {
                selectingGeometry = true;
                selectGeometry(x, y);
            }
        }
        if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)  {
            if (translating)  {
                // Apply translation!
                translation[0] += dragTranslation[0];
                translation[1] += dragTranslation[1];
                translation[2] += dragTranslation[2];
                translating = false;
            }
            else if (selectingGeometry) {
                    selectingGeometry = false;
                    selectGeometry(x, y);
            }
            else if (draggingVertex) {
                draggingVertex = false;
            }
        }

        glutPostRedisplay();
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! The window size changed.
    //  @param[in]  width   new width  (in pixels)
    //  @param[in]  height  new height (in pixels)
    *///////////////////////////////////////////////////////////////////////////
    void Reshape(int width, int height)
    {
        this->width = width;
        this->height = height;

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

    ////////////////////////////////////////////////////////////////////////////
    /*! Applies the view transformation to the current matrix
    *///////////////////////////////////////////////////////////////////////////
    void applyViewTransforms(bool includeInteractive = true) const
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

    ////////////////////////////////////////////////////////////////////////////
    /*! Retrieves the worldspace coordinates of a click location
    //  @param[in]  x, y, z     window coordinates (y = 0 means top of screen)
    //  @param[out] wx, wy, wz  worldspace coordinates
    *///////////////////////////////////////////////////////////////////////////
    void getWorldCoords(GLdouble   x, GLdouble   y, GLdouble   z,
                        GLdouble &wx, GLdouble &wy, GLdouble &wz) const
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

    ////////////////////////////////////////////////////////////////////////////
    /*! Determines the mesh triangle/vertex clicked on.
    //  Vertices are selected if a barycentric coordinate is > .90. Triangles
    //  are selected otherwise.
    //  @param[in]  x, y    GLUT window coordinates
    *///////////////////////////////////////////////////////////////////////////
    void selectGeometry(int x, int y)
    {
        GLdouble wx, wy, wz;
        getWorldCoords(x, y, 0, wx, wy, wz);
        Point2D p(wx, wy);
        Point3D baryCoords;
        size_t tidx;
        selectedVertex = selectedHalfedge = selectedTriangle = -1;
        if (mesh->barycentricCoords(p, tidx, baryCoords)) {
            for (unsigned int i = 0; i < 3; ++i) {
                if (baryCoords[i] > .90)
                    selectedVertex = triangles[tidx][i];
            }
            if (selectedVertex == -1) {
                const LMesh::Halfedge *h =
                             mesh->facet(tidx)->halfedge()->prev();
                for (unsigned int i = 0; i < 3; ++i, h = h->next()) {
                    if (baryCoords[i] < .10)
                        selectedHalfedge = mesh->halfedge_index(h);
                }
                selectedTriangle = (selectedHalfedge == -1) ? tidx : -1;
            }
        }
    }

};

#endif // UICONTROLLER_HH
