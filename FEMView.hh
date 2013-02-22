////////////////////////////////////////////////////////////////////////////////
// FEMView.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      OpenGL-based viewer for the MeshlessFEM/CSG code.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/28/2013 15:09:13
////////////////////////////////////////////////////////////////////////////////
#ifndef FEMVIEW_HH
#define FEMVIEW_HH

#include <QGLWidget>
#include <QBasicTimer>
#include <QTimerEvent>
#include <QKeyEvent>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#include "GlobalTypes.hh"
#include "MeshlessFEM.hh"

class FEMView2D : public QGLWidget
{
    Q_OBJECT

public:
    typedef enum {MODEL_STATE, ELEMENTS_STATE, FORCES_STATE,
                  DISPLACEMENTS_STATE} GUIState;
    typedef MeshlessFEM_t::VField VField;

    FEMView2D(MeshlessFEM_t &fem, QWidget *parent = NULL);
    ~FEMView2D() {
        delete m_rgbaBuffer;
    }

    void setGUIState(GUIState state) {
        m_guiState = state;
        m_gesture = NONE;
        update();

        if (m_guiState == DISPLACEMENTS_STATE) {
            m_timer.start(1000.0 / 60, this);
        }
        else {
            m_timer.stop();
        }
    }

    void selectDeformation(size_t i) {
        assert(i < m_fem.numModes());
        m_selectedDeformation = i;
    }
    
public slots:
    void csgNodesSelected(const NodeList &nList);

protected:
    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event) {
    }

    // Get the world coordinates corresponding to buffer coordinates
    void getWorldCoords(int r, int c, Scalar &x, Scalar &y) const {
        Vector frameDim = m_frameMax - m_frameMin;
        x = m_frameMin[0] + frameDim[0] * ((c + .5) / m_width);
        y = m_frameMin[1] + frameDim[1] * ((r + .5) / m_height);
    }

    void getTextureCoordinates(Scalar x, Scalar y, Scalar &s, Scalar &t) const
    {
        Vector frameDim = m_frameMax - m_frameMin;
        s = (x - m_frameMin[0]) / frameDim[0];
        t = (y - m_frameMin[1]) / frameDim[1];
    }

    // Non-rounded for opengl drawing
    void getScreenCoords(Scalar x, Scalar y, Scalar &sx, Scalar &sy) const {
        Scalar s, t;
        getTextureCoordinates(x, y, s, t);
        sx = m_width * s;
        sy = m_height * t;
    }

    // Rounded for buffer drawing
    void getBufferCoords(Scalar x, Scalar y, int &r, int &c) const {
        Scalar s, t;
        getTextureCoordinates(x, y, s, t);
        r = floor(m_height * t);
        c = floor(m_width * s);
    }

    void timerEvent(QTimerEvent *event) {
        if (event->timerId() == m_timer.timerId()) {
            m_displacementPhase += .05;
            if (m_displacementPhase > 2.0 * M_PI) {
                m_displacementPhase = 0;
            }
            update();
        }
        else {
            QGLWidget::timerEvent(event);
        }
    }


private:
    template<typename Object>
    void drawObject(const Object *obj, const QColor &c) const;
    typedef enum {DRAW_CELLS, DRAW_NODES, DRAW_EDGES} DrawOp;
    void drawObjectTextureCells(const VField &deformation = VField());
    void drawGrid(DrawOp op, const VField &deformation = VField());
    void draw();
    void m_drawObject();
    void m_drawSelectedObjects();
    void m_drawWorldBox(const BBox_t &b);
    void m_drawWorldVertex(const Vector &v);
    void m_loadTexture(GLuint tex);
    void m_clearBuffer();

    Vector m_frameMin, m_frameMax;
    int m_width, m_height;
    GLuint m_modelTex, m_overlayTex;
    GLuint m_bilinearShader = 0;
    // Vertex coordinate attributes for bilinear displacement shader.
    GLuint m_vCoordLoc[4];
    // Texture coordinate attributes for bilinear displacement shader.
    GLuint m_tCoordLoc[4];
    // Object texture sampler loc
    GLuint m_objectTexLoc;

    char *m_rgbaBuffer;
    bool m_overlayDirty, m_objectDirty;

    MeshlessFEM_t &m_fem;
    NodeList m_selectedObjects;
    size_t m_selectedDeformation;

    GUIState m_guiState;
    typedef enum {DRAGGING, NONE} MouseGesture;
    MouseGesture m_gesture;
    QPoint m_prevMouseLoc;

    QBasicTimer m_timer;
    Scalar m_displacementPhase;
};

#endif // FEMVIEW_HH
