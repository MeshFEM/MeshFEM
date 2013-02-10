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
    typedef MeshlessFEM_t::VectorField VectorField;

    FEMView2D(MeshlessFEM_t &fem, QWidget *parent = NULL);
    ~FEMView2D() {
        delete m_rgbaBuffer;
    }

    void setGUIState(GUIState state) {
        m_guiState = state;
        m_gesture = NONE;
        update();

        m_selectedElement = 0;
        m_selectedCorner  = 0;

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
        if (event->key() == Qt::Key_Up) {
            const ElementGrid2D_t &grid = m_fem.elementGrid();
            if (grid.numElements() > 0) {
                if (++m_selectedCorner == 4) {
                    m_selectedCorner = 0;
                    m_selectedElement =
                        (m_selectedElement + 1) % grid.numElements();
                }
                ElementGrid2D_t::AdjacencyVec corners = grid.elementCorners(m_selectedElement);
                std::cout << "Selected element, corner, node: "
                    << m_selectedElement << ", " << m_selectedCorner << ", "
                    << corners[m_selectedCorner] << std::endl;
            }
            else {
                m_selectedElement = 0;
                m_selectedCorner = 0;
            }
            update();
        }
    }

    void getWorldCoords(int r, int c, Scalar &x, Scalar &y) const {
        Vector frameDim = m_frameMax - m_frameMin;
        x = m_frameMin[0] + frameDim[0] * ((c + .5) / m_width);
        y = m_frameMin[1] + frameDim[1] * ((r + .5) / m_height);
    }

    void getBufferCoords(Scalar x, Scalar y, int &r, int &c) const {
        Vector frameDim = m_frameMax - m_frameMin;
        r = floor((y - m_frameMin[1]) * (m_height / frameDim[1]));
        c = floor((x - m_frameMin[0]) * (m_width / frameDim[0]));
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
    void drawGrid(DrawOp op, const VectorField &deformation =
                  VectorField());
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

    size_t m_selectedCorner, m_selectedElement;
};

#endif // FEMVIEW_HH
