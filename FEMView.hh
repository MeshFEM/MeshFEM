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
#include <Eigen/Dense>
#include <cmath>

#include "GlobalTypes.hh"

class FEMView2D : public QGLWidget
{
    Q_OBJECT

public:
    typedef enum {MODEL_STATE, ELEMENTS_STATE, FORCES_STATE,
                  DISPLACEMENTS_STATE} GUIState;

    FEMView2D(MeshlessFEM_t &fem, QWidget *parent = NULL);
    ~FEMView2D() {
        delete m_rgbaBuffer;
    }

    void setGUIState(GUIState state) {
        m_guiState = state;
        m_gesture = NONE;
        update();
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

private:
    template<typename Object>
    void drawObject(const Object *obj, const QColor &c) const;
    typedef enum {DRAW_CELLS, DRAW_NODES, DRAW_EDGES} DrawOp;
    void drawGrid(DrawOp op, const std::vector<Vector> &deformation =
                  std::vector<Vector>());
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
    std::vector<Vector> selectedDeformation;

    GUIState m_guiState;
    typedef enum {DRAGGING, NONE} MouseGesture;
    MouseGesture m_gesture;
    QPoint m_prevMouseLoc;
};

#endif // FEMVIEW_HH
