////////////////////////////////////////////////////////////////////////////////
// CSGView.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      OpenGL-based viewer for the CSG object
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/28/2013 15:09:13
////////////////////////////////////////////////////////////////////////////////
#ifndef CSGVIEW_HH
#define CSGVIEW_HH

#include <QGLWidget>
#include <Eigen/Dense>
#include <cmath>

#include "CSGTree.hh"
#include "GlobalTypes.hh"

class CSGView2D : public QGLWidget
{
    Q_OBJECT

public:
    CSGView2D(CSGTree_t &tree, QWidget *parent = NULL);
    ~CSGView2D() {
        delete m_rgbaBuffer;
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

    void getWorldCoords(int r, int c, float &x, float &y) const {
        Vector frameDim = m_frameMax - m_frameMin;
        x = m_frameMin[0] + frameDim[0] * ((c + .5) / m_width);
        y = m_frameMin[1] + frameDim[1] * ((r + .5) / m_height);
    }

    void getBufferCoords(float x, float y, int &r, int &c) const {
        Vector frameDim = m_frameMax - m_frameMin;
        r = floor((y - m_frameMin[1]) * (m_height / frameDim[1]));
        c = floor((x - m_frameMin[0]) * (m_width / frameDim[0]));
    }

private:
    template<typename CSGObject>
    void drawCSG(const CSGObject *obj, const QColor &c) const;
    void draw();

    Vector m_frameMin, m_frameMax;
    int m_width, m_height;
    GLuint m_renderTex;
    char *m_rgbaBuffer;

    CSGTree_t &m_csgTree;
    NodeList m_selectedNodes;

    typedef enum {DRAGGING, NONE} UIMode;
    UIMode m_uiMode;
    QPoint m_prevMouseLoc;
};

#endif // CSGVIEW_HH
