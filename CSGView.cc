////////////////////////////////////////////////////////////////////////////////
// CSGView.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      OpenGL-based viewer for the CSG object
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//
//  Created:  01/28/2013 15:22:32
//  Revision History:
//      01/28/2013  Julian Panetta    Initial Revision
////////////////////////////////////////////////////////////////////////////////
#include "CSGView.hh"
#include <QtGui>
#include <QGLWidget>
#include <QColor>
#include <cassert>
#include <iostream>

CSGView2D::CSGView2D(CSGTree_t &csgTree, QWidget *parent)
    : QGLWidget(parent), m_frameMin(-2, -1.5), m_frameMax(2, 1.5),
      m_rgbaBuffer(NULL), m_csgTree(csgTree)
{
    setFormat(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer));
}

void CSGView2D::csgNodesSelected(const NodeList &nList)
{
    m_selectedNodes = nList;
    update();
}

void CSGView2D::initializeGL()
{
    glClearColor(0.8, 0.8, 0.8, 1.0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    
    glGenTextures(1, &m_renderTex);
    glBindTexture(GL_TEXTURE_2D, m_renderTex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // Allow texture to wrap to enable checkerboard resolution scaling
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

void CSGView2D::resizeGL(int width, int height)
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

    delete m_rgbaBuffer;
    m_rgbaBuffer = new char[4 * m_width * m_height];

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
}

template<typename CSGObject>
void CSGView2D::drawCSG(const CSGObject *obj, const QColor &fg) const
{
    memset(m_rgbaBuffer, 0, 4 * m_width * m_height);
    for (int r = 0; r < m_height; ++r) {
        for (int c = 0; c < m_width; ++c) {
            Vector p;
            getWorldCoords(r, c, p[0], p[1]);
            if (obj->isInside(p)) {
                m_rgbaBuffer[4 * (r * m_width + c) + 0] = (char) fg.red();
                m_rgbaBuffer[4 * (r * m_width + c) + 1] = (char) fg.green();
                m_rgbaBuffer[4 * (r * m_width + c) + 2] = (char) fg.blue();
                m_rgbaBuffer[4 * (r * m_width + c) + 3] = (char) fg.alpha();
            }
        }
    }

    glBindTexture(GL_TEXTURE_2D, m_renderTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, m_rgbaBuffer);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);

    glTexCoord2f(1, 0);
    glVertex2f(m_width, 0);

    glTexCoord2f(1, 1);
    glVertex2f(m_width, m_height);

    glTexCoord2f(0, 1);
    glVertex2f(0, m_height);

    glEnd();
    glDisable(GL_TEXTURE_2D);

    // Draw the bounding box
    BBox_t b = obj->boundingBox();
    int minx, miny, maxx, maxy;
    getBufferCoords(b.minCorner[0], b.minCorner[1], miny, minx);
    getBufferCoords(b.maxCorner[0], b.maxCorner[1], maxy, maxx);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3f(fg.red() / 255.0f, fg.green() / 255.0f, fg.blue() / 255.0f);
    glBegin(GL_QUADS);
    glVertex2f(minx, miny);
    glVertex2f(maxx, miny);
    glVertex2f(maxx, maxy);
    glVertex2f(minx, maxy);
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void CSGView2D::draw()
{
    glDisable(GL_TEXTURE_2D);

    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
    glVertex2f(0      , 0);
    glVertex2f(m_width, 0);
    glVertex2f(m_width, m_height);
    glVertex2f(0      , m_height);
    glEnd();

    QColor fullObjectColor(128, 192, 255, 255);
    drawCSG(&m_csgTree, fullObjectColor);

    QColor selectedObjectColor(128, 128, 128, 128);
    for (NodeList::iterator it = m_selectedNodes.begin();
                            it != m_selectedNodes.end(); ++it) {
        drawCSG(*it, selectedObjectColor);
    }

}

void CSGView2D::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw();
}

void CSGView2D::mousePressEvent(QMouseEvent *event)
{
    prevMouseLoc = event->pos();
}

void CSGView2D::mouseMoveEvent(QMouseEvent *event)
{
    Vector start, end;
    getWorldCoords(-prevMouseLoc.y(), prevMouseLoc.x(), start[0], start[1]);
    getWorldCoords(-event->pos().y(), event->pos().x(), end[0], end[1]);
    if (event->buttons() & Qt::LeftButton) {
        for (NodeList::iterator it = m_selectedNodes.begin();
                                it != m_selectedNodes.end(); ++it) {
            (*it)->applyTranslation(end - start);
        }
        update();
    }
    prevMouseLoc = event->pos();
}

void CSGView2D::mouseDoubleClickEvent(QMouseEvent *event)
{
    
}
