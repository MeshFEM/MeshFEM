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

#include "MeshlessFEM.hh"

FEMView2D::FEMView2D(MeshlessFEM_t &fem, QWidget *parent)
    : QGLWidget(parent), m_frameMin(-2, -1.5), m_frameMax(2, 1.5),
      m_rgbaBuffer(NULL), m_overlayDirty(true), m_objectDirty(true),
      m_fem(fem), m_guiState(MODEL_STATE), m_gesture(NONE)
{
    setFormat(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer));
}

void FEMView2D::csgNodesSelected(const NodeList &nList)
{
    m_selectedObjects = nList;
    m_overlayDirty = true;
    update();
}

void FEMView2D::initializeGL()
{
    glClearColor(0.8, 0.8, 0.8, 1.0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    
    glGenTextures(1, &m_modelTex);
    glBindTexture(GL_TEXTURE_2D, m_modelTex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glGenTextures(1, &m_overlayTex);
    glBindTexture(GL_TEXTURE_2D, m_overlayTex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
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

    delete m_rgbaBuffer;
    m_rgbaBuffer = new char[4 * m_width * m_height];
    m_objectDirty = m_overlayDirty = true;

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

void FEMView2D::m_clearBuffer()
{
    memset(m_rgbaBuffer, 0, 4 * m_width * m_height);
}

template<typename Object>
void FEMView2D::drawObject(const Object *obj, const QColor &fg) const
{
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
}

void FEMView2D::m_loadTexture(GLuint tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, m_rgbaBuffer);
}

void FEMView2D::m_drawObject()
{
    QColor modelColor(128, 192, 255, 255);

    if (m_objectDirty) {
        m_clearBuffer();
        drawObject(&(m_fem.model()), modelColor);
        m_loadTexture(m_modelTex);
        m_objectDirty = false;
    }

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_modelTex);
    drawQuad(0, 0, m_width, m_height);
}

void FEMView2D::m_drawWorldBox(const BBox_t &b)
{
    int minx, miny, maxx, maxy;
    getBufferCoords(b.minCorner[0], b.minCorner[1], miny, minx);
    getBufferCoords(b.maxCorner[0], b.maxCorner[1], maxy, maxx);
    drawQuad(minx, miny, maxx, maxy);
}

void FEMView2D::m_drawSelectedObjects()
{
    QColor selectedObjectColor(128, 128, 128, 128);
    if (m_overlayDirty) {
        m_clearBuffer();
        for (NodeList::iterator it = m_selectedObjects.begin();
                                it != m_selectedObjects.end(); ++it) {
            drawObject(*it, selectedObjectColor);
        }
        m_loadTexture(m_overlayTex);
        m_overlayDirty = false;
    }

    glBindTexture(GL_TEXTURE_2D, m_overlayTex);
    drawQuad(0, 0, m_width, m_height);
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

void FEMView2D::draw()
{
    glColor3f(1, 1, 1);
    glDisable(GL_TEXTURE_2D);
    drawQuad(0, 0, m_width, m_height);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    m_drawObject();
    if (m_guiState == MODEL_STATE) {
        m_drawSelectedObjects();
    }
    if (m_guiState == ELEMENTS_STATE) {
        glDisable(GL_TEXTURE_2D);
        ElementGrid2D_t &grid = m_fem.elementGrid();
        for (size_t i = 0; i < grid.numElements(); ++i) {
            glColor4f(.8f, .8f, .8f, .5f);
            if (!grid.elementIsFull(i)) 
                glColor4f(.8f, 0.0f, 0.0f, .5f);
            m_drawWorldBox(grid.elementBoundingBox(i));
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glColor4f(0, 0, 0, 1);
        for (size_t i = 0; i < grid.numElements(); ++i) {
            m_drawWorldBox(grid.elementBoundingBox(i));
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glEnable(GL_POINT_SMOOTH);
        // Draw quadrature points
        glPointSize(2.0f);
        glColor3f(1.0, 1.0, 0);
        glBegin(GL_POINTS);
        for (unsigned int i = 0; i < grid.numElements(); ++i) {
            BBox_t b = grid.elementBoundingBox(i);
            std::vector<Vector> qpoints =
                m_fem.quadrature().quadraturePoints(b);
            for (unsigned int p = 0; p < qpoints.size(); ++p) {
                int r, c;
                getBufferCoords(qpoints[p][0], qpoints[p][1], r, c);
                glVertex2f(c, r);
            }
        }
        glEnd();

        // Draw nodes
        glPointSize(4.0f);
        glColor3f(0.0, 0.0, 0);
        glBegin(GL_POINTS);
        for (unsigned int i = 0; i < grid.numNodes(); ++i) {
            Vector p = grid.nodePosition(i);
            int r, c;
            getBufferCoords(p[0], p[1], r, c);
            glVertex2f(c, r);
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
        m_objectDirty = m_overlayDirty = true;
        update();
    }
    m_prevMouseLoc = event->pos();
}

void FEMView2D::mouseDoubleClickEvent(QMouseEvent *event)
{
    
}
