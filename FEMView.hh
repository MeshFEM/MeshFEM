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
#include <memory>
#include <FTGL/ftgl.h>

#include <OpenGL/OpenGL.h>
extern "C" {
#include "cl-helper.h"
}

#include "GlobalTypes.hh"
#include "MeshlessFEM.hh"
#include "ResultsCollector.hh"
#include "ViewSettings.hh"
#include "colors.hh"

class FEMView2D : public QGLWidget
{
    Q_OBJECT

public:
    typedef enum { STATE_MODEL = 0, STATE_ELEMENTS = 1,
                   STATE_PRESSURE_DRAW = 2, STATE_RESULT = 3 } GUIState;

    struct SelectionTool {
        typedef enum { NONE = 0, NODE, ELEM, BOUNDARY, NUM_TYPES } SelType;

        SelectionTool()
            : m_selType(NONE), m_mode(NODE)
        { }

        SelType type() const { return m_selType; }
        size_t  index() const { return m_selIndex; }
        SelType mode() const { return m_mode; }
        void clear() { m_selType = NONE; }

        void cycleMode() {
            m_mode = (SelType) ((m_mode + 1) % NUM_TYPES);
            if (m_mode == NONE) m_mode = NODE;
        }

        void select(size_t index) {
            m_selIndex = index;
            m_selType = m_mode;
        }

private:
        size_t m_selIndex;
        SelType m_selType;
        SelType m_mode;
    };

    typedef MeshlessFEM_t::SField SField;
    typedef MeshlessFEM_t::VField VField;
    typedef ResultsCollector_t::Result Result;

    FEMView2D(MeshlessFEM_t &fem, const ViewSettings &vs,
              QWidget *parent = NULL);
    ~FEMView2D() {
        // Clean up OpenCL stuff
        CALL_CL_GUARDED(clReleaseKernel, (m_renderKernel));
        CALL_CL_GUARDED(clReleaseKernel, (m_clearKernel));
        CALL_CL_GUARDED(clReleaseCommandQueue, (m_clQueue));
        CALL_CL_GUARDED(clReleaseContext, (m_clContext));

        CALL_CL_GUARDED(clReleaseMemObject, (m_nodeBuf));
        CALL_CL_GUARDED(clReleaseMemObject, (m_nodeHostBuf));
        CALL_CL_GUARDED(clReleaseMemObject, (m_primBuf));
        CALL_CL_GUARDED(clReleaseMemObject, (m_primHostBuf));
        if (m_modelTexBuf)
            CALL_CL_GUARDED(clReleaseMemObject, (m_modelTexBuf));
        if (m_overlayTexBuf)
            CALL_CL_GUARDED(clReleaseMemObject, (m_overlayTexBuf));
    }

    bool isVibrating() const {
        return (m_guiState == STATE_RESULT) &&
               (m_result) && (m_result->hasNodeVField()) && 
               (m_viewSettings.vfDisplayMode == ViewSettings::VFIELD_VIBRATE);
    }

    void setGUIState(GUIState state) {
        m_guiState = state;
        m_gesture = NONE;
        m_select.clear();

        update();

        if (isVibrating()) {
            m_timer.start(1000.0 / 60, this);
        }
        else {
            m_timer.stop();
        }
    }

    void displayResult(std::shared_ptr<const Result> r) {
        m_result = r;
        if (m_result) {
            setGUIState(STATE_RESULT);
        }
        else if (m_guiState == STATE_RESULT) {
            setGUIState(STATE_ELEMENTS);
        }
    }

    GUIState getGUIState() const {
        return m_guiState;
    }

    void setPressurePaintValue(double value) {
        m_pressurePaintValue = value;
    }

public slots:
    void csgNodesSelected(const NodeList &nList);
    void viewSettingsUpdated();

    // To be called, for instance, when new .csg is loaded.
    void modelChanged() {
        m_selectedObjects.clear();
        m_rerenderObject();
        m_rerenderOverlay();
        update();
    }

protected:
    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    template<typename Collection>
    void getClosest(const Collection &points, const Vector &pt,
                    size_t &closest, Scalar &sDist);
    void paintPressure(const Vector &screenPt, bool erase = false);
    void performSelection(const Vector &screenPt);
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event) {
        if (event->key() == Qt::Key_S)
            m_select.cycleMode();

        update();
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

    // Get the size of a pixel in world coordinates
    // Assumes pixel box is square in world coordinates.
    Scalar getPixelSize() const {
        return (m_frameMax[0] - m_frameMin[0]) / m_width;
    }

    // Non-rounded for opengl drawing
    void getScreenCoords(Scalar x, Scalar y, Scalar &sx, Scalar &sy) const {
        Scalar s, t;
        getTextureCoordinates(x, y, s, t);
        sx = m_width * s;
        sy = m_height * t;
    }

    void getScreenCoords(const Vector &world, Vector &screen) const
    {
        getScreenCoords(world[0], world[1], screen[0], screen[1]);
    }

    void qtToScreenCoords(const QPoint &pt, Vector &spt) const {
        spt[0] = pt.x() + m_screenLeft;
        spt[1] = m_screenTop - pt.y();
    }

    Vector qtToScreenCoords(const QPoint &pt) const {
        Vector spt;
        qtToScreenCoords(pt, spt);
        return spt;
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
            // Pass up the unhandled timer event
            QGLWidget::timerEvent(event);
        }
    }

private:
    void m_clClearCSGRender(cl_mem texBuf);
    void m_clRenderCSGNode(CSGNode *node, cl_mem texBuf, const QColor &fg);

    typedef enum {DRAW_CELLS, DRAW_NODES, DRAW_EDGES} DrawOp;

    ////////////////////////////////////////////////////////////////////////////
    /*! Draw all elements textured with a rendering of the element's part of the
    //  model. Also, color with the passed scalar field, if available.
    //  @param[in]  deformation     optional per-node displacement
    //  @param[in]  elemScalarField optional per-elem scalar field for shading
    //  @return     true if shading was done (and color legend is needed)
    *///////////////////////////////////////////////////////////////////////////
    bool drawObjectTextureCells(const VField &deformation = VField(),
                  const SField &elemScalarField = SField());
    void drawGrid(DrawOp op, const VField &deformation = VField(),
                  const SField &elemScalarField = SField());
    void drawBoundary(bool pressureField = false,
            const QColor &color = QColor(0, 0, 0),
            const QColor &selColor = QColor(0, 255, 0));
    void drawSelection(const VField &deformation = VField());
    void draw();
    bool m_drawResult();
    bool m_drawElements();
    void m_drawObject();
    void m_drawSelectedObjects();
    void m_rerenderObject();
    void m_rerenderOverlay();
    void m_drawWorldBox(const BBox_t &b);
    void m_drawWorldArrow(const Vector &p, const Vector &n,
                          Scalar length = 1.0);
    void m_drawWorldVertex(const Vector &v);
    void m_drawColorbar(float x, float y, float width, float height);

    ////////////////////////////////////////////////////////////////////////////
    // Instance variables
    ////////////////////////////////////////////////////////////////////////////
    FTGLBitmapFont m_font;
    Vector m_frameMin, m_frameMax;
    int m_width, m_height, m_screenTop, m_screenLeft;
    GLuint m_modelTex, m_overlayTex;
    GLuint m_bilinearShader;
    // Vertex coordinate attributes for bilinear displacement shader.
    GLuint m_vCoordLoc[4];
    // Texture coordinate attributes for bilinear displacement shader.
    GLuint m_tCoordLoc[4];
    // Object texture sampler loc
    GLuint m_objectTexLoc;

    MeshlessFEM_t &m_fem;
    NodeList m_selectedObjects;
    std::shared_ptr<const Result> m_result;
    VField m_activeDeformation;

    SelectionTool m_select;
    Scalar m_pressurePaintValue;

    GUIState m_guiState;
    typedef enum {DRAGGING, NONE} MouseGesture;
    MouseGesture m_gesture;
    QPoint m_prevMouseLoc;

    QBasicTimer m_timer;
    Scalar m_displacementPhase;

    const ViewSettings &m_viewSettings;
    ColorMap<RGBColorf, Scalar> m_scalarColorMap;

    ////////////////////////////////////////////////////////////////////////////
    // OpenCL stuff
    ////////////////////////////////////////////////////////////////////////////
    cl_context       m_clContext;
    cl_kernel        m_renderKernel;
    cl_kernel        m_clearKernel;
    cl_command_queue m_clQueue;
    cl_mem           m_nodeBuf, m_primBuf;
    cl_mem           m_nodeHostBuf, m_primHostBuf;
    cl_mem           m_modelTexBuf, m_overlayTexBuf;
};

#endif // FEMVIEW_HH
