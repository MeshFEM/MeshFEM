////////////////////////////////////////////////////////////////////////////////
// CSGWindowController.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Controller for the CSGWindow class (the main window).
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/30/2013 00:43:42
////////////////////////////////////////////////////////////////////////////////
#ifndef CSGWINDOW_CONTROLLER_HH
#define CSGWINDOW_CONTROLLER_HH
#include <QtGui>
#include "CSGTreeModel.hh"
#include "CSGTree.hh"
#include "GlobalTypes.hh"
#include "FEMView.hh"
#include "MeshlessFEM.hh"
#include "AnalysisSettings.hh"

class CSGWindowController : public QObject {
    Q_OBJECT

public:
    CSGWindowController(CSGTreeModel *treeModel, QTreeView *treeView,
                        CSGTree_t *tree, FEMView2D *femView,
                        MeshlessFEM_t &fem)
        : m_state(CONTROLLER_STATE_MODEL),
          m_csgTreeModel(treeModel), m_csgTreeView(treeView),
          m_csgTree(tree), m_femView(femView), m_fem(fem) { }

    QTreeView *csgTreeView()  { return m_csgTreeView; }

public slots:
    void changedSidebarTab(int newTab);
    // Modeling actions
    void csgTreeSelectionChanged(const QItemSelection &selected,
                                 const QItemSelection &deselected);
    void saveBoundaryPolygon();
    void loadCSG();
    void saveCSG();

    // Analysis actions
    void elementGridChanged(const AnalysisSettings &settings);
    void boundaryPointSettingsChanged(const AnalysisSettings &settings);
    void matrixOrMaterialSettingsChanged(const AnalysisSettings &settings);
    void modalAnalysisSettingsChanged(const AnalysisSettings &settings);
    void runModalAnalysis();
    void dumpModalData();
    void modeSelectionChanged(int index);

signals:
    void csgTreeApplyModifiedSelection(const QItemSelection &selection,
            QItemSelectionModel::SelectionFlags command =
            QItemSelectionModel::ClearAndSelect);
    void csgNodesSelected(const NodeList &nList);
    void modesUpdated(const MeshlessFEM_t *fem);
    
private:
    enum { CONTROLLER_STATE_MODEL, CONTROLLER_STATE_ANALYSIS } m_state;
    CSGTreeModel  *m_csgTreeModel;
    QTreeView     *m_csgTreeView;
    CSGTree_t     *m_csgTree;
    FEMView2D     *m_femView;
    MeshlessFEM_t &m_fem;

    typedef CSGTree_t::CSGNode CSGNode;

    void m_modesUpdated() {
    }
};

#endif // CSGWINDOW_CONTROLLER_HH
