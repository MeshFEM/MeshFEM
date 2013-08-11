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
#include <string>
#include "CSGWindow.hh"
#include "CSGTreeModel.hh"
#include "CSGTree.hh"
#include "GlobalTypes.hh"
#include "FEMView.hh"
#include "MeshlessFEM.hh"
#include "AnalysisSettings.hh"

class CSGWindowController : public QObject {
    Q_OBJECT

public:
    CSGWindowController(CSGWindow *window, CSGTreeModel *treeModel,
                        QTreeView *treeView,
                        CSGTree_t *tree, AnalysisSettings &settings,
                        FEMView2D *femView, MeshlessFEM_t &fem,
                        ResultsCollector_t &results)
        : m_state(CONTROLLER_STATE_MODEL), m_window(window),
          m_csgTreeModel(treeModel), m_csgTreeView(treeView),
          m_csgTree(tree), m_settings(settings), m_femView(femView), m_fem(fem),
          m_results(results), m_modelName("Untitled Model"),
          m_settingsName("Untitled Settings") { }

    QTreeView *csgTreeView()  { return m_csgTreeView; }

public slots:
    void changedSidebarTab(int newTab);
    // Modeling actions
    void csgTreeSelectionChanged(const QItemSelection &selected,
                                 const QItemSelection &deselected);
    void saveBoundaryPolygon();
    void loadCSG();
    void saveCSG();

    void modelChanged(bool refitGrid = true);

    // Analysis actions
    void elementGridChanged(const AnalysisSettings &settings);
    void boundaryPointSettingsChanged(const AnalysisSettings &settings);
    void matrixOrMaterialSettingsChanged(const AnalysisSettings &settings);
    void modalAnalysisSettingsChanged(const AnalysisSettings &settings);
    void runModalAnalysis();

    // Simulation actions
    void configureSimulation();
    void savePressure();
    void loadPressure();
    void runSimulation();
    void pressurePaintValueChanged(double);

    // Weakness analysis actions
    void weaknessAnalysisSettingsChanged(const AnalysisSettings &settings);
    void runWeakRegionExtraction();
    void runWeaknessAnalysis();

    // Shape optimization actions
    void runShapeOptimization();
    void runTranslationTest(const AnalysisSettings &settings);
    void runForceTranslationTest(const AnalysisSettings &settings);
    void runFunctionRadiusTest(const AnalysisSettings &settings);
    void runRefinementTest();

    void resultSelected(const std::string &path);
    void resultDeslected();

signals:
    void csgTreeApplyModifiedSelection(const QItemSelection &selection,
            QItemSelectionModel::SelectionFlags command =
            QItemSelectionModel::ClearAndSelect);
    void csgNodesSelected(const NodeList &nList);
    void resultsUpdated();
    void reloadSettings();
    
private:
    void prepareResultsCollector();

    enum { CONTROLLER_STATE_MODEL, CONTROLLER_STATE_ANALYSIS } m_state;
    CSGWindow           *m_window;
    CSGTreeModel        *m_csgTreeModel;
    QTreeView           *m_csgTreeView;
    CSGTree_t           *m_csgTree;
    AnalysisSettings    &m_settings;
    FEMView2D           *m_femView;
    MeshlessFEM_t       &m_fem;
    ResultsCollector_t  &m_results;
    std::string          m_modelName;
    std::string          m_settingsName;
    std::string          m_csgPath;

    typedef CSGTree_t::CSGNode CSGNode;
};

#endif // CSGWINDOW_CONTROLLER_HH
