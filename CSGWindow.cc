////////////////////////////////////////////////////////////////////////////////
// CSGWindow.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      The main window for CSG operations and visualization.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/28/2013 14:52:21
////////////////////////////////////////////////////////////////////////////////
#include "CSGWindow.hh"
#include "FEMView.hh"
#include <QtGui>

#include "CSGTree.hh"
#include "GlobalTypes.hh"
#include "ModelForm.hh"
#include "AnalysisSettings.hh"
#include "AnalysisForm.hh"

CSGWindow::CSGWindow(MeshlessFEM_t &fem, AnalysisSettings &settings)
{
    FEMView2D *femView = new FEMView2D(fem);
    femView->setMinimumSize(100, 100);
    QSplitter *splitter = new QSplitter();

    CSGTreeModel *treeModel = new CSGTreeModel(fem.model());
    QTreeView *treeView = new QTreeView();
    treeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    treeView->setModel(treeModel);

    controller = new CSGWindowController(treeModel, treeView, &fem.model(),
                                         femView, fem);

    ModelForm *modelForm = new ModelForm(controller);
    QWidget *sideBar = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout();
    QTabWidget *sideBarTab = new QTabWidget(sideBar);
    sideBarTab->addTab(modelForm, "Model");
    layout->addWidget(sideBarTab);
    sideBar->setLayout(layout);

    AnalysisForm *analysisForm = new AnalysisForm(settings, controller);
    sideBarTab->addTab(analysisForm, "Analyze");
    splitter->addWidget(sideBar);
    splitter->addWidget(femView);
    // splitter->setOrientation(Qt::Vertical);
    splitter->setCollapsible(0, false);
    splitter->setCollapsible(1, false);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);

    // Set up connections
    QObject::connect(controller, SIGNAL(csgNodesSelected(const NodeList &)),
                     femView, SLOT(csgNodesSelected(const NodeList &)));
    QObject::connect(sideBarTab, SIGNAL(currentChanged(int)),
                     controller, SLOT(changedSidebarTab(int)));
    QObject::connect(analysisForm,
                     SIGNAL(eqSettingsChanged(const AnalysisSettings &)),
                     controller,
                     SLOT(elementGridChanged(const AnalysisSettings &)));
    QObject::connect(analysisForm,
                     SIGNAL(matrixOrMaterialSettingsChanged(const AnalysisSettings &)),
                     controller,
                     SLOT(matrixOrMaterialSettingsChanged(const AnalysisSettings &)));
    QObject::connect(analysisForm,
                     SIGNAL(modalAnalysisSettingsChanged(const AnalysisSettings &)),
                     controller,
                     SLOT(modalAnalysisSettingsChanged(const AnalysisSettings &)));
    QObject::connect(controller, SIGNAL(modesUpdated(const MeshlessFEM_t *)),
                     analysisForm, SLOT(modesUpdated(const MeshlessFEM_t *)));

    setCentralWidget(splitter);

    QToolBar *tb = new QToolBar("Mouse Mode");
    QActionGroup *uiActionGroup = new QActionGroup(tb);
    QAction *panZoomAction   = new QAction("Pan/Zoom",  uiActionGroup);
    QAction *transformAction = new QAction("Transform", uiActionGroup);
    QAction *selectAction    = new QAction("Select",    uiActionGroup);
    panZoomAction->setCheckable(true);
    selectAction->setCheckable(true);
    transformAction->setCheckable(true);

    panZoomAction->setChecked(true);

    tb->addActions(uiActionGroup->actions());

    QMenu *viewMenu = menuBar()->addMenu("View");
    QAction *showGridAction = new QAction("Show Grid During Deformation", this);
    showGridAction->setCheckable(true);
    viewMenu->addAction(showGridAction);

    // tb->addAction(uiActionGroup);
    addToolBar(tb);
}
