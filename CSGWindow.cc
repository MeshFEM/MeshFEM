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

#include "CSGWindowController.hh"
#include "CSGTree.hh"
#include "GlobalTypes.hh"
#include "ModelForm.hh"
#include "AnalysisSettings.hh"
#include "AnalysisForm.hh"
#include "ViewSettings.hh"
#include "ViewSettingsWidget.hh"
#include "SolverLibrary.hh"

CSGWindow::CSGWindow(MeshlessFEM_t &fem, AnalysisSettings &settings,
                     SolverLibrary<Scalar> &solvers,
                     ResultsCollector_t &results)
{
    g_vsWidget = new ViewSettingsWidget(vsettings);
    g_vsWidget->setWindowTitle("View Settings");

    g_resultsWindow = new ResultsWindow(results);

    FEMView2D *femView = new FEMView2D(fem, vsettings);
    femView->setMinimumSize(100, 100);
    QSplitter *splitter = new QSplitter();

    CSGTreeModel *treeModel = new CSGTreeModel(fem.model());
    QTreeView *treeView = new QTreeView();
    treeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    treeView->setModel(treeModel);

    controller = new CSGWindowController(this, treeModel, treeView,
                                        &fem.model(), settings, femView, fem,
                                        results);

    ModelForm *modelForm = new ModelForm(controller);
    QWidget *sideBar = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout();
    QTabWidget *sideBarTab = new QTabWidget(sideBar);
    sideBarTab->addTab(modelForm, "Model");
    layout->addWidget(sideBarTab);
    sideBar->setLayout(layout);

    AnalysisForm *analysisForm = new AnalysisForm(settings, controller, solvers);
    QScrollArea *scroller = new QScrollArea();
    scroller->setWidget(analysisForm);
    scroller->setWidgetResizable(true);
    sideBarTab->addTab(scroller, "Analyze");
    splitter->addWidget(sideBar);
    splitter->addWidget(femView);
    // splitter->setOrientation(Qt::Vertical);
    splitter->setCollapsible(0, false);
    splitter->setCollapsible(1, false);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);

    // File Menu
    QMenu *fileMenu = menuBar()->addMenu("File");
    QAction *saveBoundaryAction = new QAction("Save &Boundary (.poly)", this);
    QAction *loadCSGAction = new QAction("&Open Object (.csg)", this);
    QAction *saveCSGAction = new QAction("&Save Object (.csg)", this);
    fileMenu->addAction(saveBoundaryAction);
    fileMenu->addAction(loadCSGAction);
    fileMenu->addAction(saveCSGAction);

    loadCSGAction->setShortcut(QKeySequence::Open);
    saveCSGAction->setShortcut(QKeySequence::Save);

    QObject::connect(saveBoundaryAction, SIGNAL(triggered()),
                     controller, SLOT(saveBoundaryPolygon()));
    QObject::connect(loadCSGAction, SIGNAL(triggered()),
                     controller, SLOT(loadCSG()));
    QObject::connect(saveCSGAction, SIGNAL(triggered()),
                     controller, SLOT(saveCSG()));

    // View Menu
    QMenu *viewMenu = menuBar()->addMenu("View");
    QAction *viewSettingsAction = new QAction("View Settings", this);
    QAction *resultsWindowAction = new QAction("Results Window", this);
    viewSettingsAction->setShortcut(Qt::CTRL + Qt::SHIFT + Qt::Key_V);
    viewMenu->addAction(viewSettingsAction);
    viewMenu->addAction(resultsWindowAction);
    QObject::connect(viewSettingsAction, SIGNAL(triggered()),
                     this, SLOT(showViewSettings()));
    QObject::connect(resultsWindowAction, SIGNAL(triggered()),
                     this, SLOT(showResultsWindow()));

    // GUI connections
    QObject::connect(controller, SIGNAL(csgNodesSelected(const NodeList &)),
                     femView, SLOT(csgNodesSelected(const NodeList &)));
    QObject::connect(sideBarTab, SIGNAL(currentChanged(int)),
                     controller, SLOT(changedSidebarTab(int)));
    QObject::connect(analysisForm,
                     SIGNAL(eqSettingsChanged(const AnalysisSettings &)),
                     controller,
                     SLOT(elementGridChanged(const AnalysisSettings &)));
    QObject::connect(analysisForm,
                     SIGNAL(bpSettingsChanged(const AnalysisSettings &)),
                     controller,
                     SLOT(boundaryPointSettingsChanged(const AnalysisSettings &)));
    QObject::connect(analysisForm,
                     SIGNAL(matrixOrMaterialSettingsChanged(const AnalysisSettings &)),
                     controller,
                     SLOT(matrixOrMaterialSettingsChanged(const AnalysisSettings &)));
    QObject::connect(analysisForm,
                     SIGNAL(modalAnalysisSettingsChanged(const AnalysisSettings &)),
                     controller,
                     SLOT(modalAnalysisSettingsChanged(const AnalysisSettings &)));
    QObject::connect(analysisForm,
                     SIGNAL(weaknessAnalysisSettingsChanged(const AnalysisSettings &)),
                     controller,
                     SLOT(weaknessAnalysisSettingsChanged(const AnalysisSettings &)));
    QObject::connect(g_vsWidget, SIGNAL(viewSettingsUpdated()),
                     femView, SLOT(viewSettingsUpdated()));
    QObject::connect(controller, SIGNAL(resultsUpdated()),
                     g_resultsWindow->controller(), SLOT(resultsUpdated()));

    QObject::connect(g_resultsWindow->controller(),
                     SIGNAL(resultSelected(const std::string &)),
                     controller, SLOT(resultSelected(const std::string &)));
    QObject::connect(controller, SIGNAL(reloadSettings()),
                     analysisForm, SLOT(reloadSettings()));

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

    // tb->addAction(uiActionGroup);
    addToolBar(tb);
}

CSGWindow::~CSGWindow()
{
    delete controller;
    delete g_vsWidget;
    delete g_resultsWindow;
}

void CSGWindow::showViewSettings()
{
    g_vsWidget->show();
    g_vsWidget->raise();
    g_vsWidget->activateWindow();
}

void CSGWindow::showResultsWindow()
{
    g_resultsWindow->show();
    g_resultsWindow->raise();
    g_resultsWindow->activateWindow();
}
