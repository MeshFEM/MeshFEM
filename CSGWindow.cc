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
#include "CSGView.hh"
#include "CSGTreeModel.hh"
#include <QtGui>

#include "CSGTree.hh"
#include "GlobalTypes.hh"

CSGWindow::CSGWindow(CSGTree_t &csgTree)
{
    CSGView2D *csgView = new CSGView2D(csgTree);
    QSplitter *splitter = new QSplitter();

    CSGTreeModel *model = new CSGTreeModel(csgTree);
    QTreeView *treeView = new QTreeView();
    treeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    treeView->setModel(model);

    splitter->addWidget(treeView);
    splitter->addWidget(csgView);
    // splitter->setOrientation(Qt::Vertical);
    splitter->setCollapsible(0, false);

    controller = new CSGWindowController(model, treeView, &csgTree);
    QObject::connect(treeView->selectionModel(),
                     SIGNAL(selectionChanged(const QItemSelection &,
                                             const QItemSelection &)),
                     controller, SLOT(csgTreeSelectionChanged(
                                        const QItemSelection &,
                                        const QItemSelection &)));
    QObject::connect(controller, SIGNAL(csgTreeApplyModifiedSelection(
                                        const QItemSelection &,
                                        QItemSelectionModel::SelectionFlags)),
                     treeView->selectionModel(), SLOT(select(
                                        const QItemSelection &,
                                        QItemSelectionModel::SelectionFlags)));
    QObject::connect(controller, SIGNAL(csgNodesSelected(const NodeList &)),
                     csgView, SLOT(csgNodesSelected(const NodeList &)));

    setCentralWidget(splitter);
}
