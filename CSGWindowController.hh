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

class CSGWindowController : public QObject {
    Q_OBJECT

public:
    CSGWindowController(CSGTreeModel *treeModel, QTreeView *treeView,
                        CSGTree_t *tree)
        : m_csgTreeModel(treeModel),
          m_csgTreeView(treeView), m_csgTree(tree) { }

public slots:
    void csgTreeSelectionChanged(const QItemSelection &selected,
                                 const QItemSelection &deselected);
signals:
    void csgTreeApplyModifiedSelection(const QItemSelection &selection,
            QItemSelectionModel::SelectionFlags command =
            QItemSelectionModel::ClearAndSelect);
    void csgNodesSelected(const NodeList &nList);
    
private:
    CSGTreeModel *m_csgTreeModel;
    QTreeView    *m_csgTreeView;
    CSGTree_t    *m_csgTree;

    typedef CSGTree_t::CSGNode CSGNode;
};

#endif // CSGWINDOW_CONTROLLER_HH
