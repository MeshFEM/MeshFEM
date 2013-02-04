////////////////////////////////////////////////////////////////////////////////
// CSGWindowController.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Controller for the CSGWindow class (the main window).
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/30/2013 00:58:08
////////////////////////////////////////////////////////////////////////////////
#include "CSGWindowController.hh"
#include <list>
#include <iostream>

struct NodeAccumulator {
    typedef NodeList::iterator iterator;

    void preVisit(CSGNode *node) {
        nodes.push_back(node);
    }

    void postVisit(CSGNode *node) { }

    iterator begin() { return nodes.begin(); }
    iterator end()   { return nodes.end(); }

    NodeList nodes;
};

void CSGWindowController::csgTreeSelectionChanged(
                const QItemSelection &selected,
                const QItemSelection &deselected) 
{
    // Extend selection to all children
    QItemSelection fullSelection;
    NodeAccumulator accum;
    foreach (QModelIndex selIndex,
             m_csgTreeView->selectionModel()->selectedIndexes()) {
        CSGNode *selRoot = m_csgTreeModel->getNode(selIndex);
        accum = m_csgTree->dfs(accum, selRoot);
    }
    for (NodeAccumulator::iterator it = accum.begin(); it != accum.end();
         ++it) {
        QModelIndex idx = m_csgTreeModel->getIndex(*it);
        fullSelection.merge(QItemSelection(idx, idx),
                            QItemSelectionModel::Select);
    }

    // Now, only select the root of each selection component
    QItemSelection newSelection;
    NodeList nList;
    for (NodeAccumulator::iterator it = accum.begin(); it != accum.end();
         ++it) {
        QModelIndex idx = m_csgTreeModel->getIndex(*it);
        if (!fullSelection.contains(m_csgTreeModel->parent(idx)) &&
            !newSelection.contains(idx)) {
            newSelection.select(idx, idx);
            nList.push_back(m_csgTreeModel->getNode(idx));
        }
    }

    emit csgTreeApplyModifiedSelection(newSelection);
    emit csgNodesSelected(nList);

    // m_csgGLView->setSelectedCSGNodes()
}

void CSGWindowController::changedSidebarTab(int newTab) {
    if (newTab == 0)
        m_femView->setGUIState(FEMView2D::MODEL_STATE);
    else {
        m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
        m_fem.elementGrid().update();
    }
}
