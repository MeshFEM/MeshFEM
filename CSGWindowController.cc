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
#include "MarchingSquaresGrid.hh"
#include "MSHWriter.hh"
#include <list>
#include <fstream>
#include <iostream>
#include <QMessageBox>

using namespace std;

void CSGWindowController::changedSidebarTab(int newTab) {
    if (newTab == 0)
        m_femView->setGUIState(FEMView2D::MODEL_STATE);
    else {
        // The model might have changed--notify m_fem
        m_fem.modelChanged();
        emit modesUpdated(&m_fem);
        m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    }
}

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

void CSGWindowController::saveBoundaryPolygon()
{
    // Output resolution... this should probably be configurable
    size_t Nx = 400, Ny = 400;
    MarchingSquaresGrid ms(Nx, Ny);
    vector<Polygon_t> polygons;
    assert(m_csgTree != NULL);
    ms.extractBoundaryPolygons(*m_csgTree, polygons);
    
    if (polygons.size() == 0) {
        QMessageBox mbox(QMessageBox::Critical,
                "Save Boundary Polygon Failed",
                "Error: no geometry boundary found", QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.exec();
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(0, "Save Boundary Polygon",
            QString(), "Text files (*.poly)");
    if (fileName.length() > 0) {
        ofstream polygonOut(fileName.toAscii());
        if (!polygonOut.is_open()) {
            QString errorMsg;
            errorMsg.sprintf("Error: couldn't open file '%s' for writing.",
                             (const char *) fileName.toAscii());

            QMessageBox mbox(QMessageBox::Critical,
                    "Save Boundary Polygon Failed",
                    errorMsg, QMessageBox::Ok);
            mbox.setDefaultButton(QMessageBox::Ok);
            mbox.exec();
        }
        else {
            polygonOut << polygons[0];
        }
    }
}

void CSGWindowController::elementGridChanged(const AnalysisSettings &settings)
{
    // When the grid changes, we must go back to the element state.
    m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    if (m_fem.configureElements(settings))
        m_femView->update();
    // Configuring the elements clears all modes
    emit modesUpdated(&m_fem);
}

void CSGWindowController::
matrixOrMaterialSettingsChanged(const AnalysisSettings &settings)
{
    // When the material settings change, we must go back to the element
    // state.
    m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    m_fem.configureMaterial(settings);
    m_fem.configureMatrices(settings);
    // Configuring modal analysis settings clears all modes
    emit modesUpdated(&m_fem);
}

void CSGWindowController::
modalAnalysisSettingsChanged(const AnalysisSettings &settings)
{
    // When the modal analysis settings change, we must go back to the element
    // state.
    m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    m_fem.configureModalAnalysis(settings);
    // Configuring modal analysis settings clears all modes
    emit modesUpdated(&m_fem);
}

void CSGWindowController::runModalAnalysis()
{
    bool success = m_fem.modalAnalysis();
    if (!success) {
        QMessageBox mbox(QMessageBox::Critical,
                "Modal analysis Failed",
                "Error: Modal analysis failed.",
                QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.exec();
    }
    emit modesUpdated(&m_fem);
}

void CSGWindowController::dumpModalData()
{
    QString fileName = QFileDialog::getSaveFileName(0, "Save Modal Data (.msh)",
            QString(), "Text files (*.msh)");
    if (fileName.length() > 0) {
        typedef MSHWriter<MeshlessFEM_t::ElementGrid> MSHWriter_t;
        MSHWriter_t mshOut(fileName.toAscii(), m_fem.elementGrid());
        if (mshOut) {
            for (size_t i = 0; i < m_fem.numModes(); ++i) {
                char name[64];
                snprintf(name, 64, "modal displacement %i", (int) i);
                mshOut.addField(name, m_fem.mode(i), MSHWriter_t::PER_NODE);
                snprintf(name, 64, "modal stress norm %i", (int) i);
                mshOut.addField(name, m_fem.modalStressNorms(i),
                                MSHWriter_t::PER_ELEMENT);
            }
        }
    }
}

void CSGWindowController::modeSelectionChanged(int index)
{
    if (index > 0) {
        m_femView->selectDeformation(index - 1);
        m_femView->setGUIState(FEMView2D::DISPLACEMENTS_STATE);
    }
    else {
        m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    }
}
