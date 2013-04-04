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
#include "CSGFile.hh"
#include <list>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <QMessageBox>

using namespace std;

void CSGWindowController::changedSidebarTab(int newTab) {
    if (newTab == 0) {
        m_femView->setGUIState(FEMView2D::MODEL_STATE);
        m_state = CONTROLLER_STATE_MODEL;
    }
    else {
        // The model might have changed--notify m_fem
        m_fem.modelChanged();
        emit modesUpdated(&m_fem);
        m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
        m_state = CONTROLLER_STATE_ANALYSIS;
    }
}

struct NodeAccumulator {
    typedef NodeList::iterator iterator;

    void preVisit(CSGNode *node) {
        nodes.push_back(node);
    }

    void postVisit(CSGNode *) { }

    iterator begin() { return nodes.begin(); }
    iterator end()   { return nodes.end(); }

    NodeList nodes;
};

void CSGWindowController::csgTreeSelectionChanged(
                const QItemSelection &selected,
                const QItemSelection &deselected) 
{
    // We must process the full selection, so ignore the deltas
    Q_UNUSED(selected)
    Q_UNUSED(deselected)

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
    MarchingSquaresGrid ms(m_fem.elementGrid().cols(),
                           m_fem.elementGrid().rows());
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

    QString fileName = QFileDialog::getSaveFileName(0,
            "Save Boundary Polygon (.poly)", QString(), "Text files (*.poly)");
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

void CSGWindowController::saveCSG()
{
    QString fileName = QFileDialog::getSaveFileName(0, "Save Object (.csg)",
            QString(), "Text files (*.csg)");
    if (fileName.length() > 0) {
        try {
            writeCSGFile(fileName.toAscii(), *m_csgTree);
        }
        catch (std::exception &e)
        {
            QMessageBox mbox(QMessageBox::Critical,
                    e.what(), e.what(),
                    QMessageBox::Ok);
            mbox.setDefaultButton(QMessageBox::Ok);
            mbox.exec();
        }
    }
}

void CSGWindowController::loadCSG()
{
    QString fileName = QFileDialog::getOpenFileName(0, "Open Object (.csg)",
            QString(), "Text files (*.csg)");
    if (fileName.length() > 0) {
        m_csgTreeModel->csgTreeAboutToUpdate();

        try {
            parseCSGFile(fileName.toAscii(), *m_csgTree);
        }
        catch (std::exception &e)
        {
            QMessageBox mbox(QMessageBox::Critical,
                    e.what(), e.what(),
                    QMessageBox::Ok);
            mbox.setDefaultButton(QMessageBox::Ok);
            mbox.exec();
        }

        m_csgTreeModel->csgTreeUpdated();
        m_fem.modelChanged();
        m_femView->modelChanged();
        // If we're in the analysis state, we must update the modes and return
        // to the element grid display
        if (m_state == CONTROLLER_STATE_ANALYSIS) {
            emit modesUpdated(&m_fem);
            m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
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
boundaryPointSettingsChanged(const AnalysisSettings &settings)
{
    // Go back to the element state.
    m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    m_fem.configureBoundaryPoints(settings);
    m_femView->update();
    // Currently, configuring the boundary points clears all modes
    emit modesUpdated(&m_fem);
    emit weakRegionsUpdated(&m_fem);
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
    emit weakRegionsUpdated(&m_fem);
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
    emit weakRegionsUpdated(&m_fem);
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

void CSGWindowController::configureSimulation()
{
    m_femView->setGUIState(FEMView2D::SIM_SETUP_STATE);
}

void CSGWindowController::loadPressure()
{
    QString fileName = QFileDialog::getOpenFileName(0, "Open boundary pressures (.bp)",
            QString(), "Text files (*.bp)");
    if (fileName.length() > 0) {
        try {
            std::ifstream bpFile(fileName.toStdString().c_str());
            if (!bpFile.is_open()) {
                throw std::runtime_error(std::string("Couldn't open file: ") +
                                         fileName.toStdString());
            }
            size_t boundarySize;
            bpFile >> boundarySize;

            if (boundarySize != m_fem.numBoundaryPoints())
                throw std::runtime_error(std::string("Boundary count mismatch"));

            const std::vector<MeshlessFEM_t::_BoundaryPoint> &bpts =
                    m_fem.boundaryPoints();
            std::vector<bool> mapped(boundarySize, false);

            for (size_t i = 0; i < boundarySize; ++i) {
                Vector p;
                Scalar pressure;
                bpFile >> p[0] >> p[1] >> pressure;
                size_t closest = 0;
                Scalar closestDist = std::numeric_limits<Scalar>::max();
                for (size_t j = 0; j < bpts.size(); ++j) {
                    Scalar dist = (bpts[j].p - p).norm();
                    if (dist < closestDist) {
                        closestDist = dist;
                        closest = j;
                    }
                }
                if (mapped[closest])

                    throw std::runtime_error(std::string("Mapping not bijective"));
                mapped[closest] = true;
                // Note: Python StructAys uses negative pressures.
                m_fem.pressure(closest) = -pressure;
            }
            if (!bpFile)
                throw std::runtime_error(std::string("Error reading file"));

            m_femView->setGUIState(FEMView2D::SIM_SETUP_STATE);
        }
        catch (std::exception &e)
        {
            QMessageBox mbox(QMessageBox::Critical,
                    e.what(), e.what(),
                    QMessageBox::Ok);
            mbox.setDefaultButton(QMessageBox::Ok);
            mbox.exec();
        }
    }
}

void CSGWindowController::runSimulation()
{
    bool success = m_fem.simulate();
    if (!success) {
        QMessageBox mbox(QMessageBox::Critical,
                "Simulation Failed",
                "Error: Simulation failed.",
                QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.exec();
        m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    }
    else {
        m_femView->setGUIState(FEMView2D::SIM_RESULT_STATE);
    }
}

void CSGWindowController::pressurePaintValueChanged(double value)
{
    m_femView->setPressurePaintValue(value);
}

void CSGWindowController::
weaknessAnalysisSettingsChanged(const AnalysisSettings &settings)
{
    m_fem.configureWeaknessAnalysis(settings);
    m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    emit weakRegionsUpdated(&m_fem);
}

void CSGWindowController::runWeakRegionExtraction()
{
    int ret = m_fem.weakRegionExtraction();
    if (ret >= 0)
        emit weakRegionsUpdated(&m_fem);
    if (ret == 1)
        emit modesUpdated(&m_fem);
}

void CSGWindowController::runWeaknessAnalysis()
{
    bool success = m_fem.weaknessAnalysis();
    if (!success) {
        QMessageBox mbox(QMessageBox::Critical,
                "Weakness Analysis Failed",
                "Error: Weakness Analysis failed.",
                QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.exec();
    }
}

void CSGWindowController::dumpModalData()
{
    QString fileName = QFileDialog::getSaveFileName(0, "Save Modal Data (.msh)",
            QString(), "Text files (*.msh)");
    if (fileName.length() > 0) {
        typedef MSHWriter<MeshlessFEM_t::ElementGrid> MSHWriter_t;
        MSHWriter_t mshOut(fileName.toAscii(), m_fem.elementGrid());
        if (mshOut) {
            VectorField<double, 3> modal3Vector(m_fem.elementGrid().numNodes());
            modal3Vector.clear();
            for (size_t i = 0; i < m_fem.numModes(); ++i) {
                char name[64];
                const MeshlessFEM_t::VField &mode = m_fem.mode(i);
                assert(mode.domainSize() == modal3Vector.domainSize());

                for (size_t j = 0; j < mode.domainSize(); ++i) {
                    modal3Vector(j)[0] = mode(j)[0];
                    modal3Vector(j)[1] = mode(j)[1];
                }

                snprintf(name, 64, "modal displacement %i", (int) i);
                mshOut.addField(name, modal3Vector, MSHWriter_t::PER_NODE);
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
        m_femView->setGUIState(FEMView2D::MODE_STATE);
    }
    else {
        m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    }
}
