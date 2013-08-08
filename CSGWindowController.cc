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
#include "ElementGrid.hh"
#include "Fields.hh"
#include <list>
#include <vector>
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
        modelChanged();
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
    QString path = QString::fromStdString(m_csgPath);
    QString fileName = QFileDialog::getSaveFileName(m_window,
            "Save Object (.csg)", path,
            "Text files (*.csg)");
    if (fileName.length() > 0) {
        try {
            writeCSGFile(fileName.toAscii(), *m_csgTree);
            m_csgPath = fileName.toStdString();
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
    QString path = QString::fromStdString(m_csgPath);
    QString fileName = QFileDialog::getOpenFileName(m_window,
            "Open Object (.csg)", path,
            "Text files (*.csg)");
    if (fileName.length() > 0) {
        m_csgTreeModel->csgTreeAboutToUpdate();

        try {
            parseCSGFile(fileName.toAscii(), *m_csgTree);
            m_csgPath = fileName.toStdString();
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
        modelChanged();
        // If we're in the analysis state, we must return to the element grid
        // display
        if (m_state == CONTROLLER_STATE_ANALYSIS) {
            m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
        }
    }
}

void CSGWindowController::modelChanged(bool refitGrid)
{
    m_fem.modelChanged(refitGrid);
    m_femView->modelChanged();
}


void CSGWindowController::elementGridChanged(const AnalysisSettings &settings)
{
    // When the grid changes, we must go back to the element state.
    m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    if (m_fem.configureElements(settings))
        m_femView->update();
    // Configuring the elements clears all modes and weak regions
}

void CSGWindowController::
boundaryPointSettingsChanged(const AnalysisSettings &settings)
{
    // Go back to the element state.
    m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    m_fem.configureBoundaryPoints(settings);
    m_femView->update();
    // Currently, configuring the boundary points clears all modes
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
}

void CSGWindowController::
modalAnalysisSettingsChanged(const AnalysisSettings &settings)
{
    // When the modal analysis settings change, we must go back to the element
    // state.
    m_femView->setGUIState(FEMView2D::ELEMENTS_STATE);
    m_fem.configureModalAnalysis(settings);
    // Configuring modal analysis settings clears all modes
}

// Prepare the results collector for adding results by inserting the current
// model/settings.
void CSGWindowController::prepareResultsCollector() {
    m_settingsName = m_results.addSettings(m_settingsName, m_settings);
    m_modelName = m_results.addModel(m_modelName, *m_csgTree,
                            m_fem.elementGrid().getBoudingBox());
}

void CSGWindowController::runModalAnalysis()
{
    prepareResultsCollector();

    bool success = m_fem.modalAnalysis(&m_results);
    if (!success) {
        QMessageBox mbox(QMessageBox::Critical,
                "Modal analysis Failed",
                "Error: Modal analysis failed.",
                QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.exec();
    }
    emit resultsUpdated();
}

void CSGWindowController::configureSimulation()
{
    m_femView->setGUIState(FEMView2D::PRESSURE_DRAW_STATE);
}

void CSGWindowController::savePressure()
{
    QString fileName = QFileDialog::getSaveFileName(0,
            "Save Boundary Pressures (.bp)", QString(), "Text files (*.bp)");
    if (fileName.length() > 0) {
        try {
            std::ofstream bpFile(fileName.toStdString().c_str());
            if (!bpFile.is_open()) {
                throw std::runtime_error(std::string("Couldn't open file: ") +
                                         fileName.toStdString());
            }

            const std::vector<MeshlessFEM_t::_BoundaryPoint> &bpts =
                    m_fem.boundaryPoints();
            bpFile << bpts.size() << endl;
            
            for (size_t i = 0; i < bpts.size(); ++i) {
                // Note: Python StructAys uses negative pressures, so our bp
                // format takes this convention.
                Scalar p = -m_fem.pressure(i);
                bpFile << bpts[i].p[0] << "\t" << bpts[i].p[1] << p << endl;
            }
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

            m_femView->setGUIState(FEMView2D::PRESSURE_DRAW_STATE);
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
    prepareResultsCollector();
    bool success = m_fem.simulate(&m_results);
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
        emit resultsUpdated();
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
}

void CSGWindowController::runWeakRegionExtraction()
{
    int ret = m_fem.weakRegionExtraction();
}

void CSGWindowController::runWeaknessAnalysis()
{
    Scalar weakness;
    prepareResultsCollector();
    bool success = m_fem.weaknessAnalysis(weakness, &m_results);
    if (!success) {
        QMessageBox mbox(QMessageBox::Critical,
                "Weakness Analysis Failed",
                "Error: Weakness Analysis failed.",
                QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.exec();
    }

    emit resultsUpdated();
}

// void CSGWindowController::dumpModalData()
// {
//     QString fileName = QFileDialog::getSaveFileName(0, "Save Modal Data (.msh)",
//             QString(), "Text files (*.msh)");
//     if (fileName.length() > 0) {
//         typedef MSHWriter<MeshlessFEM_t::ElementGrid> MSHWriter_t;
//         MSHWriter_t mshOut(fileName.toAscii(), m_fem.elementGrid());
//         if (mshOut) {
//             VectorField<double, 3> modal3Vector(m_fem.elementGrid().numNodes());
//             modal3Vector.clear();
//             for (size_t i = 0; i < m_fem.numModes(); ++i) {
//                 char name[64];
//                 const MeshlessFEM_t::VField &mode = m_fem.mode(i);
//                 assert(mode.domainSize() == modal3Vector.domainSize());
// 
//                 for (size_t j = 0; j < mode.domainSize(); ++i) {
//                     modal3Vector(j)[0] = mode(j)[0];
//                     modal3Vector(j)[1] = mode(j)[1];
//                 }
// 
//                 snprintf(name, 64, "modal displacement %i", (int) i);
//                 mshOut.addField(name, modal3Vector, MSHWriter_t::PER_NODE);
//                 snprintf(name, 64, "modal stress norm %i", (int) i);
//                 mshOut.addField(name, m_fem.modalStressNorms(i),
//                                 MSHWriter_t::PER_ELEMENT);
//             }
//         }
//     }
// }

void CSGWindowController::runShapeOptimization()
{

    Scalar weakness;
    bool success = m_fem.weaknessAnalysis(weakness);
    assert(success);

    // compute gradient
    Scalar delta = .01;
    std::vector<Scalar> params = m_csgTree->getParameters();
    cout << "Optimizing over " << params.size() << " parameters." << endl;
    DVector grad(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        Scalar old = params[i];
        params[i] = old + delta;

        m_csgTree->setParameters(params);
        modelChanged();
        Scalar weaknessPerturb;
        success = m_fem.weaknessAnalysis(weaknessPerturb);
        assert(success);
        grad[i] = (weaknessPerturb - weakness) / delta;

        params[i] = old;
    }

    if (grad.norm() > 0) {
        grad /= grad.norm();
        for (size_t i = 0; i < params.size(); ++i) {
            params[i] -= .125 * grad[i];
        }
    }
    cout << "gradient: " << grad << endl;
    
    m_csgTree->setParameters(params);
    modelChanged();
}


void CSGWindowController::runTranslationTest(const AnalysisSettings &settings)
{
    Scalar weakness;
    // bool success = m_fem.weaknessAnalysis(weakness);
    // cout << "Translation test" << endl << "----------------------" << endl;
    // cout << weakness << endl;
    // assert(success);

    Vector cellSize = m_fem.elementGrid().cellSize();
    const int TRANS_TEST_STEPS = 5;
    Vector cellDelta = cellSize * (1.0 / TRANS_TEST_STEPS);

    std::vector<Scalar> params = m_csgTree->getParameters();
    std::vector<Scalar> translated(params);
    // Center, dimensions, rotation
    assert(params.size() % 5 == 0);
    size_t numPrimitives = params.size() / 5;

    // typedef std::pair<ElementGrid2D<CSGTree_t>, ScalarField<Scalar> > GridField;
    // std::list<GridField> *weaknessGrids = new std::list<GridField>();

    if (settings.fixedTranslation) {
        Vector offset(cellSize[0] * settings.xTranslation,
                      cellSize[1] * settings.yTranslation);
        std::cout << "Offset: " << offset << endl;
        for (size_t p = 0; p < numPrimitives; ++p) {
            translated[5 * p + 0] = params[5 * p + 0] + offset[0];
            translated[5 * p + 1] = params[5 * p + 1] + offset[1];
        }

        m_csgTree->setParameters(translated);
        modelChanged(false);

        QString cwPath, cwPercentilePath;
        cwPath.sprintf("ftranslation_%f_%f.cw", (float) settings.xTranslation,
                       (float) settings.yTranslation);
        cwPercentilePath.sprintf("ftranslation_%f_%f.cwp",
                (float) settings.xTranslation, (float) settings.yTranslation);

        prepareResultsCollector();
        bool success = m_fem.weaknessAnalysis(weakness, &m_results);
        assert(success);
    }
    else {
        for (int xStep = 0; xStep < TRANS_TEST_STEPS; ++xStep) {
            for (int yStep = 0; yStep < TRANS_TEST_STEPS; ++yStep) {
                Vector offset(cellDelta[0] * xStep, cellDelta[1] * yStep);
                std::cout << "Offset: " << offset << endl;
                
                for (size_t p = 0; p < numPrimitives; ++p) {
                    translated[5 * p + 0] = params[5 * p + 0] + offset[0];
                    translated[5 * p + 1] = params[5 * p + 1] + offset[1];
                }

                m_csgTree->setParameters(translated);
                modelChanged(false);

                QString cwPath, cwPercentilePath;
                cwPath.sprintf("translation_%i_%i.cw", xStep, yStep);
                cwPercentilePath.sprintf("translation_%i_%i.cwp", xStep, yStep);

                prepareResultsCollector();
                bool success = m_fem.weaknessAnalysis(weakness, &m_results);

                // weaknessGrids->push_back(make_pair(m_fem.elementGrid(),
                //             m_fem.combinedWeakness()));
                            
                assert(success);
            }
        }
    }

    // m_femView->loadGridFields(weaknessGrids);

    // Note the grid must now be manually recreated from the bounding box to
    // repeat experiments.

    // m_fem.modelChanged(false);
    m_femView->modelChanged();

    // m_csgTree->setParameters(params);
}

void CSGWindowController::
runForceTranslationTest(const AnalysisSettings &settings)
{
    Vector cellSize = m_fem.elementGrid().cellSize();
    const int TRANS_TEST_STEPS = 5;
    Vector cellDelta = cellSize * (1.0 / TRANS_TEST_STEPS);

    std::vector<Scalar> params = m_csgTree->getParameters();
    std::vector<Scalar> translated(params);
    // Center, dimensions, rotation
    assert(params.size() % 5 == 0);
    size_t numPrimitives = params.size() / 5;

    if (settings.fixedTranslation) {
        Vector offset(cellSize[0] * settings.xTranslation,
                      cellSize[1] * settings.yTranslation);
        std::cout << "Offset: " << offset << endl;
        for (size_t p = 0; p < numPrimitives; ++p) {
            translated[5 * p + 0] = params[5 * p + 0] + offset[0];
            translated[5 * p + 1] = params[5 * p + 1] + offset[1];
        }

        m_csgTree->setParameters(translated);
        modelChanged(false);

        QString simPath;
        simPath.sprintf("sim_ftranslation_%f_%f.msh",
                       (float) settings.xTranslation,
                       (float) settings.yTranslation);

        // bool success = m_fem.simulate(simPath.toAscii());
        prepareResultsCollector();
        bool success = m_fem.simulate(&m_results);
        assert(success);
    }
    else {
        for (int xStep = 0; xStep < TRANS_TEST_STEPS; ++xStep) {
            for (int yStep = 0; yStep < TRANS_TEST_STEPS; ++yStep) {
                Vector offset(cellDelta[0] * xStep, cellDelta[1] * yStep);
                std::cout << "Offset: " << offset << endl;
                
                for (size_t p = 0; p < numPrimitives; ++p) {
                    translated[5 * p + 0] = params[5 * p + 0] + offset[0];
                    translated[5 * p + 1] = params[5 * p + 1] + offset[1];
                }

                m_csgTree->setParameters(translated);
                modelChanged(false);

                QString simPath;
                simPath.sprintf("sim_translation_%i_%i.msh", xStep, yStep);

                // bool success = m_fem.simulate(simPath.toAscii());
                prepareResultsCollector();
                bool success = m_fem.simulate(&m_results);

                assert(success);
            }
        }
    }
}

void CSGWindowController::
runFunctionRadiusTest(const AnalysisSettings &settings)
{
    const int RADIUS_TEST_STEPS = 15;
    Scalar minScale = 0.1;
    Scalar maxScale = 4.0;
    Scalar scaleDelta = (maxScale - minScale) / RADIUS_TEST_STEPS;
    
    Scalar rScale = minScale;
    for (size_t i = 0; i < RADIUS_TEST_STEPS; ++i) {
        rScale += scaleDelta;
        m_fem.buildBoundaryFunctions(rScale);

        QString simPath;
        simPath.sprintf("sim_radius_%f.msh", rScale);

        // bool success = m_fem.simulate(simPath.toAscii());
        prepareResultsCollector();
        bool success = m_fem.simulate(&m_results);
        assert(success);
    }
}

void CSGWindowController::
runRefinementTest(const AnalysisSettings &settings)
{
    vector<Scalar> pressures(m_fem.numBoundaryPoints());
    for (size_t i = 0; i < pressures.size(); ++i) {
        pressures[i] = m_fem.pressure(i);
    }

    const int REFINEMENT_TEST_STEPS = 40;
    Scalar minScale = 0.1;
    Scalar maxScale = 8.0;
    Scalar scaleDelta = (maxScale - minScale) / REFINEMENT_TEST_STEPS;
    
    AnalysisSettings newSettings = settings;
    Scalar scale = minScale;
    for (size_t i = 0; i < REFINEMENT_TEST_STEPS; ++i) {
        scale += scaleDelta;
        newSettings.Nx = settings.Nx * scale;
        newSettings.Ny = settings.Ny * scale;
        m_fem.configureElements(newSettings);
        m_fem.setPressures(pressures);

        QString simPath;
        simPath.sprintf("sim_refinement_%i_%i.msh", (int) newSettings.Nx,
                        (int) newSettings.Ny);

        prepareResultsCollector();
        // bool success = m_fem.simulate(simPath.toAscii());
        bool success = m_fem.simulate(&m_results);
        assert(success);
    }

    m_fem.configureElements(settings);
    m_fem.setPressures(pressures);
}

void CSGWindowController::resultSelected(const string &resultPath)
{
    std::string modelName = getModelPathComponent(resultPath);
    std::string settingsName = getSettingsPathComponent(resultPath);

    BBox_t bbox = m_fem.elementGrid().getBoudingBox();
    if (m_results.modelDiffers(modelName, m_fem.model(), bbox)) {
        m_results.getModel(modelName, m_fem.model(), bbox);
        m_fem.elementGrid().setBoundingBox(bbox);
        modelChanged(false);
    }

    if (m_results.settingsDiffer(settingsName, m_settings)) {
        m_results.getSettings(settingsName, m_settings);
        emit reloadSettings();
    }

    m_femView->displayResult(m_results.getResultWithPath(resultPath));
}
