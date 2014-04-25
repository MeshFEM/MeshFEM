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
#include "ParameterSweepDialog.hh"
#include <list>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <limits>
#include <QMessageBox>
#include <QTreeview>
#include <QFileDialog>
#include <boost/format.hpp>

using namespace std;

void CSGWindowController::changedSidebarTab(int newTab) {
    if (newTab == 0) {
        m_femView->setGUIState(FEMView2D::STATE_MODEL);
        m_state = CONTROLLER_STATE_MODEL;
    }
    else {
        modelChanged();
        m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
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
        ofstream polygonOut(fileName.toLatin1());
        if (!polygonOut.is_open()) {
            QString errorMsg;
            errorMsg.sprintf("Error: couldn't open file '%s' for writing.",
                             (const char *) fileName.toLatin1());

            QMessageBox mbox(QMessageBox::Critical,
                    "Save Boundary Polygon Failed",
                    errorMsg, QMessageBox::Ok);
            mbox.setDefaultButton(QMessageBox::Ok);
            mbox.exec();
        }
        else {
            polygonOut << polygons;
        }
    }
}

void CSGWindowController::saveCSG()
{
    QString path = QString();
    QString fileName = QFileDialog::getSaveFileName(m_window,
            "Save Object (.csg)", path,
            "Text files (*.csg)");
    if (fileName.length() > 0) {
        try {
            writeCSGFile(fileName.toStdString(), *m_csgTree);
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

void CSGWindowController::loadCSG(QString path)
{
    if (path.length() == 0) {
        path = QFileDialog::getOpenFileName(m_window, "Open Object (.csg)",
                QString(), "Text files (*.csg)");
    }
    if (path.length() > 0) {
        m_csgTreeModel->csgTreeAboutToUpdate();

        try {
            parseCSGFile(path.toStdString(), *m_csgTree);
            QFileInfo fi(path);
            string modelName = fi.completeBaseName().toStdString();
            if (m_modelName != modelName) {
                m_modelName = modelName;
                emit namesUpdated(m_modelName, m_settingsName);
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

        m_csgTreeModel->csgTreeUpdated();
        modelChanged();
        // If we're in the analysis state, we must return to the element grid
        // display
        if (m_state == CONTROLLER_STATE_ANALYSIS) {
            m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
        }
    }
}

void CSGWindowController::loadSettings()
{
    QString path = QFileDialog::getOpenFileName(m_window, "Open Settings (.cfg)",
            QString(), "Text files (*.cfg)");
    if (path.length() > 0) {
        try {
            std::ifstream settingsFile(path.toStdString());
            m_settings.parseOptions(settingsFile);
            emit reloadSettings();

            QFileInfo fi(path);
            string settingsName = fi.completeBaseName().toStdString();
            if (m_settingsName != settingsName) {
                m_settingsName = settingsName;
                emit namesUpdated(m_modelName, m_settingsName);
            }

            validateNames();
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

void CSGWindowController::saveSettings()
{
    QString path = QFileDialog::getSaveFileName(m_window, "Save Settings (.cfg)",
            QString(), "Text files (*.cfg)");
    if (path.length() > 0) {
        try {
            std::ofstream settingsFile(path.toStdString());
            m_settings.writeOptions(settingsFile);
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

void CSGWindowController::modelChanged(bool refitGrid)
{
    bool locked = m_fem.elementGrid().boundingBoxIsLocked();
    m_fem.elementGrid().setBoundingBoxLocked(!refitGrid);
    m_fem.modelChanged();
    m_femView->modelChanged();
    m_fem.elementGrid().setBoundingBoxLocked(locked);
    validateNames();
}

void CSGWindowController::settingsChanged()
{
    validateNames();
}

void CSGWindowController::elementGridChanged(const AnalysisSettings &settings)
{
    // When the grid changes, we must go back to the element state.
    m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
    if (m_fem.configureElements(settings))
        m_femView->elementsChanged();
    // Configuring the elements clears all modes and weak regions
}

void CSGWindowController::
boundaryPointSettingsChanged(const AnalysisSettings &settings)
{
    // Go back to the element state.
    m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
    m_fem.configureBoundaryPoints(settings);
    m_femView->update();
}

void CSGWindowController::
matrixOrMaterialSettingsChanged(const AnalysisSettings &settings)
{
    // When the material settings change, we must go back to the element
    // state.
    m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
    m_fem.configureMaterial(settings);
    m_fem.configureMatrices(settings);
    // Configuring modal analysis settings clears all modes
}

void CSGWindowController::
modalAnalysisSettingsChanged(const AnalysisSettings &settings)
{
    // When the modal analysis settings change, we must go back to the element
    // state.
    m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
    m_fem.configureModalAnalysis(settings);
    // Configuring modal analysis settings clears all modes
}

// Prepare the results collector for adding results by inserting the current
// model/settings.
void CSGWindowController::prepareResultsCollector() {
    m_settingsName = m_results.addSettings(m_settingsName, m_settings);
    m_modelName = m_results.addModel(m_modelName, *m_csgTree,
                            m_fem.elementGrid().getBoundingBox());
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
    m_femView->setGUIState(FEMView2D::STATE_PRESSURE_DRAW);
}

void CSGWindowController::saveBC()
{
    QString fileName = QFileDialog::getSaveFileName(0,
            "Save Boundary Conditions (.bc)", QString(), "Text files (*.bc)");
    if (fileName.length() > 0) {
        try {
            m_fem.boundaryConditions().writeConditions(fileName.toStdString());
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

void CSGWindowController::loadBC()
{
    QString fileName = QFileDialog::getOpenFileName(0,
            "Load Boundary Conditions (.bc)", QString(), "Text files (*.bc)");
    if (fileName.length() > 0) {
        try {
            m_fem.boundaryConditions().readConditions(fileName.toStdString());
            m_femView->setGUIState(FEMView2D::STATE_PRESSURE_DRAW);
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

// void CSGWindowController::savePressure()
// {
//     QString fileName = QFileDialog::getSaveFileName(0,
//             "Save Boundary Pressures (.bp)", QString(), "Text files (*.bp)");
//     if (fileName.length() > 0) {
//         try {
//             std::ofstream bpFile(fileName.toStdString().c_str());
//             if (!bpFile.is_open()) {
//                 throw std::runtime_error(std::string("Couldn't open file: ") +
//                                          fileName.toStdString());
//             }
// 
//             const std::vector<MeshlessFEM_t::_BoundaryPoint> &bpts =
//                     m_fem.boundaryPoints();
//             bpFile << bpts.size() << endl;
//             
//             for (size_t i = 0; i < bpts.size(); ++i) {
//                 // Note: Python StructAys uses negative pressures, so our bp
//                 // format takes this convention.
//                 Scalar p = -m_fem.boundaryConditions().paintedPressure(i);
//                 bpFile << bpts[i].p[0] << '\t' << bpts[i].p[1] << '\t'
//                        << p << endl;
//             }
//         }
//         catch (std::exception &e)
//         {
//             QMessageBox mbox(QMessageBox::Critical,
//                     e.what(), e.what(),
//                     QMessageBox::Ok);
//             mbox.setDefaultButton(QMessageBox::Ok);
//             mbox.exec();
//         }
//     }
// }
// 
// void CSGWindowController::loadPressure()
// {
//     QString fileName = QFileDialog::getOpenFileName(0, "Open boundary pressures (.bp)",
//             QString(), "Text files (*.bp)");
//     if (fileName.length() > 0) {
//         try {
//             std::ifstream bpFile(fileName.toStdString().c_str());
//             if (!bpFile.is_open()) {
//                 throw std::runtime_error(std::string("Couldn't open file: ") +
//                                          fileName.toStdString());
//             }
//             size_t boundarySize;
//             bpFile >> boundarySize;
// 
//             if (boundarySize != m_fem.numBoundaryPoints())
//                 throw std::runtime_error(std::string("Boundary count mismatch"));
// 
//             const std::vector<MeshlessFEM_t::_BoundaryPoint> &bpts =
//                     m_fem.boundaryPoints();
//             std::vector<bool> mapped(boundarySize, false);
// 
//             for (size_t i = 0; i < boundarySize; ++i) {
//                 Vector p;
//                 Scalar pressure;
//                 bpFile >> p[0] >> p[1] >> pressure;
//                 size_t closest = 0;
//                 Scalar closestDist = std::numeric_limits<Scalar>::max();
//                 for (size_t j = 0; j < bpts.size(); ++j) {
//                     Scalar dist = (bpts[j].p - p).norm();
//                     if (dist < closestDist) {
//                         closestDist = dist;
//                         closest = j;
//                     }
//                 }
//                 if (mapped[closest])
// 
//                     throw std::runtime_error(std::string("Mapping not bijective"));
//                 mapped[closest] = true;
//                 // Note: Python StructAys uses negative pressures.
//                 m_fem.boundaryConditions().paintPressure(closest, -pressure);
//             }
//             if (!bpFile)
//                 throw std::runtime_error(std::string("Error reading file"));
// 
//             m_femView->setGUIState(FEMView2D::STATE_PRESSURE_DRAW);
//         }
//         catch (std::exception &e)
//         {
//             QMessageBox mbox(QMessageBox::Critical,
//                     e.what(), e.what(),
//                     QMessageBox::Ok);
//             mbox.setDefaultButton(QMessageBox::Ok);
//             mbox.exec();
//         }
//     }
// }

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
        m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
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
    m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
}

void CSGWindowController::runWeakRegionExtraction()
{
    m_fem.weakRegionExtraction();
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
//         MSHWriter_t mshOut(fileName.toLatin1(), m_fem.elementGrid());
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

    Vector cellSize = m_fem.elementGrid().cellSize();
    const int TRANS_TEST_STEPS = 5;
    Vector cellDelta = cellSize * (1.0 / TRANS_TEST_STEPS);

    std::vector<Scalar> params = m_csgTree->getParameters();
    std::vector<Scalar> translated(params);
    // Center, dimensions, rotation
    assert(params.size() % 5 == 0);
    size_t numPrimitives = params.size() / 5;

    std::string baseName = m_modelName;
    boost::format formatter("%s + (%f, %f)");

    if (settings.Bool("fixedTranslation")) {
        Vector offset(cellSize[0] * settings.Real("xTranslation"),
                      cellSize[1] * settings.Real("yTranslation"));
        std::cout << "Offset: " << offset << endl;
        for (size_t p = 0; p < numPrimitives; ++p) {
            translated[5 * p + 0] = params[5 * p + 0] + offset[0];
            translated[5 * p + 1] = params[5 * p + 1] + offset[1];
        }

        m_csgTree->setParameters(translated);
        modelChanged(false);

        m_modelName = boost::str(formatter % baseName % settings.Real("xTranslation") %
                                 settings.Real("yTranslation"));

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

                m_modelName = boost::str(formatter % baseName % offset[0] %
                                         offset[1]);

                prepareResultsCollector();
                bool success = m_fem.weaknessAnalysis(weakness, &m_results);

                assert(success);
            }
        }
    }

    m_femView->modelChanged();

    m_modelName = baseName;

    // m_csgTree->setParameters(params);

    emit resultsUpdated();
}

void CSGWindowController::runSimulationSweep() {
    vector<string> settingsNames;
    for (const string &name : m_settings.getNames()) {
        if ((m_settings.type(name) == AnalysisSettings::TYPE_REAL)
            || (m_settings.type(name) == AnalysisSettings::TYPE_INT))
            settingsNames.push_back(name);
    }

    vector<string> csgParameterNames = m_csgTree->getParameterNames();

    ParameterSweepDialog *pdialog =
        new ParameterSweepDialog(m_modelName, m_settingsName, settingsNames,
                                 csgParameterNames, m_window);
    int ret = pdialog->exec();
    if (ret) {
        vector<string> settingNames, settingRanges, csgParameterRanges;
        vector<size_t> csgParameterIndices;
        pdialog->selectedIdentifiersAndRanges(settingNames, settingRanges,
                csgParameterIndices, csgParameterRanges);

        boost::format modelNameFormat(pdialog->modelNameFormat());
        boost::format settingsNameFormat(pdialog->settingsNameFormat());
        modelNameFormat.exceptions(boost::io::all_error_bits ^
            (boost::io::too_many_args_bit | boost::io::too_few_args_bit));
        settingsNameFormat.exceptions(boost::io::all_error_bits ^
            (boost::io::too_many_args_bit | boost::io::too_few_args_bit));

        typedef ParameterSweep<Scalar> PS;
        PS::SweepMode mode = (PS::SweepMode) pdialog->sweepMode();
            assert((mode == PS::SWEEP_ZIP) || (mode == PS::SWEEP_PRODUCT));

        auto_ptr<PS> ps;
        try {
            ps = auto_ptr<PS>(new PS(mode, settingNames, settingRanges,
                                    csgParameterIndices, csgParameterRanges));
        }
        catch (exception &e) {
            string errorMsg("Sweep Configuration Failed: ");
            errorMsg += e.what();

            QMessageBox mbox(QMessageBox::Critical,
                             "Sweep Configuration Failed",
                             errorMsg.c_str(), QMessageBox::Ok);
            mbox.setDefaultButton(QMessageBox::Ok);
            mbox.exec();

            return;
        }

        bool running = true;
        QString dir;
        if (pdialog->operation() == ParameterSweepDialog::SWEEP_OP_SAVE) {
            running = false;
            dir = QFileDialog::getExistingDirectory(0,
                "Inputs Save Directory", QString(), QFileDialog::ShowDirsOnly);

            if (dir.length() == 0)
                return;

            // TODO: write boundary conditions!
        }

        // Get the current parameters--the sweep will apply diffs to these.
        vector<Scalar> params = m_csgTree->getParameters();
        AnalysisSettings settings = m_settings;
        vector<Scalar> origParams(params);

        int frame = 0;
        int failures = 0;
        do {
            vector<Scalar> settingValues = ps->getSettingValues();
            vector<Scalar> csgParamValues = ps->getCSGParameterValues();

            assert(settingValues.size() == settingNames.size());
            for (size_t i = 0; i < settingValues.size(); ++i) {
                const string &setting = settingNames[i];
                Scalar value = settingValues[i];
                switch (settings.type(setting)) {
                    case AnalysisSettings::TYPE_REAL:
                        settings.Real(setting) = value;
                        break;
                    case AnalysisSettings::TYPE_INT:
                        value = std::round(value);
                        settings.Int(setting) = (int) value;
                        break;
                    default:
                        assert(false);
                }
                modelNameFormat % value;
                settingsNameFormat % value;
            }

            assert(csgParamValues.size() == csgParameterIndices.size());
            for (size_t i = 0; i < csgParamValues.size(); ++i) {
                Scalar value = csgParamValues[i];
                assert(csgParameterIndices[i] < params.size());
                params[csgParameterIndices[i]] = value;
                modelNameFormat % value;
                settingsNameFormat % value;
            }
            m_csgTree->setParameters(params);

            string modelName = boost::str(modelNameFormat);
            string settingsName = boost::str(settingsNameFormat);
            cout << "(" << modelName << ", " << settingsName << ")" << endl;

            if (running) {
                m_settings = settings;
                m_fem.elementGrid().setUpdatesEnabled(false);
                modelChanged();
                settingsChanged();
                m_fem.loadSettings(m_settings);
                m_fem.elementGrid().setUpdatesEnabled(true);
                if (m_fem.elementGrid().updatePending())
                    m_fem.elementGrid().update();

                m_modelName = modelName;
                m_settingsName = settingsName;
                prepareResultsCollector();

                try {
                    m_fem.simulate(&m_results);
                }
                catch (exception &e) {
                    ++failures;
                }
            }
            else {
                // Save frame inputs
                string baseName = dir.toStdString() + "/" + to_string(frame); 
                writeCSGFile(baseName + ".csg", *m_csgTree);
                ofstream infoFile(baseName + ".txt");
                infoFile << modelName << endl << settingsName << endl;
                ofstream settingsFile(baseName + ".cfg");
                settings.writeOptions(settingsFile);
            }

            ++frame;
        } while (ps->advance());

        if (running) {
            if (failures > 0) {
                QString errorMsg = QString::number(failures);
                errorMsg += " simulations failed.";

                QMessageBox mbox(QMessageBox::Critical, "Simulation Failed",
                        errorMsg, QMessageBox::Ok);
                mbox.setDefaultButton(QMessageBox::Ok);
                mbox.exec();
            }

            emit resultsUpdated();
        }
        else {
            // Settings weren't altered, but CSG tree was
            m_csgTree->setParameters(origParams);
        }
    }
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

    std::string baseName = m_modelName;
    boost::format formatter("%s + (%f, %f)");

    if (settings.Bool("fixedTranslation")) {
        Vector offset(cellSize[0] * settings.Real("xTranslation"),
                      cellSize[1] * settings.Real("yTranslation"));

        for (size_t p = 0; p < numPrimitives; ++p) {
            translated[5 * p + 0] = params[5 * p + 0] + offset[0];
            translated[5 * p + 1] = params[5 * p + 1] + offset[1];
        }

        m_csgTree->setParameters(translated);
        modelChanged(false);

        m_modelName = boost::str(formatter % baseName % settings.Real("xTranslation") %
                                 settings.Real("yTranslation"));

        // bool success = m_fem.simulate(simPath.toLatin1());
        prepareResultsCollector();
        bool success = m_fem.simulate(&m_results);
        assert(success);
    }
    else {
        for (int xStep = 0; xStep < TRANS_TEST_STEPS; ++xStep) {
            for (int yStep = 0; yStep < TRANS_TEST_STEPS; ++yStep) {
                Vector offset(cellDelta[0] * xStep, cellDelta[1] * yStep);
                
                for (size_t p = 0; p < numPrimitives; ++p) {
                    translated[5 * p + 0] = params[5 * p + 0] + offset[0];
                    translated[5 * p + 1] = params[5 * p + 1] + offset[1];
                }

                m_csgTree->setParameters(translated);
                modelChanged(false);

                m_modelName = boost::str(formatter % baseName % offset[0] %
                                         offset[1]);

                // bool success = m_fem.simulate(simPath.toLatin1());
                prepareResultsCollector();
                bool success = m_fem.simulate(&m_results);

                assert(success);
            }
        }
    }

    m_modelName = baseName;

    emit resultsUpdated();
}

void CSGWindowController::
runFunctionRadiusTest(const AnalysisSettings &/*settings*/)
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

        // bool success = m_fem.simulate(simPath.toLatin1());
        prepareResultsCollector();
        bool success = m_fem.simulate(&m_results);
        assert(success);
    }
}

void CSGWindowController::
runRefinementTest()
{
    const int REFINEMENT_TEST_STEPS = 10;
    Scalar minScale = 0.1;
    Scalar maxScale = 8.0;
    Scalar scaleDelta = (maxScale - minScale) / REFINEMENT_TEST_STEPS;
    boost::format formatter("%s (x %f)");
    
    AnalysisSettings oldSettings = m_settings;
    string oldSettingsName = m_settingsName;

    Scalar scale = minScale;
    for (size_t i = 0; i < REFINEMENT_TEST_STEPS; ++i) {
        scale += scaleDelta;
        m_settings.Int("Nx") = oldSettings.Int("Nx") * scale;
        m_settings.Int("Ny") = oldSettings.Int("Ny") * scale;
        m_fem.configureElements(m_settings);

        m_settingsName = boost::str(formatter % oldSettingsName % scale);

        prepareResultsCollector();
        // bool success = m_fem.simulate(simPath.toLatin1());
        bool success = m_fem.simulate(&m_results);
        assert(success);
    }

    m_settingsName = oldSettingsName;

    m_fem.configureElements(oldSettings);

    emit resultsUpdated();
}

void CSGWindowController::resultSelected(const string &resultPath)
{
    std::string modelName = getModelPathComponent(resultPath);
    std::string settingsName = getSettingsPathComponent(resultPath);
    MeshlessFEM_t::ElementGrid &grid = m_fem.elementGrid();
    grid.setUpdatesEnabled(false);
    modelSelected(modelName);
    settingsSelected(settingsName);
    grid.setUpdatesEnabled(true);
    
    std::shared_ptr<const ResultsCollector_t::Result> r =
                m_results.getResultWithPath(resultPath);

    if (grid.updatePending()) {
        grid.update(r->cellOverlaps());
        size_t expectedNodes = r->numNodes(), expectedElems = r->numElems();
        bool rebuild = false;
        if ((expectedNodes > 0) && (expectedNodes != grid.numNodes())) {
            rebuild = true;
            cout << "WARNING: result's cellTypes gave incompatible nodes! "
                 << "Rebuilding element grid." << endl;
        }
        if ((expectedElems > 0) && (expectedElems != grid.numElements())) {
            rebuild = true;
            cout << "WARNING: result's cellTypes gave incompatible elems! "
                 << "Rebuilding element grid." << endl;
        }
        if (rebuild)
            grid.update();
    }

    m_femView->displayResult(r);
}

void CSGWindowController::resultDeslected()
{
    m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);
}

void CSGWindowController::modelSelected(const string &name) {
    BBox_t bbox = m_fem.elementGrid().getBoundingBox();
    if (m_results.modelDiffers(m_fem.model(), bbox, name)) {
        m_results.getModel(m_fem.model(), bbox, name);
        m_fem.elementGrid().setBoundingBox(bbox);
        modelChanged(false);
    }

    if (m_modelName != name) {
        m_modelName = name;
        emit namesUpdated(m_modelName, m_settingsName);
    }

    m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);

    validateNames();
}

void CSGWindowController::settingsSelected(const string &name) {
    if (m_results.settingsDiffer(m_settings, name)) {
        bool locked = m_fem.elementGrid().boundingBoxIsLocked();
        m_fem.elementGrid().setBoundingBoxLocked(true);
        m_results.getSettings(m_settings, name);
        emit reloadSettings();
        m_fem.elementGrid().setBoundingBoxLocked(locked);
    }

    if (m_settingsName != name) {
        m_settingsName = name;
        emit namesUpdated(m_modelName, m_settingsName);
    }

    m_femView->setGUIState(FEMView2D::STATE_ELEMENTS);

    validateNames();
}

void CSGWindowController::modelNameEdited(const QString &name)
{
    m_modelName = name.toStdString();
    validateNames();
}

void CSGWindowController::settingsNameEdited(const QString &name)
{
    m_settingsName = name.toStdString();
    validateNames();
}

void CSGWindowController::validateNames()
{
    bool modelConflict = m_results.modelNameConflict(m_modelName, m_fem.model(),
                                    m_fem.elementGrid().getBoundingBox());
    bool settingsConflict = m_results.settingsNameConflict(m_settingsName,
                                                           m_settings);
    // std::cout << "Validated names: " << modelConflict << ", "
    //           << settingsConflict << std::endl;
    emit nameConflictsUpdated(modelConflict, settingsConflict);
}
