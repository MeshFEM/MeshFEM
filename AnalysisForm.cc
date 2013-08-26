////////////////////////////////////////////////////////////////////////////////
// AnalysisForm.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        The GUI holding all analysis settings.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/02/2013 00:47:54
////////////////////////////////////////////////////////////////////////////////
#include "AnalysisForm.hh"
#include <QtGui>
#include "QuadraturePointsSpinBox.hh"
#include "CSGWindowController.hh"
#include "SolverLibrary.hh"
#include <iostream>

AnalysisForm::AnalysisForm(AnalysisSettings &settings,
                           CSGWindowController *controller,
                           SolverLibrary<Scalar> &solvers, QWidget *parent)
    : QWidget(parent), m_settings(settings), m_solvers(solvers),
      m_settingGUIFromSettings(false)
{
    // Construct all widgets
    g_solverSelector = new QComboBox();

    g_nxStepper = new QSpinBox();
    g_nyStepper = new QSpinBox();
    g_borderWidthStepper = new QSpinBox();
    g_nxStepper->setMinimum(1);
    g_nyStepper->setMinimum(1);
    g_nxStepper->setMaximum(999);
    g_nyStepper->setMaximum(999);
    g_gaussQuadratureCheck = new QCheckBox();
    g_quadraturePointsStepper = new QuadraturePointsSpinBox();
    g_cellOverlapStepper = new QDoubleSpinBox();
    g_cellOverlapStepper->setMinimum(0.0);
    g_cellOverlapStepper->setMaximum(1.0);
    g_cellOverlapStepper->setSingleStep(.05);

    g_exactFullElementsCheck = new QCheckBox();
    g_antialiasedElementsCheck = new QCheckBox();

    g_massMatrixSelector = new QComboBox();
    g_youngModulusStepper = new QDoubleSpinBox();
    g_poissonRatioStepper = new QDoubleSpinBox();
    g_densityStepper = new QDoubleSpinBox();

    g_numModesStepper = new QSpinBox();
    g_laplacianModesCheck = new QCheckBox();
    g_modalAnalysisButton = new QPushButton("Modal Analysis");

    g_useMarchingSquaresCheck = new QCheckBox();
    g_boundaryPointStepper = new QDoubleSpinBox();
    g_boundaryPointStepper->setMinimum(0.01);
    g_boundaryPointStepper->setMaximum(1.0);
    g_boundaryPointStepper->setSingleStep(.01);
    g_kernelRadiusStepper = new QDoubleSpinBox();
    g_kernelRadiusStepper->setMinimum(0.01);
    g_kernelRadiusStepper->setMaximum(8.0);
    g_kernelRadiusStepper->setSingleStep(.01);
    g_configureSimulationButton = new QPushButton("Configure");
    g_savePressureButton = new QPushButton("Save P");
    g_loadPressureButton = new QPushButton("Load P");
    g_runSimulationButton = new QPushButton("Simulate");
    g_pressurePaintValueStepper = new QDoubleSpinBox();
    g_pressurePaintValueStepper->setSingleStep(.01);
    g_pressurePaintValueStepper->setMaximum(2.0);
    g_pressurePaintValueStepper->setValue(0.1);

    g_numWeakRegionsStepper = new QSpinBox();
    g_weaknessCutoffStepper = new QDoubleSpinBox();
    g_weakRegionExtractionButton = new QPushButton("Extract Weak Regions");
    g_pressureBoundStepper = new QDoubleSpinBox();
    g_forceBoundStepper = new QDoubleSpinBox();
    g_weaknessAnalysisButton = new QPushButton("Weakness Analysis");
    g_equalizeCombinedWeaknessCheck = new QCheckBox();
    g_pressureBoundStepper->setSingleStep(.01);
    g_pressureBoundStepper->setMaximum(5.0);
    g_pressureBoundStepper->setValue(.1);
    g_forceBoundStepper->setSingleStep(.01);
    g_forceBoundStepper->setMaximum(5.0);
    g_forceBoundStepper->setValue(.1);

    g_optimizeShapeButton = new QPushButton("Optimize Shape");
    g_xTranslationStepper = new QDoubleSpinBox();
    g_yTranslationStepper = new QDoubleSpinBox();
    g_translationFixedCheckbox = new QCheckBox();
    g_translationTestButton = new QPushButton("Weakness Translation Test");
    g_forceTranslationTestButton = new QPushButton("Force Translation Test");
    g_functionRadiusTestButton = new QPushButton("Function Radius Test");
    g_refinementTestButton = new QPushButton("Refinement Test");

    QGroupBox *solverGroup = new QGroupBox("Solvers");
    QGroupBox *elementsQuadratureGroup = new QGroupBox("Elements and Quadrature");
    QGroupBox *materialGroup = new QGroupBox("Materials and Matrices");
    QGroupBox *modalAnalysisGroup = new QGroupBox("Modal Analysis");
    QGroupBox *simulationGroup = new QGroupBox("Simulation");
    QGroupBox *weaknessAnalysisGroup = new QGroupBox("Weakness Analysis");

    // Solver Config
    QFormLayout *solverForm = new QFormLayout();
    solverForm->addRow("Solver", g_solverSelector);
    for (const std::string &name: solvers.names()) {
        g_solverSelector->addItem(name.c_str());
    }
    solverGroup->setLayout(solverForm);

    // Elements and Quadrature
    QFormLayout *eqForm = new QFormLayout();
    QHBoxLayout *rowColLayout = new QHBoxLayout();
    rowColLayout->addWidget(g_nyStepper);
    rowColLayout->addWidget(g_nxStepper);
    eqForm->addRow("Grid Rows/Cols", rowColLayout);
    eqForm->addRow("Border width", g_borderWidthStepper);
    eqForm->addRow("Gauss Quadrature", g_gaussQuadratureCheck);
    eqForm->addRow("Quadrature Points", g_quadraturePointsStepper);
    eqForm->addRow("Cell Overlap Threshold", g_cellOverlapStepper);
    eqForm->addRow("Exact Full Cell Integrals", g_exactFullElementsCheck);
    eqForm->addRow("Antialiased Cut Cells", g_antialiasedElementsCheck);
    elementsQuadratureGroup->setLayout(eqForm);

    // Material/Matrix Settings
    QFormLayout *matForm = new QFormLayout();
    matForm->addRow("Mass matrix", g_massMatrixSelector);
    g_massMatrixSelector->addItem("Full");
    g_massMatrixSelector->addItem("Lumped");
    g_massMatrixSelector->addItem("Quarter Cell");
    matForm->addRow("Young's Modulus", g_youngModulusStepper);
    matForm->addRow("Poisson Ratio", g_poissonRatioStepper);
    g_poissonRatioStepper->setMinimum(-1.0);
    g_poissonRatioStepper->setMaximum(0.5);
    matForm->addRow("Density", g_densityStepper);
    materialGroup->setLayout(matForm);

    // Modal Analysis
    QFormLayout *modalForm = new QFormLayout();
    modalForm->addRow("Number of Modes", g_numModesStepper);
    modalForm->addRow("Laplacian Modes", g_laplacianModesCheck);
    g_numModesStepper->setMinimum(1);
    g_numModesStepper->setMaximum(50);
    modalForm->addRow(g_modalAnalysisButton);
    modalAnalysisGroup->setLayout(modalForm);

    // Simulation
    QFormLayout *simForm = new QFormLayout();
    simForm->addRow("Marching Squares Boundary", g_useMarchingSquaresCheck);
    simForm->addRow("Boundary Point Spacing", g_boundaryPointStepper);
    simForm->addRow("Blur Kernel Radius Scale", g_kernelRadiusStepper);
    QHBoxLayout *simButtonLayout = new QHBoxLayout();
    simButtonLayout->addWidget(g_configureSimulationButton);
    simButtonLayout->addWidget(g_runSimulationButton);
    simForm->addRow(simButtonLayout);

    QHBoxLayout *pressureButtonLayout = new QHBoxLayout();
    pressureButtonLayout->addWidget(g_savePressureButton);
    pressureButtonLayout->addWidget(g_loadPressureButton);
    simForm->addRow(pressureButtonLayout);

    simForm->addRow("PressurePaint Value", g_pressurePaintValueStepper);
    simulationGroup->setLayout(simForm);

    // Weakness Analysis
    QFormLayout *weakForm = new QFormLayout();
    weakForm->addRow("Weak Regions/Mode", g_numWeakRegionsStepper);
    weakForm->addRow("Weak Region Cutoff", g_weaknessCutoffStepper);
    weakForm->addRow(g_weakRegionExtractionButton);
    weakForm->addRow("Pressure Bound", g_pressureBoundStepper);
    weakForm->addRow("Total Force Bound", g_forceBoundStepper);
    weakForm->addRow("Equalize Weakness", g_equalizeCombinedWeaknessCheck);
    weakForm->addRow(g_weaknessAnalysisButton);
    weakForm->addRow(g_optimizeShapeButton);

    // Translation Test
    QHBoxLayout *ttestLayout = new QHBoxLayout();
    ttestLayout->addWidget(g_translationFixedCheckbox);
    ttestLayout->addWidget(g_xTranslationStepper);
    ttestLayout->addWidget(g_yTranslationStepper);
    weakForm->addRow("Fixed XY Trans", ttestLayout);
    weakForm->addRow(g_translationTestButton);
    weakForm->addRow(g_forceTranslationTestButton);
    weakForm->addRow(g_functionRadiusTestButton);
    weakForm->addRow(g_refinementTestButton);
    weaknessAnalysisGroup->setLayout(weakForm);

    // Initialize all the GUI values
    m_setGUIFromSettings();

    // Connections
    assert(controller);
    QObject::connect(g_solverSelector, SIGNAL(currentIndexChanged(int)),
            this, SLOT(solverControlsChanged(int)));
    QObject::connect(g_nxStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_nyStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_borderWidthStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_quadraturePointsStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_gaussQuadratureCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_cellOverlapStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(elementGridControlsChanged(double)));

    QObject::connect(g_exactFullElementsCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_antialiasedElementsCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));

    QObject::connect(g_massMatrixSelector, SIGNAL(currentIndexChanged(int)),
                     this, SLOT(matrixControlsChanged(int)));
    QObject::connect(g_youngModulusStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(materialControlsChanged(double)));
    QObject::connect(g_poissonRatioStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(materialControlsChanged(double)));
    QObject::connect(g_densityStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(materialControlsChanged(double)));

    QObject::connect(g_modalAnalysisButton, SIGNAL(clicked()),
                     controller, SLOT(runModalAnalysis()));
    QObject::connect(g_numModesStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(modalAnalysisControlsChanged(int)));
    QObject::connect(g_laplacianModesCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(modalAnalysisControlsChanged(int)));

    QObject::connect(g_useMarchingSquaresCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(boundaryPointControlsChanged(int)));
    QObject::connect(g_boundaryPointStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(boundaryPointControlsChanged(double)));
    QObject::connect(g_kernelRadiusStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(boundaryPointControlsChanged(double)));

    QObject::connect(g_configureSimulationButton, SIGNAL(clicked()),
                     controller, SLOT(configureSimulation()));
    QObject::connect(g_savePressureButton, SIGNAL(clicked()),
                     controller, SLOT(savePressure()));
    QObject::connect(g_loadPressureButton, SIGNAL(clicked()),
                     controller, SLOT(loadPressure()));
    QObject::connect(g_runSimulationButton, SIGNAL(clicked()),
                     controller, SLOT(runSimulation()));
    QObject::connect(g_pressurePaintValueStepper, SIGNAL(valueChanged(double)),
                     controller, SLOT(pressurePaintValueChanged(double)));

    QObject::connect(g_numWeakRegionsStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(weaknessAnalysisControlsChanged(int)));
    QObject::connect(g_weaknessCutoffStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(weaknessAnalysisControlsChanged(double)));
    QObject::connect(g_weakRegionExtractionButton, SIGNAL(clicked()),
                     controller, SLOT(runWeakRegionExtraction()));
    QObject::connect(g_weaknessAnalysisButton, SIGNAL(clicked()),
                     controller, SLOT(runWeaknessAnalysis()));
    QObject::connect(g_pressureBoundStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(weaknessAnalysisControlsChanged(double)));
    QObject::connect(g_forceBoundStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(weaknessAnalysisControlsChanged(double)));
    QObject::connect(g_equalizeCombinedWeaknessCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(weaknessAnalysisControlsChanged(int)));

    QObject::connect(g_optimizeShapeButton, SIGNAL(clicked()),
                     controller, SLOT(runShapeOptimization()));
    QObject::connect(g_translationFixedCheckbox, SIGNAL(stateChanged(int)),
                     this, SLOT(ttestControlsChanged(int)));
    QObject::connect(g_xTranslationStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(ttestControlsChanged(double)));
    QObject::connect(g_yTranslationStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(ttestControlsChanged(double)));
    QObject::connect(g_translationTestButton, SIGNAL(clicked()),
                     this, SLOT(ttestButtonClicked()));
    QObject::connect(g_forceTranslationTestButton, SIGNAL(clicked()),
                     this, SLOT(fttestButtonClicked()));
    QObject::connect(g_functionRadiusTestButton, SIGNAL(clicked()),
                     this, SLOT(frtestButtonClicked()));
    QObject::connect(g_refinementTestButton, SIGNAL(clicked()),
                     this, SLOT(reftestButtonClicked()));

    QObject::connect(this, SIGNAL(runTranslationTest(const AnalysisSettings &)),
                     controller, SLOT(runTranslationTest(const AnalysisSettings &)));
    QObject::connect(this, SIGNAL(runForceTranslationTest(const AnalysisSettings &)),
                     controller, SLOT(runForceTranslationTest(const AnalysisSettings &)));
    QObject::connect(this, SIGNAL(runFunctionRadiusTest(const AnalysisSettings &)),
                     controller, SLOT(runFunctionRadiusTest(const AnalysisSettings &)));
    QObject::connect(this, SIGNAL(runRefinementTest()),
                     controller, SLOT(runRefinementTest()));

    // Layout all the groups
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(solverGroup);
    layout->addWidget(elementsQuadratureGroup);
    layout->addWidget(materialGroup);
    layout->addWidget(modalAnalysisGroup);
    layout->addWidget(simulationGroup);
    layout->addWidget(weaknessAnalysisGroup);
    layout->addStretch(1.0);
    layout->setContentsMargins(5, 5, 5, 5);

    // QVBoxLayout *fullLayout = new QVBoxLayout();
    // QScrollArea *scrollArea = new QScrollArea(this);
    // scrollArea->setWidget(layout);
    // fullLayout->addWidget(scrollArea);
    // setLayout(fullLayout);

    setLayout(layout);
}

void AnalysisForm::m_setGUIFromSettings() {
    m_settingGUIFromSettings = true;

    m_solvers.selectSolver(m_settings.solver);
    g_solverSelector->setCurrentIndex(m_solvers.selectedIndex());
    g_nxStepper->setValue(m_settings.Nx);
    g_nyStepper->setValue(m_settings.Ny);
    g_borderWidthStepper->setValue(m_settings.borderWidth);
    g_gaussQuadratureCheck->setChecked(m_settings.quadrature ==
                                       GAUSS_QUADRATURE);
    g_quadraturePointsStepper->setValue(m_settings.quadraturePoints);

    g_exactFullElementsCheck->setChecked(m_settings.exactFullElements);
    g_antialiasedElementsCheck->setChecked(m_settings.antialiasedElements);

    g_laplacianModesCheck->setChecked(m_settings.laplacianModes);
    g_numModesStepper->setValue(m_settings.numModes);
    g_cellOverlapStepper->setValue(m_settings.cellOverlapThreshold);
    g_useMarchingSquaresCheck->setChecked(m_settings.useMSBoundary);
    g_boundaryPointStepper->setValue(m_settings.boundarySpacing);
    g_kernelRadiusStepper->setValue(m_settings.kernelRadius);

    // Note: assumes MassMatrixType enum index matches combo box index
    g_massMatrixSelector->setCurrentIndex(m_settings.massMatrixType);
    g_youngModulusStepper->setValue(m_settings.young_modulus);
    g_poissonRatioStepper->setValue(m_settings.poisson_ratio);
    g_densityStepper->setValue(m_settings.density);

    g_numWeakRegionsStepper->setValue(m_settings.weakRegionsPerMode);
    g_weaknessCutoffStepper->setValue(m_settings.weaknessCutoff);

    g_forceBoundStepper->setValue(m_settings.totalForceBound);
    g_pressureBoundStepper->setValue(m_settings.pointwisePressureBound);

    g_equalizeCombinedWeaknessCheck->setChecked(m_settings.equalizeCombinedWeakness);

    // translation test
    g_translationFixedCheckbox->setChecked(m_settings.fixedTranslation);
    g_xTranslationStepper->setValue(m_settings.xTranslation);
    g_yTranslationStepper->setValue(m_settings.yTranslation);
    if (m_settings.fixedTranslation) {
        g_xTranslationStepper->setEnabled(true);
        g_yTranslationStepper->setEnabled(true);
    }
    else {
        g_xTranslationStepper->setEnabled(false);
        g_yTranslationStepper->setEnabled(false);
    }

    m_settingGUIFromSettings = false;
}

void AnalysisForm::m_readSettingsFromGUI() {
    // Never read settings from the gui controls while we're setting the GUI
    // controls... (Setting the controls will spawn signals that in turn call
    // this function).
    if (m_settingGUIFromSettings)
        return;

    m_solvers.selectSolver(g_solverSelector->currentIndex());
    m_settings.solver = m_solvers.selectedName();
    m_settings.Nx = g_nxStepper->value();
    m_settings.Ny = g_nyStepper->value();
    m_settings.borderWidth = g_borderWidthStepper->value();
    m_settings.quadrature = g_gaussQuadratureCheck->isChecked()
                                    ? GAUSS_QUADRATURE : UNIFORM_QUADRATURE;
    m_settings.quadraturePoints = g_quadraturePointsStepper->value();

    m_settings.laplacianModes = g_laplacianModesCheck->isChecked();
    m_settings.numModes = g_numModesStepper->value();
    m_settings.cellOverlapThreshold = g_cellOverlapStepper->value();

    m_settings.exactFullElements = g_exactFullElementsCheck->isChecked();
    m_settings.antialiasedElements = g_antialiasedElementsCheck->isChecked();

    // Note: assumes MassMatrixType enum index matches combo box index
    m_settings.massMatrixType =
        (MassMatrixType) g_massMatrixSelector->currentIndex();
    m_settings.young_modulus = g_youngModulusStepper->value();
    m_settings.poisson_ratio = g_poissonRatioStepper->value();
    m_settings.density       = g_densityStepper->value();

    m_settings.useMSBoundary   = g_useMarchingSquaresCheck->isChecked();
    m_settings.boundarySpacing = g_boundaryPointStepper->value();
    m_settings.kernelRadius    = g_kernelRadiusStepper->value();

    m_settings.weakRegionsPerMode = g_numWeakRegionsStepper->value();
    m_settings.weaknessCutoff = g_weaknessCutoffStepper->value();

    m_settings.totalForceBound = g_forceBoundStepper->value();
    m_settings.pointwisePressureBound = g_pressureBoundStepper->value();
    m_settings.equalizeCombinedWeakness = g_equalizeCombinedWeaknessCheck->isChecked();

    // translation test
    m_settings.fixedTranslation = g_translationFixedCheckbox->isChecked();
    m_settings.xTranslation = g_xTranslationStepper->value();
    m_settings.yTranslation = g_yTranslationStepper->value();
}

void AnalysisForm::solverControlsChanged(int) {
    m_readSettingsFromGUI();
    // Changing the solver doesn't invalidate anything, so we needn't emit a
    // notification.
}

void AnalysisForm::elementGridControlsChanged(int) {
    m_readSettingsFromGUI();
    emit eqSettingsChanged(m_settings);
}

void AnalysisForm::elementGridControlsChanged(double) {
    elementGridControlsChanged((int) 0);
}

void AnalysisForm::boundaryPointControlsChanged(double) {
    m_readSettingsFromGUI();
    emit bpSettingsChanged(m_settings);
}

void AnalysisForm::boundaryPointControlsChanged(int) {
    m_readSettingsFromGUI();
    emit bpSettingsChanged(m_settings);
}

void AnalysisForm::modalAnalysisControlsChanged(int) {
    m_readSettingsFromGUI();
    emit modalAnalysisSettingsChanged(m_settings);
}

void AnalysisForm::matrixControlsChanged(int) {
    m_readSettingsFromGUI();
    emit matrixOrMaterialSettingsChanged(m_settings);
}

void AnalysisForm::materialControlsChanged(double) {
    m_readSettingsFromGUI();
    emit matrixOrMaterialSettingsChanged(m_settings);
}

void AnalysisForm::weaknessAnalysisControlsChanged(int) {
    m_readSettingsFromGUI();
    emit weaknessAnalysisSettingsChanged(m_settings);
}

void AnalysisForm::weaknessAnalysisControlsChanged(double) {
    m_readSettingsFromGUI();
    emit weaknessAnalysisSettingsChanged(m_settings);
}

void AnalysisForm::ttestControlsChanged(int) {
    m_readSettingsFromGUI();
    if (g_translationFixedCheckbox->isChecked()) {
        g_xTranslationStepper->setEnabled(true);
        g_yTranslationStepper->setEnabled(true);
    }
    else {
        g_xTranslationStepper->setEnabled(false);
        g_yTranslationStepper->setEnabled(false);
    }
}

void AnalysisForm::ttestControlsChanged(double) {
    m_readSettingsFromGUI();
}

void AnalysisForm::ttestButtonClicked() {
    emit runTranslationTest(m_settings);
}

void AnalysisForm::fttestButtonClicked() {
    emit runForceTranslationTest(m_settings);
}

void AnalysisForm::frtestButtonClicked() {
    emit runFunctionRadiusTest(m_settings);
}

void AnalysisForm::reftestButtonClicked() {
    emit runRefinementTest();
}

void AnalysisForm::reloadSettings() {
    m_setGUIFromSettings();
}
