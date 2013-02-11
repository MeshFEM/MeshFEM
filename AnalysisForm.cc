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
#include <iostream>

AnalysisForm::AnalysisForm(AnalysisSettings &settings,
                           CSGWindowController *controller, QWidget *parent)
    : QWidget(parent), m_settings(settings)
{
    // Construct all widgets
    g_nxStepper = new QSpinBox();
    g_nyStepper = new QSpinBox();
    g_nxStepper->setMinimum(1);
    g_nyStepper->setMinimum(1);
    g_lumpedMassCheck = new QCheckBox();
    g_gaussQuadratureCheck = new QCheckBox();
    g_quadraturePointsStepper = new QuadraturePointsSpinBox();

    g_youngModulusStepper = new QDoubleSpinBox();
    g_poissonRatioStepper = new QDoubleSpinBox();
    g_densityStepper = new QDoubleSpinBox();

    g_numModesStepper = new QSpinBox();
    g_modalAnalysisButton = new QPushButton("Modal Analysis");
    g_modeSelector = new QComboBox();

    g_numWeakRegionsStepper = new QSpinBox();
    modesUpdated(NULL);

    QGroupBox *elementsQuadratureGroup = new QGroupBox("Elements and Quadrature");
    QGroupBox *materialGroup = new QGroupBox("Material");
    QGroupBox *modalAnalysisGroup = new QGroupBox("Modal Analysis");
    QGroupBox *weaknessAnalysisGroup = new QGroupBox("Weakness Analysis");
    QGroupBox *simulationGroup = new QGroupBox("Simulation");

    // Elements and Quadrature
    QFormLayout *eqForm = new QFormLayout();
    eqForm->addRow("Number of Columns", g_nxStepper);
    eqForm->addRow("Number of Rows", g_nyStepper);
    eqForm->addRow("Lumped Mass", g_lumpedMassCheck);
    eqForm->addRow("Gauss Quadrature", g_gaussQuadratureCheck);
    eqForm->addRow("Quadrature Points", g_quadraturePointsStepper);
    elementsQuadratureGroup->setLayout(eqForm);

    // Material Settings
    QFormLayout *matForm = new QFormLayout();
    matForm->addRow("Young's Modulus", g_youngModulusStepper);
    matForm->addRow("Poisson Ratio", g_poissonRatioStepper);
    g_poissonRatioStepper->setMinimum(-1.0);
    g_poissonRatioStepper->setMaximum(0.5);
    matForm->addRow("Density", g_densityStepper);
    materialGroup->setLayout(matForm);

    // Modal Analysis
    QFormLayout *modalForm = new QFormLayout();
    modalForm->addRow("Number of Modes", g_numModesStepper);
    g_numModesStepper->setMinimum(1);
    g_numModesStepper->setMaximum(50);
    modalForm->addRow(g_modalAnalysisButton);
    modalForm->addRow(g_modeSelector);
    modalAnalysisGroup->setLayout(modalForm);

    // Initialize all the GUI values
    m_setGUIFromSettings();

    // Connections
    assert(controller);
    QObject::connect(g_nxStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_nyStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_quadraturePointsStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_gaussQuadratureCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));

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
    QObject::connect(g_modeSelector, SIGNAL(currentIndexChanged(int)),
                     controller, SLOT(modeSelectionChanged(int)));

    // Layout all the groups
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(elementsQuadratureGroup);
    layout->addWidget(materialGroup);
    layout->addWidget(modalAnalysisGroup);
    layout->addWidget(weaknessAnalysisGroup);
    layout->addWidget(simulationGroup);
    layout->addStretch();

    // QScrollArea *scrollArea = new QScrollArea(this);
    // scrollArea->setLayout(layout);
    setLayout(layout);
}

void AnalysisForm::m_setGUIFromSettings() {
    g_nxStepper->setValue(m_settings.Nx);
    g_nyStepper->setValue(m_settings.Ny);
    g_lumpedMassCheck->setChecked(m_settings.lumpedMass);
    g_gaussQuadratureCheck->setChecked(m_settings.quadrature ==
                                       GAUSS_QUADRATURE);
    g_quadraturePointsStepper->setValue(m_settings.quadraturePoints);

    g_numModesStepper->setValue(m_settings.numModes);   

    g_youngModulusStepper->setValue(m_settings.young_modulus);
    g_poissonRatioStepper->setValue(m_settings.poisson_ratio);
    g_densityStepper->setValue(m_settings.density);
}

void AnalysisForm::m_readSettingsFromGUI() {
    m_settings.Nx = g_nxStepper->value();
    m_settings.Ny = g_nyStepper->value();
    m_settings.lumpedMass = g_lumpedMassCheck->isChecked();
    m_settings.quadrature = g_gaussQuadratureCheck->isChecked()
                                    ? GAUSS_QUADRATURE : UNIFORM_QUADRATURE;
    m_settings.quadraturePoints = g_quadraturePointsStepper->value();

    m_settings.numModes = g_numModesStepper->value();

    m_settings.young_modulus = g_youngModulusStepper->value();
    m_settings.poisson_ratio = g_poissonRatioStepper->value();
    m_settings.density       = g_densityStepper->value();
}

void AnalysisForm::modesUpdated(const MeshlessFEM_t *fem) {
    size_t numModes = (fem != NULL) ? fem->numModes() : 0;
    g_modeSelector->clear();
    g_modeSelector->addItem("Select Mode");

    if (numModes > 0) {
        QString label;
        for (size_t m = 0; m < numModes; ++m) {
            Scalar lambda = fem->eigenvalue(m);
            label.sprintf("Mode %i (lambda = %f)", (int) m, (float) lambda);
            g_modeSelector->addItem(label);
        }
        g_modeSelector->setEnabled(true);
    }
    else {
        g_modeSelector->setEnabled(false);
    }
}

void AnalysisForm::elementGridControlsChanged(int i) {
    m_readSettingsFromGUI();
    emit eqSettingsChanged(m_settings);
}

void AnalysisForm::materialControlsChanged(double v) {
    m_readSettingsFromGUI();
    emit materialSettingsChanged(m_settings);
}

void AnalysisForm::modalAnalysisControlsChanged(int i) {
    m_readSettingsFromGUI();
    emit modalAnalysisSettingsChanged(m_settings);
}
