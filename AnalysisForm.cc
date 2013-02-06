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

AnalysisForm::AnalysisForm(AnalysisSettings &settings,
                           CSGWindowController *controller, QWidget *parent)
    : QWidget(parent), m_settings(settings)
{
    // Construct all widgets
    g_nxStepper = new QSpinBox();
    g_nyStepper = new QSpinBox();
    g_nxStepper->setMinimum(1);
    g_nyStepper->setMinimum(1);
    g_numModesStepper = new QSpinBox();
    g_numWeakRegionsStepper = new QSpinBox();
    g_lumpedMassCheck = new QCheckBox();
    g_gaussQuadratureCheck = new QCheckBox();
    g_quadraturePointsStepper = new QuadraturePointsSpinBox();
    g_modalAnalysisButton = new QPushButton("Modal Analysis");

    QGroupBox *elementsQuadratureGroup = new QGroupBox("Elements and Quadrature");
    QGroupBox *modalAnalysisGroup = new QGroupBox("Modal Analysis");
    QGroupBox *weaknessAnalysisGroup = new QGroupBox("Weakness Analysis");
    QGroupBox *simulationGroup = new QGroupBox("Simulation");

    // Elements and Quadrature
    QFormLayout *eqForm = new QFormLayout();
    eqForm->addRow("Nx", g_nxStepper);
    eqForm->addRow("Ny", g_nyStepper);
    eqForm->addRow("Lumped Mass", g_lumpedMassCheck);
    eqForm->addRow("Gauss Quadrature", g_gaussQuadratureCheck);
    eqForm->addRow("Quadrature Points", g_quadraturePointsStepper);
    elementsQuadratureGroup->setLayout(eqForm);

    // Modal Analysis
    QFormLayout *modalForm = new QFormLayout();
    modalForm->addRow(g_modalAnalysisButton);
    modalAnalysisGroup->setLayout(modalForm);

    // Connections
    QObject::connect(g_nxStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_nyStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_quadraturePointsStepper, SIGNAL(valueChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    QObject::connect(g_gaussQuadratureCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(elementGridControlsChanged(int)));
    assert(controller);
    QObject::connect(g_modalAnalysisButton, SIGNAL(clicked()),
                     controller, SLOT(runModalAnalysis()));

    // Layout all the groups
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(elementsQuadratureGroup);
    layout->addWidget(modalAnalysisGroup);
    layout->addWidget(weaknessAnalysisGroup);
    layout->addWidget(simulationGroup);
    layout->addStretch();

    // QScrollArea *scrollArea = new QScrollArea(this);
    // scrollArea->setLayout(layout);
    setLayout(layout);
}

void AnalysisForm::elementGridControlsChanged(int i) {
    int Nx = g_nxStepper->value();
    int Ny = g_nyStepper->value();
    int quadraturePoints = g_quadraturePointsStepper->value();
    bool gaussNodes = g_gaussQuadratureCheck->isChecked();
    emit elementGridChanged(Nx, Ny, quadraturePoints, gaussNodes);
}
