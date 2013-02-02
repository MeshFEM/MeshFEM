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

AnalysisForm::AnalysisForm(AnalysisSettings &settings, QWidget *parent)
    : QWidget(parent), m_settings(settings)
{
    // Construct all widgets
    g_nxStepper = new QSpinBox();
    g_nyStepper = new QSpinBox();
    g_numModesStepper = new QSpinBox();
    g_numWeakRegionsStepper = new QSpinBox();
    g_lumpedMassCheck = new QCheckBox();
    g_gaussQuadratureCheck = new QCheckBox();
    g_quadraturePointsStepper = new QuadraturePointsSpinBox();

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
