////////////////////////////////////////////////////////////////////////////////
// AnalysisForm.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        The GUI holding all analysis settings.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/02/2013 00:42:57
////////////////////////////////////////////////////////////////////////////////
#ifndef ANALYSIS_FORM_HH
#define ANALYSIS_FORM_HH

#include <QWidget>
#include "AnalysisSettings.hh"

class CSGWindowController;
class QSpinBox;
class QDoubleSpinBox;
class QuadraturePointsSpinBox;
class QCheckBox;
class QPushButton;
class QComboBox;

class AnalysisForm : public QWidget
{
    Q_OBJECT
public:
    AnalysisForm(AnalysisSettings &settings,
                 CSGWindowController *controller, QWidget *parent = NULL);

public slots:
    void modesUpdated(const MeshlessFEM_t *fem);

private slots:
    void elementGridControlsChanged(int);
    void elementGridControlsChanged(double);
    void boundaryPointControlsChanged(double);
    void modalAnalysisControlsChanged(int);
    void matrixControlsChanged(int);
    void materialControlsChanged(double);

signals:
    void eqSettingsChanged(const AnalysisSettings &settings);
    void bpSettingsChanged(const AnalysisSettings &settings);
    void matrixOrMaterialSettingsChanged(const AnalysisSettings &settings);
    void modalAnalysisSettingsChanged(const AnalysisSettings &settings);

private:
    AnalysisSettings &m_settings;

    // Elements and quadrature settings
    QSpinBox *g_nxStepper, *g_nyStepper;
    QuadraturePointsSpinBox *g_quadraturePointsStepper;
    QCheckBox *g_gaussQuadratureCheck;
    QDoubleSpinBox *g_cellOverlapStepper;

    // Matrix/material settings
    QComboBox *g_massMatrixSelector;
    QDoubleSpinBox *g_youngModulusStepper;
    QDoubleSpinBox *g_poissonRatioStepper;
    QDoubleSpinBox *g_densityStepper;

    // Modal analysis settings
    QSpinBox *g_numModesStepper;
    QPushButton *g_modalAnalysisButton;
    QPushButton *g_dumpModalDataButton;
    QComboBox *g_modeSelector;

    // Simulation settings
    QDoubleSpinBox *g_boundaryPointStepper;
    QPushButton *g_configureSimulationButton;
    QPushButton *g_runSimulationButton;
    QDoubleSpinBox *g_pressurePaintValueStepper;

    // Weakness analysis settings
    QSpinBox *g_numWeakRegionsStepper;
    QPushButton *g_weaknessAnalysisButton;

    void m_setGUIFromSettings();
    void m_readSettingsFromGUI();
};
#endif // ANALYSIS_FORM_HH
