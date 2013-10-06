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
#include <string>
#include "AnalysisSettings.hh"

class CSGWindowController;
template<typename Real>
class SolverLibrary;
class QSpinBox;
class QDoubleSpinBox;
class QuadraturePointsSpinBox;
class QCheckBox;
class QPushButton;
class QComboBox;
class QLineEdit;

class AnalysisForm : public QWidget
{
    Q_OBJECT
public:
    AnalysisForm(AnalysisSettings &settings, CSGWindowController *controller,
                 SolverLibrary<Scalar> &solvers, QWidget *parent = NULL);

public slots:
    void reloadSettings();
    void nameConflictsUpdated(bool modelConflict, bool settingsConflict);
    void namesUpdated(const std::string &modelName,
                      const std::string &settingsName);

private slots:
    void solverControlsChanged(int);
    void elementGridControlsChanged(int);
    void elementGridControlsChanged(double);
    void boundaryPointControlsChanged(double);
    void boundaryPointControlsChanged(int);
    void modalAnalysisControlsChanged(int);
    void matrixControlsChanged(int);
    void materialControlsChanged(double);
    void weaknessAnalysisControlsChanged(int);
    void weaknessAnalysisControlsChanged(double);
    void ttestControlsChanged(int);
    void ttestControlsChanged(double);
    void ttestButtonClicked();
    void fttestButtonClicked();
    void frtestButtonClicked();
    void reftestButtonClicked();

signals:
    void settingsChanged();
    void eqSettingsChanged(const AnalysisSettings &settings);
    void bpSettingsChanged(const AnalysisSettings &settings);
    void matrixOrMaterialSettingsChanged(const AnalysisSettings &settings);
    void modalAnalysisSettingsChanged(const AnalysisSettings &settings);
    void weaknessAnalysisSettingsChanged(const AnalysisSettings &settings);
    void runTranslationTest(const AnalysisSettings &settings);
    void runForceTranslationTest(const AnalysisSettings &settings);
    void runFunctionRadiusTest(const AnalysisSettings &settings);
    void runRefinementTest();

private:
    AnalysisSettings &m_settings;
    SolverLibrary<Scalar> &m_solvers;

    // Names section
    QLineEdit *g_modelNameEdit;
    QLineEdit *g_settingsNameEdit;

    // Solver selection
    QComboBox *g_solverSelector;

    // Elements and quadrature settings
    QSpinBox *g_nxStepper, *g_nyStepper;
    QSpinBox *g_borderWidthStepper;
    QuadraturePointsSpinBox *g_quadraturePointsStepper;
    QCheckBox *g_gaussQuadratureCheck;
    QDoubleSpinBox *g_cellOverlapStepper;

    QCheckBox *g_exactFullElementsCheck, *g_antialiasedElementsCheck;

    // Matrix/material settings
    QComboBox *g_massMatrixSelector;
    QDoubleSpinBox *g_youngModulusStepper;
    QDoubleSpinBox *g_poissonRatioStepper;
    QDoubleSpinBox *g_densityStepper;

    // Modal analysis settings
    QSpinBox *g_numModesStepper;
    QCheckBox *g_laplacianModesCheck, *g_consistentSignsCheck;
    QPushButton *g_modalAnalysisButton;

    // Simulation settings
    QCheckBox *g_useMarchingSquaresCheck, *g_blurForcesCheck;
    QSpinBox *g_newtonIterationsStepper;
    QDoubleSpinBox *g_boundaryPointStepper;
    QDoubleSpinBox *g_kernelRadiusStepper;
    QPushButton *g_configureSimulationButton;
    QPushButton *g_savePressureButton;
    QPushButton *g_loadPressureButton;
    QPushButton *g_runSimulationButton;
    QDoubleSpinBox *g_pressurePaintValueStepper;

    // Weakness analysis settings
    QSpinBox *g_numWeakRegionsStepper;
    QDoubleSpinBox *g_weaknessCutoffStepper;
    QPushButton *g_weakRegionExtractionButton;
    QDoubleSpinBox *g_pressureBoundStepper, *g_forceBoundStepper;
    QCheckBox *g_abstraceCheck, *g_plusMinusObjectiveCheck;

    QPushButton *g_weaknessAnalysisButton;
    QPushButton *g_optimizeShapeButton;
    QDoubleSpinBox *g_xTranslationStepper, *g_yTranslationStepper;
    QCheckBox *g_translationFixedCheckbox;
    QPushButton *g_translationTestButton;
    QPushButton *g_forceTranslationTestButton;
    QPushButton *g_functionRadiusTestButton;
    QPushButton *g_refinementTestButton;

    void m_setGUIFromSettings();
    void m_readSettingsFromGUI();
    bool m_settingGUIFromSettings;
};
#endif // ANALYSIS_FORM_HH
