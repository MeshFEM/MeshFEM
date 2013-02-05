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

class QSpinBox;
class QDoubleSpinBox;
class QuadraturePointsSpinBox;
class QCheckBox;
class QPushButton;

class AnalysisForm : public QWidget
{
    Q_OBJECT
public:
    AnalysisForm(AnalysisSettings &settings, QWidget *parent = NULL);

private slots:
    void elementGridControlsChanged(int i);

signals:
    void elementGridChanged(int Nx, int Ny, int numQuadraturePoints,
                            bool gaussQuadrature);

private:
    AnalysisSettings &m_settings;

    QSpinBox *g_nxStepper, *g_nyStepper, *g_numModesStepper,
             *g_numWeakRegionsStepper;
    QCheckBox *g_lumpedMassCheck, *g_gaussQuadratureCheck;
    QuadraturePointsSpinBox *g_quadraturePointsStepper;
};
#endif // ANALYSIS_FORM_HH
