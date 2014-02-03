////////////////////////////////////////////////////////////////////////////////
// ViewSettingsWidget.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        GUI for changing the view settings.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/01/2013 14:16:15
////////////////////////////////////////////////////////////////////////////////
#ifndef VIEW_SETTINGS_WIDGET_HH
#define VIEW_SETTINGS_WIDGET_HH

#include <QObject>
#include <QWidget>

class QComboBox;
class QCheckBox;
class QSlider;
class QDoubleSpinBox;
struct ViewSettings;

class ViewSettingsWidget : public QWidget
{
    Q_OBJECT

private slots:
    void m_guiIntChanged(int);
    void m_guiDoubleChanged(double);

signals:
    void viewSettingsUpdated();

public:
    ViewSettingsWidget(ViewSettings &settings, QWidget *parent = NULL);

private:
    ViewSettings &m_viewSettings;

    QCheckBox *g_signedDistanceCheck,
              *g_showQuadraturePointsCheck,
              *g_hilightCutCellsCheck,
              *g_showGridOverResultsCheck,
              *g_showStressesDuringDeformationCheck,
              *g_fitVectorFieldsCheck,
              *g_showColorbarCheck;
    QSlider *g_autofitMagnitudeSlider;
    QComboBox *g_vfieldStyleSelector, *g_colormapSelector;
    QDoubleSpinBox *g_colormapMinStepper, *g_colormapMaxStepper;
    QCheckBox *g_colormapAutoRangeCheck;

    void m_setGUIFromSettings();
    void m_readSettingsFromGUI();
};

#endif // VIEW_SETTINGS_WIDGET_HH
