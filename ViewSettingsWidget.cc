#include <QtGui>

#include "ViewSettingsWidget.hh"
#include "ViewSettings.hh"
#include "colors.hh"

using namespace std;

ViewSettingsWidget::ViewSettingsWidget(ViewSettings &settings, QWidget *parent)
    : QWidget(parent), m_viewSettings(settings)
{
    QFormLayout *form = new QFormLayout();
    // Allow the slider to expand to the full width.
    form->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    g_showQuadraturePointsCheck = new QCheckBox();
    g_showGridOverResultsCheck = new QCheckBox();
    g_showStressesDuringDeformationCheck = new QCheckBox();
    g_fitVectorFieldsCheck = new QCheckBox();
    g_autofitMagnitudeSlider = new QSlider(Qt::Horizontal);
    g_vfieldStyleSelector = new QComboBox();
    g_showColorbarCheck = new QCheckBox();
    g_colormapSelector = new QComboBox();
    g_colormapMinStepper = new QDoubleSpinBox();
    g_colormapMaxStepper = new QDoubleSpinBox();
    g_colormapAutoRangeCheck = new QCheckBox("Auto");

    g_autofitMagnitudeSlider->setSizePolicy(QSizePolicy::MinimumExpanding,
                                            QSizePolicy::Fixed);

    form->addRow("Show Quadrature Points", g_showQuadraturePointsCheck);
    form->addRow("Show Grid Over Results", g_showGridOverResultsCheck);
    form->addRow("Show Stresses During Deformation", g_showStressesDuringDeformationCheck);
    form->addRow("Auto-Fit Vector Fields", g_fitVectorFieldsCheck);
    form->addRow("Auto-Fit Magnitude", g_autofitMagnitudeSlider);
    form->addRow("Vector Field Style", g_vfieldStyleSelector);
    form->addRow("Show colorbar", g_showColorbarCheck);
    form->addRow("Colormap", g_colormapSelector);

    g_autofitMagnitudeSlider->setMinimum(1);
    g_autofitMagnitudeSlider->setMaximum(1000);

    g_vfieldStyleSelector->addItem("Deform");
    g_vfieldStyleSelector->addItem("Vibrate");
    g_vfieldStyleSelector->addItem("Arrows");

    g_colormapSelector->addItem("Jet");
    g_colormapSelector->addItem("Combined Weakness");
    g_colormapSelector->addItem("Fire Print");

    QHBoxLayout *colormapRangeLayout = new QHBoxLayout();
    QLabel *toLabel = new QLabel("to");
    colormapRangeLayout->addWidget(g_colormapAutoRangeCheck);
    colormapRangeLayout->addWidget(g_colormapMinStepper);
    colormapRangeLayout->addWidget(toLabel);
    colormapRangeLayout->addWidget(g_colormapMaxStepper);
    form->addRow("Colormap range", colormapRangeLayout);

    setLayout(form);
    layout()->setSizeConstraint(QLayout::SetFixedSize);

    m_setGUIFromSettings();

    QObject::connect(g_showQuadraturePointsCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_showGridOverResultsCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_showStressesDuringDeformationCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_fitVectorFieldsCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_autofitMagnitudeSlider, SIGNAL(valueChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_vfieldStyleSelector, SIGNAL(currentIndexChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_showColorbarCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_colormapSelector, SIGNAL(currentIndexChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_colormapAutoRangeCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_colormapMinStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(m_guiDoubleChanged(double)));
    QObject::connect(g_colormapMaxStepper, SIGNAL(valueChanged(double)),
                     this, SLOT(m_guiDoubleChanged(double)));
}

void ViewSettingsWidget::m_setGUIFromSettings() {
    g_showQuadraturePointsCheck->setChecked(m_viewSettings.showQuadraturePoints);
    g_showGridOverResultsCheck->setChecked(m_viewSettings.showGridOverResults);
    g_showStressesDuringDeformationCheck->setChecked(m_viewSettings.showStressesDuringDeformation);

    g_fitVectorFieldsCheck->setChecked(m_viewSettings.autofitVectorField);
    g_autofitMagnitudeSlider->setValue(m_viewSettings.autofitMagnitude *
                                       g_autofitMagnitudeSlider->maximum());

    // Note: assumes VFieldDisplayStyle enum index matches combo box index
    g_vfieldStyleSelector->setCurrentIndex(m_viewSettings.vfDisplayStyle);

    g_showColorbarCheck->setChecked(m_viewSettings.showColorbar);
    // Note: assumes CMapName enum index matches combo box index
    g_colormapSelector->setCurrentIndex(m_viewSettings.colormap);
    g_colormapAutoRangeCheck->setChecked(m_viewSettings.colormapRangeAuto);

    g_colormapMinStepper->setEnabled(!m_viewSettings.colormapRangeAuto);
    g_colormapMaxStepper->setEnabled(!m_viewSettings.colormapRangeAuto);
    g_colormapMinStepper->setValue(m_viewSettings.colormapRangeMin);
    g_colormapMaxStepper->setValue(m_viewSettings.colormapRangeMax);
    
}

void ViewSettingsWidget::m_readSettingsFromGUI() {
    m_viewSettings.showQuadraturePoints = g_showQuadraturePointsCheck->isChecked();
    m_viewSettings.showGridOverResults = g_showGridOverResultsCheck->isChecked();
    m_viewSettings.showStressesDuringDeformation = g_showStressesDuringDeformationCheck->isChecked();

    m_viewSettings.autofitVectorField = g_fitVectorFieldsCheck->isChecked();
    m_viewSettings.autofitMagnitude = g_autofitMagnitudeSlider->value() /
        ((Scalar) g_autofitMagnitudeSlider->maximum());

    // Note: assumes VFieldDisplayStyle enum index matches combo box index
    m_viewSettings.vfDisplayStyle = (ViewSettings::VFieldDisplayStyle)
                    g_vfieldStyleSelector->currentIndex();

    m_viewSettings.showColorbar = g_showColorbarCheck->isChecked();
    // Note: assumes CMapName enum index matches combo box index
    m_viewSettings.colormap = (CMapName) g_colormapSelector->currentIndex();
    m_viewSettings.colormapRangeAuto = g_colormapAutoRangeCheck->isChecked();
    m_viewSettings.colormapRangeMin = g_colormapMinStepper->value();
    m_viewSettings.colormapRangeMax = g_colormapMaxStepper->value();
}

void ViewSettingsWidget::m_guiIntChanged(int) {
    m_readSettingsFromGUI();
    // Some settings affect enabled properties of the GUI...
    m_setGUIFromSettings();

    emit viewSettingsUpdated();
}

void ViewSettingsWidget::m_guiDoubleChanged(double) {
    m_readSettingsFromGUI();
    // Some settings affect enabled properties of the GUI...
    m_setGUIFromSettings();

    emit viewSettingsUpdated();
}
