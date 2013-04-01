#include <QtGui>

#include "ViewSettingsWidget.hh"
#include "ViewSettings.hh"
#include "colors.hh"

ViewSettingsWidget::ViewSettingsWidget(ViewSettings &settings, QWidget *parent)
    : QWidget(parent), m_viewSettings(settings)
{
    QFormLayout *form = new QFormLayout();
    g_showGridDuringDeformationCheck = new QCheckBox();
    g_showStressesDuringDeformationCheck = new QCheckBox();
    g_showColorbar = new QCheckBox();
    g_colormapSelector = new QComboBox();
    g_colormapMinStepper = new QDoubleSpinBox();
    g_colormapMaxStepper = new QDoubleSpinBox();
    g_colormapAutoRangeCheck = new QCheckBox("Auto");

    form->addRow("Show Grid During Deformation", g_showGridDuringDeformationCheck);
    form->addRow("Show Stresses During Deformation", g_showStressesDuringDeformationCheck);
    form->addRow("Show colorbar", g_showColorbar);
    form->addRow("Colormap", g_colormapSelector);

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
    m_setGUIFromSettings();

    QObject::connect(g_showGridDuringDeformationCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_showStressesDuringDeformationCheck, SIGNAL(stateChanged(int)),
                     this, SLOT(m_guiIntChanged(int)));
    QObject::connect(g_showColorbar, SIGNAL(stateChanged(int)),
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
    g_showGridDuringDeformationCheck->setChecked(m_viewSettings.showGridDuringDeformation);
    g_showStressesDuringDeformationCheck->setChecked(m_viewSettings.showStressesDuringDeformation);
    g_showColorbar->setChecked(m_viewSettings.showColorbar);
    // Note: assumes CMapName enum index matches combo box index
    g_colormapSelector->setCurrentIndex(m_viewSettings.colormap);
    g_colormapAutoRangeCheck->setChecked(m_viewSettings.colormapRangeAuto);

    g_colormapMinStepper->setEnabled(!m_viewSettings.colormapRangeAuto);
    g_colormapMaxStepper->setEnabled(!m_viewSettings.colormapRangeAuto);
    g_colormapMinStepper->setValue(m_viewSettings.colormapRangeMin);
    g_colormapMaxStepper->setValue(m_viewSettings.colormapRangeMax);
}

void ViewSettingsWidget::m_readSettingsFromGUI() {
    m_viewSettings.showGridDuringDeformation = g_showGridDuringDeformationCheck->isChecked();
    m_viewSettings.showStressesDuringDeformation = g_showStressesDuringDeformationCheck->isChecked();
    m_viewSettings.showColorbar = g_showColorbar->isChecked();
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
