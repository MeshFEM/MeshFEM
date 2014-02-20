////////////////////////////////////////////////////////////////////////////////
// BoundaryConditionDialog.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Dialog box for configuring boundary conditions.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/19/2014 17:31:43
////////////////////////////////////////////////////////////////////////////////
#include "BoundaryConditionDialog.hh"

#include <QLineEdit>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QDoubleValidator>
#include <QComboBox>
#include <QTabWidget>
#include <cassert>

BoundaryConditionDialog::BoundaryConditionDialog(BCs &c, int condition,
                                                 QWidget *parent)
        : QDialog(parent), m_conditions(c), m_selectedCondition(condition)
{
    setWindowTitle("Boundary Conditions");

    QVBoxLayout *layout = new QVBoxLayout();
    g_conditionSelector = new QComboBox();

    layout->addWidget(g_conditionSelector);

    QFormLayout *regionForm = new QFormLayout();
    regionForm->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    QGroupBox *regionGroup = new QGroupBox("Region Box");
    g_regionMinX = new QLineEdit();
    g_regionMaxX = new QLineEdit();
    g_regionMinY = new QLineEdit();
    g_regionMaxY = new QLineEdit();
    regionForm->addRow("Left X",   g_regionMinX);
    regionForm->addRow("Right X",  g_regionMaxX);
    regionForm->addRow("Bottom Y", g_regionMinY);
    regionForm->addRow("Top Y",    g_regionMaxY);
    QDoubleValidator *doubleValidator = new QDoubleValidator(this);
    g_regionMinX->setValidator(doubleValidator);
    g_regionMaxX->setValidator(doubleValidator);
    g_regionMinY->setValidator(doubleValidator);
    g_regionMaxY->setValidator(doubleValidator);
    regionGroup->setLayout(regionForm);
    layout->addWidget(regionGroup);

    QFormLayout *tractionForm = new QFormLayout();
    tractionForm->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    g_tractionX = new QLineEdit();
    g_tractionY = new QLineEdit();
    tractionForm->addRow("Traction X", g_tractionX);
    tractionForm->addRow("Traction Y", g_tractionY);
    g_tractionX->setValidator(doubleValidator);
    g_tractionY->setValidator(doubleValidator);

    QFormLayout *pressureForm = new QFormLayout();
    pressureForm->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    g_pressure = new QLineEdit();
    pressureForm->addRow("Pressure", g_pressure);
    g_pressure->setValidator(doubleValidator);

    g_tractionPressureTab = new QTabWidget();
    QWidget *tractionWidget = new QWidget();
    tractionWidget->setLayout(tractionForm);
    g_tractionPressureTab->addTab(tractionWidget, "Traction");
    QWidget *pressureWidget = new QWidget();
    pressureWidget->setLayout(pressureForm);
    g_tractionPressureTab->addTab(pressureWidget, "Pressure");

    layout->addWidget(g_tractionPressureTab);

    QDialogButtonBox *buttons = new QDialogButtonBox(QDialogButtonBox::Ok);
    layout->addWidget(buttons);

    QObject::connect(g_conditionSelector, SIGNAL(currentIndexChanged(int)),
                     this, SLOT(selectedConditionChanged(int)));

    for (size_t i = 0; i < m_conditions.numConditions(); ++i)
        g_conditionSelector->addItem("Condition " + QString::number(i + 1));

    if (condition < m_conditions.numConditions())
        g_conditionSelector->setCurrentIndex(condition);
    setLayout(layout);
}

void BoundaryConditionDialog::selectedConditionChanged(int newIdx) {
    if (m_selectedCondition >= 0) {
        // Save old values
    }
    else {
        const BCs::Condition &c = m_conditions.condition(newIdx);
        g_regionMinX->setText(QString::number(c.region.minCorner[0]));
        g_regionMaxX->setText(QString::number(c.region.maxCorner[0]));
        g_regionMinY->setText(QString::number(c.region.minCorner[1]));
        g_regionMaxY->setText(QString::number(c.region.maxCorner[1]));
        
        g_tractionPressureTab->setCurrentIndex(
                (c.type == BCs::CONDITION_TRACTION) ? 0 : 1);

        g_tractionX->setText(QString::number(c.value[0]));
        g_tractionY->setText(QString::number(c.value[1]));
        g_pressure->setText(QString::number(c.value[0]));
    }
}

BoundaryConditionDialog::~BoundaryConditionDialog() { }
