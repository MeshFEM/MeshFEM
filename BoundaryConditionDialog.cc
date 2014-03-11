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
#include <QHBoxLayout>
#include <QComboBox>
#include <QToolButton>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QDoubleValidator>
#include <QTabWidget>
#include <cassert>

BoundaryConditionDialog::BoundaryConditionDialog(BCs &c, int condition,
                                                 QWidget *parent)
        : QDialog(parent), m_conditions(c), m_selectedCondition(-1)
{
    setWindowTitle("Boundary Conditions");

    QVBoxLayout *layout = new QVBoxLayout();

    QHBoxLayout *managerLayout = new QHBoxLayout();
    g_conditionSelector = new QComboBox();
    QToolButton *addButton = new QToolButton();
    QToolButton *removeButton = new QToolButton();
    addButton->setText("+");
    removeButton->setText("-");
    managerLayout->addWidget(g_conditionSelector);
    managerLayout->addWidget(addButton);
    managerLayout->addWidget(removeButton);

    layout->addLayout(managerLayout);

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

    g_conditionTypeTab = new QTabWidget();
    QWidget *tractionWidget = new QWidget();
    tractionWidget->setLayout(tractionForm);
    g_conditionTypeTab->addTab(tractionWidget, "Traction");
    QWidget *pressureWidget = new QWidget();
    pressureWidget->setLayout(pressureForm);
    g_conditionTypeTab->addTab(pressureWidget, "Pressure");
    QWidget *dirichletWidget = new QWidget();
    QLabel *dirichletInfo = new QLabel("(Nothing to configure for Dirichlet.)",
                                       dirichletWidget);
    g_conditionTypeTab->addTab(dirichletWidget, "Dirichlet");

    layout->addWidget(g_conditionTypeTab);

    QDialogButtonBox *buttons = new QDialogButtonBox(QDialogButtonBox::Ok);
    layout->addWidget(buttons);

    QObject::connect(g_conditionSelector, SIGNAL(currentIndexChanged(int)),
                     this, SLOT(selectedConditionChanged(int)));
    QObject::connect(g_conditionSelector, SIGNAL(currentIndexChanged(int)),
                     this, SLOT(selectedConditionChanged(int)));
    QObject::connect(buttons, SIGNAL(accepted()), this, SLOT(okClicked()));

    for (size_t i = 0; i < m_conditions.numConditions(); ++i)
        g_conditionSelector->addItem("Condition " + QString::number(i + 1));

    // Select a valid condition
    if (condition > (int) m_conditions.numConditions())
        condition = -1;

    if ((condition < 0) && (m_conditions.numConditions() > 0))
        condition = 0;

    if (condition >= 0)
        selectCondition(condition);

    setLayout(layout);
}

void BoundaryConditionDialog::saveCondition() {
    if (m_selectedCondition >= 0) {
        BCs::Condition &c = m_conditions.condition(m_selectedCondition);
        c.region.minCorner[0] = g_regionMinX->text().toDouble();
        c.region.maxCorner[0] = g_regionMaxX->text().toDouble();
        c.region.minCorner[1] = g_regionMinY->text().toDouble();
        c.region.maxCorner[1] = g_regionMaxY->text().toDouble();

        switch (g_conditionTypeTab->currentIndex()) {
            case 0:
                c.setTraction(Vector(g_tractionX->text().toDouble(),
                                     g_tractionY->text().toDouble()));
                break;
            case 1:
                c.setPressure(g_pressure->text().toDouble());
                break;
            case 2:
                c.setDirichlet();
                break;
            default:
                assert(false);
        }
    }
}

void BoundaryConditionDialog::selectCondition(int idx) {
    saveCondition();

    assert((size_t) idx < m_conditions.numConditions());

    const BCs::Condition &c = m_conditions.condition(idx);
    g_regionMinX->setText(QString::number(c.region.minCorner[0]));
    g_regionMaxX->setText(QString::number(c.region.maxCorner[0]));
    g_regionMinY->setText(QString::number(c.region.minCorner[1]));
    g_regionMaxY->setText(QString::number(c.region.maxCorner[1]));
    
    g_conditionTypeTab->setCurrentIndex(
            (c.type == BCs::CONDITION_TRACTION) ? 0 :
            (c.type == BCs::CONDITION_PRESSURE) ? 1 : 2);

    g_tractionX->setText(QString::number(c.value[0]));
    g_tractionY->setText(QString::number(c.value[1]));
    g_pressure->setText(QString::number(c.value[0]));

    m_selectedCondition = idx;
    g_conditionSelector->setCurrentIndex(idx);
}

void BoundaryConditionDialog::okClicked() {
    saveCondition();
    accept();
}

void BoundaryConditionDialog::selectedConditionChanged(int newIdx) {
    if (newIdx != m_selectedCondition)
        selectCondition(newIdx);
}

BoundaryConditionDialog::~BoundaryConditionDialog() { }
