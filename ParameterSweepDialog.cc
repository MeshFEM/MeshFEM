////////////////////////////////////////////////////////////////////////////////
// ParameterSweepDialog.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Dialog box for configuring parameter sweeps.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/17/2014 16:04:04
////////////////////////////////////////////////////////////////////////////////
#include "ParameterSweepDialog.hh"

#include <vector>
#include <string>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QGroupBox>
#include <QDialogButtonBox>
#include <QSplitter>
#include <QComboBox>

using namespace std;

ParameterSweepDialog::
ParameterSweepDialog(const string &modelName, const string &settingsName,
                     const std::vector<std::string> &settingNames,
                     const std::vector<std::string> &csgParameterNames,
                     QWidget *parent)
    : QDialog(parent), m_op(SWEEP_OP_RUN)
{
    setWindowTitle("Parameter Sweep Configuration");

    QVBoxLayout *layout = new QVBoxLayout();
    QFormLayout *formLayout = new QFormLayout();

    g_modelNameEdit = new QLineEdit(QString(modelName.c_str()));
    formLayout->addRow("Model name", g_modelNameEdit);
    g_settingsNameEdit = new QLineEdit(QString(settingsName.c_str()));
    formLayout->addRow("Settings name", g_settingsNameEdit);
    formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    layout->addLayout(formLayout);

    QGroupBox *settingsGroup = new QGroupBox("Parameters To Sweep");
    g_settingsList = new QListWidget();
    g_csgParameterList = new QListWidget();
    QVBoxLayout *settingsGroupLayout = new QVBoxLayout();
    QSplitter *splitter = new QSplitter(Qt::Vertical);
    splitter->addWidget(g_settingsList);
    splitter->addWidget(g_csgParameterList);
    settingsGroup->setLayout(settingsGroupLayout);
    settingsGroupLayout->setContentsMargins(5, 5, 5, 5);
    layout->addWidget(splitter);

    QGroupBox *rangesGroup = new QGroupBox("Parameter Ranges");
    g_rangesForm = new QFormLayout();
    g_rangesForm->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    rangesGroup->setLayout(g_rangesForm);
    g_sweepModeSelector = new QComboBox();
    g_sweepModeSelector->addItem("Zip");
    g_sweepModeSelector->addItem("Product");
    g_rangesForm->addRow("Sweep Mode", g_sweepModeSelector);
    layout->addWidget(rangesGroup);

    g_buttons = new QDialogButtonBox(QDialogButtonBox::Cancel);
    g_buttons->addButton("Run", QDialogButtonBox::AcceptRole);
    g_buttons->addButton("Save Inputs...", QDialogButtonBox::ActionRole);
    layout->addWidget(g_buttons);

    // Left top right bottom
    layout->setContentsMargins(0, 10, 0, 0);
    setLayout(layout);

    // Populate lists
    int id = 0;
    for (const string &name : settingNames) {
        QListWidgetItem *item = new QListWidgetItem(name.c_str());
        item->setData(Qt::UserRole, QVariant(id));
        item->setCheckState(Qt::Unchecked);
        g_settingsList->addItem(item);
        ++id;
    }

    m_numSettings = id;

    for (const string &name : csgParameterNames) {
        QListWidgetItem *item = new QListWidgetItem(name.c_str());
        item->setCheckState(Qt::Unchecked);
        item->setData(Qt::UserRole, QVariant(id));
        g_csgParameterList->addItem(item);
        ++id;
    }

    m_numParameters = id;

    QObject::connect(g_buttons, SIGNAL(accepted()), this, SLOT(accept()));
    QObject::connect(g_buttons, SIGNAL(rejected()), this, SLOT(reject()));
    QObject::connect(g_buttons, SIGNAL(clicked(QAbstractButton *)), this,
                     SLOT(buttonClicked(QAbstractButton *)));
    QObject::connect(g_settingsList, SIGNAL(itemChanged(QListWidgetItem *)),
                     this, SLOT(itemChanged(QListWidgetItem *)));
    QObject::connect(g_csgParameterList, SIGNAL(itemChanged(QListWidgetItem *)),
                     this, SLOT(itemChanged(QListWidgetItem *)));

}

string ParameterSweepDialog::modelNameFormat() const {
    return g_modelNameEdit->text().toStdString();
}

string ParameterSweepDialog::settingsNameFormat() const {
    return g_settingsNameEdit->text().toStdString();
}

int ParameterSweepDialog::sweepMode() const {
    return g_sweepModeSelector->currentIndex();
}

void ParameterSweepDialog::
selectedIdentifiersAndRanges(vector<string> &settingNames,
                             vector<string> &settingRanges,
                             vector<size_t> &csgParameterIndices,
                             vector<string> &csgParameterRanges) const
{
    settingNames.clear(), settingRanges.clear();
    csgParameterIndices.clear(), csgParameterRanges.clear();
    for (pair<int, QLineEdit *> e : m_rangeFields) {
        if ((size_t) e.first < m_numSettings) {
            string name = g_settingsList->item(e.first)->text().toStdString();
            settingNames.push_back(name);
            settingRanges.push_back(e.second->text().toStdString());
        }
        else {
            assert((size_t) e.first < m_numParameters);
            // CSG parameter ids are really the parameters' indices (in the CSG
            // tree's parameter collections) offset by the number of settings.
            csgParameterIndices.push_back(e.first - m_numSettings);
            csgParameterRanges.push_back(e.second->text().toStdString());
        }
    }
}

ParameterSweepDialog::~ParameterSweepDialog() { }

void ParameterSweepDialog::itemChanged(QListWidgetItem *item)
{
    int id = item->data(Qt::UserRole).toInt();
    if (item->checkState() == Qt::Checked) {
        QLineEdit *edit = new QLineEdit();

        // Insert the row in order by id...
        auto it = m_rangeFields.upper_bound(id);
        if (it != m_rangeFields.end()) {
            int row;
            QFormLayout::ItemRole role;
            g_rangesForm->getWidgetPosition(it->second, &row, &role);
            assert(row > 0); // sweep type is row 0, range fields should be > 0
            g_rangesForm->insertRow(row, item->text(), edit);
        }
        else {
            g_rangesForm->addRow(item->text(), edit);
        }

        m_rangeFields[id] = edit;
    }
    else {
        auto it = m_rangeFields.find(id);
        assert(it != m_rangeFields.end());
        QLineEdit *edit= it->second;
        QWidget *label = g_rangesForm->labelForField(edit);
        edit->deleteLater();
        label->deleteLater();
        m_rangeFields.erase(it);
    }
}

void ParameterSweepDialog::buttonClicked(QAbstractButton *button)
{
    if (g_buttons->buttonRole(button) == QDialogButtonBox::ActionRole) {
        m_op = SWEEP_OP_SAVE;
        accept();
    }
}
