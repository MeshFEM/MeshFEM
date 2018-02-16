////////////////////////////////////////////////////////////////////////////////
// FlipbookDialog.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Dialog box for configuring flipbook output.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  09/09/2013 01:34:45
////////////////////////////////////////////////////////////////////////////////
#include "FlipbookDialog.hh"
#include "AnalysisSettings.hh"
#include <vector>
#include <string>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QGroupBox>
#include <QDialogButtonBox>

using namespace std;

FlipbookDialog::FlipbookDialog(const std::vector<MPID> &mparams,
                               QWidget *parent)
    : QDialog(parent), m_modelParams(mparams)
{
    setWindowTitle("Flipbook Configuration");
    QVBoxLayout *layout = new QVBoxLayout();
    QFormLayout *formLayout = new QFormLayout();

    g_titleEdit = new QLineEdit();
    formLayout->addRow("Title", g_titleEdit);
    formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    layout->addLayout(formLayout);

    QGroupBox *settingsGroup = new QGroupBox("Setting/Parameter Values to Write");
    g_settingList = new QListWidget();
    g_csgParamList = new QListWidget();
    QVBoxLayout *settingsGroupLayout = new QVBoxLayout();
    QSplitter *splitter = new QSplitter(Qt::Vertical);
    splitter->addWidget(g_settingList);
    splitter->addWidget(g_csgParamList);
    settingsGroupLayout->setContentsMargins(5, 5, 5, 5);
    settingsGroupLayout->addWidget(splitter);
    settingsGroup->setLayout(settingsGroupLayout);

    layout->addWidget(settingsGroup);

    QDialogButtonBox *buttons = new QDialogButtonBox(QDialogButtonBox::Save |
            QDialogButtonBox::Cancel);

    layout->addWidget(buttons);

    // Left top right bottom
    layout->setContentsMargins(0, 10, 0, 0);
    setLayout(layout);

    QObject::connect(buttons, SIGNAL(accepted()), this, SLOT(accept()));
    QObject::connect(buttons, SIGNAL(rejected()), this, SLOT(reject()));

    AnalysisSettings dummySettings;
    vector<string> names = dummySettings.getNames();

    for (string &name : names) {
        QListWidgetItem *item = new QListWidgetItem(name.c_str());
        item->setCheckState(Qt::Unchecked);
        g_settingList->addItem(item);
    }

    for (MPID &param : m_modelParams) {
        QListWidgetItem *item = new QListWidgetItem(param.second.c_str());
        item->setCheckState(Qt::Unchecked);
        g_csgParamList->addItem(item);
    }
}

string FlipbookDialog::title() const {
    return g_titleEdit->text().toStdString();
}

vector<string> FlipbookDialog::selectedSettingNames() const {
    vector<string> names;

    size_t numItems = g_settingList->count();
    for (size_t i = 0; i < numItems; ++i) {
        QListWidgetItem *item = g_settingList->item(i);
        if (item->checkState() == Qt::Checked) {
            names.push_back(item->text().toStdString());
        }
    }

    return names;
}

vector<FlipbookDialog::MPID> FlipbookDialog::selectedCSGParameterIDs() const {
    vector<MPID> ids;

    size_t numItems = g_csgParamList->count();
    for (size_t i = 0; i < numItems; ++i) {
        QListWidgetItem *item = g_csgParamList->item(i);
        if (item->checkState() == Qt::Checked) {
            assert(i < m_modelParams.size());
            ids.push_back(m_modelParams[i]);
        }
    }

    return ids;
}

FlipbookDialog::~FlipbookDialog()
{

}
