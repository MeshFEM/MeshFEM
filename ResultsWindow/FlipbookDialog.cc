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
#include <QtGui>
#include <vector>
#include <string>

using namespace std;

FlipbookDialog::FlipbookDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Flipbook Configuration");
    QVBoxLayout *layout = new QVBoxLayout();
    QFormLayout *formLayout = new QFormLayout();

    g_titleEdit = new QLineEdit();
    formLayout->addRow("Title", g_titleEdit);
    formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    layout->addLayout(formLayout);

    g_settingList = new QListWidget();
    QGroupBox *settingsGroup = new QGroupBox("Settings Values to Write");
    QVBoxLayout *settingsGroupLayout = new QVBoxLayout();
    settingsGroupLayout->setContentsMargins(5, 5, 5, 5);
    settingsGroupLayout->addWidget(g_settingList);
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

    generateSettingItems();
}

void FlipbookDialog::generateSettingItems()
{
    AnalysisSettings dummySettings;
    vector<string> names = dummySettings.getNames();

    for (string &name : names) {
        QListWidgetItem *item = new QListWidgetItem(name.c_str());
        item->setCheckState(Qt::Unchecked);
        g_settingList->addItem(item);

    }
}

string FlipbookDialog::title() const {
    return g_titleEdit->text().toStdString();
}

vector<string> FlipbookDialog::selectedSettingNames() const {
    vector<string> names;

    int numItems = g_settingList->count();
    for (size_t i = 0; i < numItems; ++i) {
        QListWidgetItem *item = g_settingList->item(i);
        if (item->checkState() == Qt::Checked) {
            names.push_back(item->text().toStdString());
        }
    }

    return names;
}

FlipbookDialog::~FlipbookDialog()
{

}
