////////////////////////////////////////////////////////////////////////////////
// ParameterSweepDialog.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Dialog box for configuring parameter sweeps.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/17/2014 16:04:04
////////////////////////////////////////////////////////////////////////////////
#ifndef PARAMETERSWEEPDIALOG_HH
#define PARAMETERSWEEPDIALOG_HH

#include <QObject>
#include <QDialog>
#include <vector>
#include <map>
#include <string>

#include "ParameterSweep.hh"

class QWidget;
class QLineEdit;
class QListWidget;
class QFormLayout;
class QComboBox;

class ParameterSweepDialog : public QDialog
{
    Q_OBJECT

public:
    ParameterSweepDialog(const std::string &modelName,
                         const std::string &settingsName,
                         const std::vector<std::string> &settingNames, 
                         const std::vector<std::string> &csgParameterNames,
                         QWidget *parent = 0);

    std::string modelNameFormat() const;
    std::string settingsNameFormat() const;
    int sweepMode() const;

    void selectedIdentifiersAndRanges(std::vector<std::string> &settingNames,
                             std::vector<std::string> &settingRanges,
                             std::vector<size_t> &csgParameterIndices,
                             std::vector<std::string> &csgParameterRanges) const;

    ~ParameterSweepDialog();

private slots:
    void itemChanged(QListWidgetItem *item);

private:
    void generateSettingItems();

    QLineEdit   *g_modelNameEdit, *g_settingsNameEdit;
    QListWidget *g_settingsList, *g_csgParameterList;
    QFormLayout *g_rangesForm;
    QComboBox   *g_sweepModeSelector;

    size_t m_numSettings, m_numParameters;

    // Table of range fields keyed off the parameter id.
    std::map<int, QLineEdit *> m_rangeFields;
};

#endif /* end of include guard: PARAMETERSWEEPDIALOG_HH */

