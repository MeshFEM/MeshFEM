////////////////////////////////////////////////////////////////////////////////
// FlipbookDialog.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Dialog box for configuring flipbook output.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  09/09/2013 01:34:45
////////////////////////////////////////////////////////////////////////////////
#ifndef FLIPBOOK_DIALOG_HH
#define FLIPBOOK_DIALOG_HH

#include <QObject>
#include <QDialog>
#include <vector>

#include "GlobalTypes.hh"
#include "MeshlessFEM2D.hh"
#include "ResultsCollector.hh"


class QWidget;
class QLineEdit;
class QListWidget;

class FlipbookDialog : public QDialog
{
    Q_OBJECT

public:
    typedef ResultsCollector_t::ModelParameterID MPID;
    FlipbookDialog(const std::vector<MPID> &mparams, QWidget *parent = 0);

    std::string title() const;
    std::vector<std::string> selectedSettingNames() const;
    std::vector<MPID> selectedCSGParameterIDs() const;

    ~FlipbookDialog();

private:
    std::vector<MPID> m_modelParams;
    QLineEdit   *g_titleEdit;
    QListWidget *g_settingList, *g_csgParamList;
};

#endif // FLIPBOOK_DIALOG_HH
