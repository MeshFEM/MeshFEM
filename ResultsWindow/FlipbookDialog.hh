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

#include <QDialog>
class QWidget;
class QLineEdit;
class QListWidget;

class FlipbookDialog : public QDialog
{
    Q_OBJECT

public:
    FlipbookDialog(QWidget *parent = 0);

    std::string title() const;
    std::vector<std::string> selectedSettingNames() const;

    ~FlipbookDialog();

private:
    void generateSettingItems();

    QLineEdit   *g_titleEdit;
    QListWidget *g_settingList;
};

#endif // FLIPBOOK_DIALOG_HH
