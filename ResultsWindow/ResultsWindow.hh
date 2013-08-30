////////////////////////////////////////////////////////////////////////////////
// ResultsWindow.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements the results collection viewer window.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/29/2013 19:28:49
////////////////////////////////////////////////////////////////////////////////
#ifndef RESULTS_WINDOW_HH
#define RESULTS_WINDOW_HH

#include <QWidget>
#include <map>
#include <memory>
#include "GlobalTypes.hh"
#include "ResultsWindowController.hh"

class ResultTreeView;
class QListWidget;
class QPushButton;
class QCheckBox;
class QLineEdit;

class ResultsWindow : public QWidget
{
    Q_OBJECT

public:
    ResultsWindow(ResultsCollector_t &rc, QWidget *parent = NULL);
    ResultsWindowController * controller() { return m_controller.get(); }

    ~ResultsWindow();

private:
    std::shared_ptr<ResultsWindowController> m_controller;
    ResultsCollector_t &m_resultsCollection;
    ResultTreeView *g_treeView;
    QListWidget *g_filterView;
    QPushButton *g_deleteButton, *g_mshButton, *g_rawButton,
                *g_flipbookButton, *g_searchButton;
    QCheckBox *g_modelSettingsGrouping;
    QLineEdit *g_searchField;

    friend class ResultsWindowController;
};

#endif // RESULTS_WINDOW_HH
