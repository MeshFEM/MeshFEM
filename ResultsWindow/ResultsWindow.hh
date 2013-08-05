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
#include "GlobalTypes.hh"
#include "ResultsWindowController.hh"

class ResultTreeView;
class QPushButton;

class ResultsWindow : public QWidget
{
    Q_OBJECT

public:
    ResultsWindow(ResultsCollector_t &rc, QWidget *parent = NULL);
    ResultsWindowController *controller() { return m_controller; }
    ~ResultsWindow();

private:
    ResultsWindowController *m_controller;
    ResultsCollector_t &m_resultsCollection;
    ResultTreeView *g_treeView;
    QPushButton *g_deleteButton;

    friend class ResultsWindowController;
};

#endif // RESULTS_WINDOW_HH
