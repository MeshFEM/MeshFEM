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
#include "GlobalTypes.hh"

class QTreeWidget;

class ResultsWindow : public QWidget
{
    Q_OBJECT

public slots:
    void resultsUpdated();

public:
    ResultsWindow(ResultsCollector_t &rc, QWidget *parent = NULL);
    ~ResultsWindow();

private:
    ResultsCollector_t &m_resultsCollection;
    QTreeWidget *g_treeView;
};

#endif // RESULTS_WINDOW_HH
