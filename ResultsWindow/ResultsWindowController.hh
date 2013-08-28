////////////////////////////////////////////////////////////////////////////////
// ResultsWindowController.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Controls the results collection viewer window.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/30/2013 17:11:35
////////////////////////////////////////////////////////////////////////////////
#ifndef RESULTS_WINDOW_CONTROLLER_HH
#define RESULTS_WINDOW_CONTROLLER_HH
#include <QObject>
#include <string>
#include <map>

class QTreeWidgetItem;
class ResultsWindow;

class ResultsWindowController : public QObject
{
    Q_OBJECT

public slots:
    void itemActivated(QTreeWidgetItem *item, int col);
    void itemChanged(QTreeWidgetItem *item, int col);
    void itemDeleted(QTreeWidgetItem *item);

    void resultsUpdated();
    void selectResult(const std::string &path);
    void deleteSelection();
    void groupingCheckToggled(bool);

signals:
    void resultDeslected();
    void resultSelected(const std::string &path);
    
public:
    ResultsWindowController(ResultsWindow &window)
        : m_window(window), m_currentResultItem(NULL),
          m_modelMajorGrouping(true), m_autoAdjustingChecks(false) { }

    ~ResultsWindowController();
private:
    void selectResult(QTreeWidgetItem *item);

    std::map<std::string, QTreeWidgetItem *> m_pathToItem;

    ResultsWindow &m_window;
    QTreeWidgetItem *m_currentResultItem;

    bool m_modelMajorGrouping;

    // Used to prevent the process of adjusting result check boxes from calling
    // itself recursively...
    // Note: this is not a thread-safe solution.
    bool m_autoAdjustingChecks;
};

#endif // RESULTS_WINDOW_CONTROLLER_HH
