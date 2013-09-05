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
class QListWidgetItem;
class QItemSelection;
class ResultsWindow;

class ResultsWindowController : public QObject
{
    Q_OBJECT

public slots:
    void browserTabChanged(int);
    void itemActivated(QTreeWidgetItem *item, int col);
    void itemChanged(QTreeWidgetItem *item, int col);
    void selectionChanged();

    void searchItemActivated(QListWidgetItem *item);
    void searchItemChanged(QListWidgetItem *item);
    void searchSelectionChanged();

    void modelItemActivated(QListWidgetItem *item);
    void settingsItemActivated(QListWidgetItem *item);

    void itemDeleted(QTreeWidgetItem *item);

    void modelsUpdated();
    void settingsUpdated();
    void resultsUpdated();
    void selectResult(const std::string &path);
    void deleteSelection();
    void dumpRaw();
    void groupingCheckToggled(bool);
    void runSearch();

signals:
    void resultDeslected();
    void resultSelected(const std::string &path);
    void modelSelected(const std::string &name);
    void settingsSelected(const std::string &name);
    
public:
    ResultsWindowController(ResultsWindow &window)
        : m_window(window), m_currentResultItem(NULL),
          m_modelMajorGrouping(true), m_autoAdjustingChecks(false),
          m_synchingResultSelections(false) { }

    ~ResultsWindowController();
private:
    void selectResult(QTreeWidgetItem *item);
    void syncSearchChecks();

    std::map<std::string, QTreeWidgetItem *> m_pathToItem;

    ResultsWindow &m_window;
    QTreeWidgetItem *m_currentResultItem;

    bool m_modelMajorGrouping;

    // Used to prevent the process of adjusting result check boxes from calling
    // itself recursively...
    // Note: this is not a thread-safe solution.
    bool m_autoAdjustingChecks;

    // Prevent feedback loops when synchronizing selections
    bool m_synchingResultSelections;
};

#endif // RESULTS_WINDOW_CONTROLLER_HH
