////////////////////////////////////////////////////////////////////////////////
// ResultsWindowController.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Controls the results collection viewer window.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/30/2013 17:22:30
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <QTreeWidgetItem>
#include <cassert>
#include <vector>
#include <string>
#include <stack>
#include <memory>
#include <QList>
#include <QListWidget>
#include <QLineEdit>
#include <QMessageBox>
#include <QFileDialog>
#include <stdexcept>
#include <regex>
#include <QBasicTimer>
#include <QTimerEvent>

#include "ResultsWindowController.hh"
#include "MeshlessFEM.hh"
#include "ResultsCollector.hh"
#include "ResultsWindow.hh"
#include "ResultTreeView.hh"
#include "Flipbook.hh"

using namespace std;

void ResultsWindowController::browserTabChanged(int newTab) {
    if (newTab == 1) {
        // When search tab is selected, focus on the search field
        m_window.g_searchField->setFocus();
    }
}

// Special QTreeWidgetItem that knows when it is deleted so we can make sure to
// invalidate ResultsWindowController's current result item pointer.
class ResultTreeWidgetItem : public QTreeWidgetItem
{
public:
    ResultTreeWidgetItem(shared_ptr<ResultsWindowController> controller,
            QTreeWidgetItem *parent, const QStringList &strings,
            int type = QTreeWidgetItem::Type)
        : QTreeWidgetItem(parent, strings, type), m_controller(controller) { }


    string path() const {
        return data(0, Qt::UserRole).toString().toStdString();
    }

    bool isResultItem() const {
        return data(0, Qt::UserRole + 1).toBool();
    }

    ~ResultTreeWidgetItem() {
        shared_ptr<ResultsWindowController> c = m_controller.lock();
        if (c) {
            c->itemDeleted(this);
        }
    }
private:
    weak_ptr<ResultsWindowController> m_controller;
};

// Items in the filtered results list point to the original result item
class FilterListWidgetItem : public QListWidgetItem
{
public:
    FilterListWidgetItem(const char *path, ResultTreeWidgetItem *ri)
        : QListWidgetItem(path), m_resultItem(ri) { }

    const ResultTreeWidgetItem *resultItem() const { return m_resultItem; }
    ResultTreeWidgetItem *resultItem() { return m_resultItem; }
private:
    ResultTreeWidgetItem *m_resultItem;
};

// Select the result with a particular path.
void ResultsWindowController::selectResult(const string &path)
{
    if (path != "") {
        auto item_it = m_pathToItem.find(path);
        if (item_it == m_pathToItem.end()) {
            throw(runtime_error(string("Path not found in path->item map: ") +
                                path));
        }

        selectResult(item_it->second);
        m_window.g_treeView->scrollToItem(item_it->second);
    }
}

// Select a result (or NULL to deselect)
void ResultsWindowController::selectResult(QTreeWidgetItem *item)
{
    m_autoAdjustingChecks = true;
    if (m_currentResultItem != item) {
        if (m_currentResultItem != NULL) {
            m_currentResultItem->setCheckState(0, Qt::Unchecked);

            QFont font = m_currentResultItem->font(0);
            font.setBold(false);
            m_currentResultItem->setFont(0, font);
        }

        if (item != NULL) {
            item->setCheckState(0, Qt::Checked);
            QFont font = item->font(0);
            font.setBold(true);
            item->setFont(0, font);

            ResultTreeWidgetItem *ri =
                dynamic_cast<ResultTreeWidgetItem *>(item);
            assert(ri);

            emit resultSelected(ri->path());
        }

        if (item == NULL) {
            emit resultDeslected();
        }

        m_currentResultItem = item;
    }

    syncSearchChecks();

    m_autoAdjustingChecks = false;
}

vector<string> ResultsWindowController::selectedResultPaths() const {
    vector<string> paths;

    QList<QTreeWidgetItem *> items = m_window.g_treeView->selectedItems();

    foreach (QTreeWidgetItem *i, items) {
        ResultTreeWidgetItem *ri = dynamic_cast<ResultTreeWidgetItem *>(i);
        assert(ri);
        if (ri->isResultItem()) paths.push_back(ri->path());
    }
    return paths;
}


void ResultsWindowController::itemActivated(QTreeWidgetItem *item, int col)
{
    Q_UNUSED(col);
    bool isResultItem = item->data(0, Qt::UserRole + 1).toBool();
    if (isResultItem) {
        item->setCheckState(0, (item->checkState(0) == Qt::Checked) ?
                                Qt::Unchecked : Qt::Checked);
    }
}

void ResultsWindowController::itemChanged(QTreeWidgetItem *item, int col)
{
    Q_UNUSED(col);
    if (!m_autoAdjustingChecks) {
        bool isResultItem = item->data(0, Qt::UserRole + 1).toBool();
        assert(isResultItem);
        if (item->checkState(0) == Qt::Checked) {
            selectResult(item);
        }
        else {
            selectResult(NULL);
        }
    }
}

void ResultsWindowController::selectionChanged()
{
    if (m_synchingResultSelections)
        return;

    m_synchingResultSelections = true;

    // // Make sure only result nodes are selected...
    // foreach (QTreeWidgetItem *i, m_window.g_treeView->selectedItems()) {
    //     ResultTreeWidgetItem *ri = dynamic_cast<ResultTreeWidgetItem *>(i);
    //     assert(ri);
    //     ri->setSelected(ri->isResultItem());
    // }

    // Sync the search list items
    for (int row = 0; row < m_window.g_filterView->count(); ++row) {
        QListWidgetItem *i = m_window.g_filterView->item(row);
        FilterListWidgetItem *fli = dynamic_cast<FilterListWidgetItem *>(i);
        assert(fli);
        fli->setSelected(fli->resultItem()->isSelected());
    }

    m_synchingResultSelections = false;
}

// Synchronize the search list checkboxes with the results tree.
void ResultsWindowController::syncSearchChecks()
{
    bool oldAuto = m_autoAdjustingChecks;
    m_autoAdjustingChecks = true;
    for (int row = 0; row < m_window.g_filterView->count(); ++row) {
        QListWidgetItem *i = m_window.g_filterView->item(row);
        FilterListWidgetItem *fli = dynamic_cast<FilterListWidgetItem *>(i);
        assert(fli);
        fli->setCheckState(fli->resultItem()->checkState(0));
        if (fli->checkState() == Qt::Checked) {
            m_window.g_filterView->scrollToItem(fli);
        }
    }
    m_autoAdjustingChecks = oldAuto;
}

void ResultsWindowController::searchItemActivated(QListWidgetItem *item)
{
    item->setCheckState((item->checkState() == Qt::Checked) ?
            Qt::Unchecked : Qt::Checked);
}

void ResultsWindowController::searchItemChanged(QListWidgetItem *item)
{
    if (m_autoAdjustingChecks) return;

    if (item->checkState() == Qt::Checked) {
        FilterListWidgetItem *fli = dynamic_cast<FilterListWidgetItem *>(item);
        assert(fli);
        selectResult(fli->resultItem());
    }
    else {
        selectResult(NULL);
    }

    syncSearchChecks();
}

void ResultsWindowController::searchSelectionChanged()
{
    if (m_synchingResultSelections)
        return;

    m_synchingResultSelections = true;

    // Clear tree selection
    foreach (QTreeWidgetItem *i, m_window.g_treeView->selectedItems()) {
        ResultTreeWidgetItem *ri = dynamic_cast<ResultTreeWidgetItem *>(i);
        assert(ri);
        ri->setSelected(false);
    }

    // Sync the tree selection
    for (int row = 0; row < m_window.g_filterView->count(); ++row) {
        QListWidgetItem *i = m_window.g_filterView->item(row);
        FilterListWidgetItem *fli = dynamic_cast<FilterListWidgetItem *>(i);
        assert(fli);
        if (fli->isSelected()) {
            fli->resultItem()->setSelected(true);
            m_window.g_treeView->scrollToItem(fli->resultItem());
        }
    }

    m_synchingResultSelections = false;
}

void ResultsWindowController::modelItemActivated(QListWidgetItem *item) {
    emit modelSelected(item->text().toStdString());
}

void ResultsWindowController::settingsItemActivated(QListWidgetItem *item) {
    emit settingsSelected(item->text().toStdString());
}

// Invalidate current item pointer when the pointed-to item is deleted.
// Also, update the path->item dictionary
void ResultsWindowController::itemDeleted(QTreeWidgetItem *item)
{
    if (m_currentResultItem == item)
        m_currentResultItem = NULL;

    ResultTreeWidgetItem *ri = dynamic_cast<ResultTreeWidgetItem *>(item);
    assert(ri);

    auto item_it = m_pathToItem.find(ri->path());
    if (item_it == m_pathToItem.end()) {
        throw(runtime_error(string("Path not found in path->item map: ") +
                    ri->path()));
    }
    m_pathToItem.erase(item_it);
}

class TreeWidgetItemGenerator {
public:
    ////////////////////////////////////////////////////////////////////////////
    /*! Widget item generator constructor
    //  @param[in]  controller      reference to the controller that manages the
    //                              results window.
    //  @param[in]  modelMajorDFS   whether the DFS will visit model nodes first
    //                              (as opposed to setting nodes)
    *///////////////////////////////////////////////////////////////////////////
    TreeWidgetItemGenerator(shared_ptr<ResultsWindowController> c,
                            bool modelMajorDFS)
        : m_controller(c), m_modelMajorDFS(modelMajorDFS) { }

    void preVisit(const string &name, bool hasResult) {
        QTreeWidgetItem *p = m_parentStack.empty() ? NULL : m_parentStack.top();
        ResultTreeWidgetItem *newItem = new ResultTreeWidgetItem(m_controller,
                p, QStringList(QString::fromStdString(name)));

        // All results should have a checkbox
        if (hasResult)
            newItem->setCheckState(0, Qt::Unchecked);
        m_parentStack.push(newItem);

        // Swap the first two path components if the search isn't model-major.
        // (So that the path is always of the form model:setting:...)
        if (!m_modelMajorDFS && (m_pathStack.size() == 1))
            m_pathStack.prepend(QString::fromStdString(name));
        else
            m_pathStack.append(QString::fromStdString(name));

        // Store the path inside the widget item.
        newItem->setData(0, Qt::UserRole, QVariant(m_pathStack.join(":")));
        newItem->setData(0, Qt::UserRole + 1, QVariant(hasResult));

        items.append(newItem);
    }

    void postVisit() {
        assert(!m_parentStack.empty());
        m_parentStack.pop();

        // Handle the case when the search isn't model-major and the first two
        // entries of the path stack are in the opposite order from the dfs.
        if (!m_modelMajorDFS && (m_pathStack.size() == 2))
            m_pathStack.removeFirst();
        else
            m_pathStack.removeLast();
    }

    // Note, these will be owned by the tree view, so they needn't be cleaned up
    // by a destructor.
    QList<QTreeWidgetItem *> items;

private:
    shared_ptr<ResultsWindowController> m_controller;
    stack<ResultTreeWidgetItem *> m_parentStack;
    QStringList m_pathStack;
    bool m_modelMajorDFS;
};

void ResultsWindowController::resultsUpdated()
{
    m_window.g_filterView->clear();
    m_window.g_treeView->clear();

    TreeWidgetItemGenerator tgen(m_window.m_controller, m_modelMajorGrouping);
    m_window.m_resultsCollection.dfs(m_modelMajorGrouping ?
            ResultsCollector_t::KEY_ORDER_MODEL_SETTINGS :
            ResultsCollector_t::KEY_ORDER_SETTINGS_MODEL, tgen);
    foreach(QTreeWidgetItem *i, tgen.items) {
        ResultTreeWidgetItem *ri = dynamic_cast<ResultTreeWidgetItem *>(i);
        assert(ri);
        m_pathToItem.insert(make_pair(ri->path(), i));
    }

    m_window.g_treeView->insertTopLevelItems(0, tgen.items);

    selectResult(m_window.m_resultsCollection.lastResultPath());

    // When the results update, the models/settings lists should have changed.
    modelsUpdated();
    settingsUpdated();

    m_window.show();
    m_window.raise();
    m_window.activateWindow();
}

void ResultsWindowController::modelsUpdated()
{
    vector<string> names = m_window.m_resultsCollection.getModelNames();
    m_window.g_modelListView->clear();
    for (const string &name : names) {
        m_window.g_modelListView->addItem(new QListWidgetItem(name.c_str()));
    }
}

void ResultsWindowController::settingsUpdated()
{
    vector<string> names = m_window.m_resultsCollection.getSettingsNames();
    m_window.g_settingsListView->clear();
    for (const string &name : names) {
        m_window.g_settingsListView->addItem(new QListWidgetItem(name.c_str()));
    }
}

void ResultsWindowController::deleteSelection()
{
    auto items = m_window.g_treeView->selectedItems();

    if (items.size() == 0)
        return;

    // Changing the results content should deselect the active result
    if (m_currentResultItem) {
        selectResult(NULL);
    }

    vector<string> pathsForDeletion;
    foreach(QTreeWidgetItem *i, items) {
        ResultTreeWidgetItem *ri = dynamic_cast<ResultTreeWidgetItem *>(i);
        pathsForDeletion.push_back(ri->path());
    }

    m_window.m_resultsCollection.removeResultsWithPaths(pathsForDeletion);
    resultsUpdated();
}

void ResultsWindowController::dumpRaw()
{
    vector<string> paths = selectedResultPaths();
    if (paths.empty())
        return;

    QString dir = QFileDialog::getExistingDirectory(0,
            "Result Output Directory", QString(), QFileDialog::ShowDirsOnly);

    try {
        for (const string &rpath : paths) {
            string fpath = dir.toStdString() + "/" + rpath;
            shared_ptr<const ResultsCollector_t::Result> r =
                m_window.m_resultsCollection.getResultWithPath(rpath);
            r->dump(fpath);
        }
    }
    catch (exception &e) {
        string errorString("Dumping Results Failed: ");
        errorString += e.what();
        QMessageBox mbox(QMessageBox::Critical, "Dumping Results Failed",
                         errorString.c_str(), QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.exec();
    }
}

void ResultsWindowController::generateFlipbook()
{
    vector<string> paths = selectedResultPaths();
    if (paths.empty())
        return;

    QString dir = QFileDialog::getExistingDirectory(0,
            "Result Output Directory", QString(), QFileDialog::ShowDirsOnly);

    m_flipbook = std::shared_ptr<Flipbook>(new Flipbook(dir.toStdString(),
                &m_window.m_resultsCollection, paths));
    m_flipbookTimer.start(0, this);
    emit attachFlipbook(m_flipbook);
}

void ResultsWindowController::timerEvent(QTimerEvent *event) {
    if (event->timerId() == m_flipbookTimer.timerId()) {
        if (m_flipbook && m_flipbook->active())
            selectResult(m_flipbook->path());
        else
            m_flipbookTimer.stop();
    }
    else {
        // Pass up the unhandled timer event
        QObject::timerEvent(event);
    }
}

void ResultsWindowController::groupingCheckToggled(bool checked)
{
    m_modelMajorGrouping = checked;
    resultsUpdated();
}

void ResultsWindowController::runSearch()
{
    m_window.g_filterView->clear();
    QString searchPattern = m_window.g_searchField->text();
    try {
        regex pattern(searchPattern.toStdString());

        FilterListWidgetItem *firstItem = nullptr;

        for (auto &pathItemPair : m_pathToItem) {
            ResultTreeWidgetItem *ri =
                dynamic_cast<ResultTreeWidgetItem *>(pathItemPair.second);
            if (ri->isResultItem()) {
                const string &path = pathItemPair.first;
                if (regex_search(path, pattern)) {
                    FilterListWidgetItem *item =
                        new FilterListWidgetItem(path.c_str(), ri);
                    item->setCheckState(ri->checkState(0));
                    m_window.g_filterView->addItem(item);

                    if (firstItem == nullptr) firstItem = item;
                }
            }
        }

        if (firstItem) firstItem->setSelected(true);
        m_window.g_filterView->setFocus();
    }
    catch (regex_error &e) {
        string errorString("Parsing regex failed. ");
        errorString += e.what();
        
        QMessageBox mbox(QMessageBox::Critical, "Regex Failed",
                         errorString.c_str(), QMessageBox::Ok);
        mbox.setDefaultButton(QMessageBox::Ok);
        mbox.exec();
    }
}

ResultsWindowController::~ResultsWindowController()
{
}
