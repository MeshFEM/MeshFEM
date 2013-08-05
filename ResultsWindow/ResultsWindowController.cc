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
#include <string>
#include <stack>
#include <QList>
#include <stdexcept>

#include "ResultsWindowController.hh"
#include "MeshlessFEM.hh"
#include "ResultsCollector.hh"
#include "ResultsWindow.hh"
#include "ResultTreeView.hh"

using namespace std;

// Special QTreeWidgetItem that knows when it is deleted so we can make sure to
// invalidate ResultsWindowController's current result item pointer.
class ResultTreeWidgetItem : public QTreeWidgetItem
{
public:
    ResultTreeWidgetItem(ResultsWindowController &controller,
            QTreeWidgetItem *parent, const QStringList &strings,
            int type = QTreeWidgetItem::Type)
        : QTreeWidgetItem(parent, strings, type), m_controller(controller) { }


    std::string path() const {
        return data(0, Qt::UserRole).toString().toStdString();
    }

    bool isResultItem() const {
        return data(0, Qt::UserRole + 1).toBool();
    }

    ~ResultTreeWidgetItem() {
        m_controller.itemDeleted(this);
    }
private:
    ResultsWindowController &m_controller;
};

void ResultsWindowController::currentItemChanged(QTreeWidgetItem *current,
                                                 QTreeWidgetItem *previous)
{
}

void ResultsWindowController::itemSelectionChanged()
{
}

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
        }

        m_currentResultItem = item;
    }
    m_autoAdjustingChecks = false;
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
    TreeWidgetItemGenerator(ResultsWindowController &c, bool modelMajorDFS)
        : m_controller(c), m_modelMajorDFS(modelMajorDFS) { }
    void preVisit(const string &name, bool hasResult) {
        ResultTreeWidgetItem *newItem = NULL;
        QTreeWidgetItem *p = m_parentStack.empty() ? NULL : m_parentStack.top();
        newItem = new ResultTreeWidgetItem(m_controller, p,
                QStringList(QString::fromStdString(name)));

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
        m_pathStack.removeLast();
    }

    // Note, these will be owned by the tree view, so they needn't be cleaned up
    // by a destructor.
    QList<QTreeWidgetItem *> items;

private:
    ResultsWindowController &m_controller;
    stack<ResultTreeWidgetItem *> m_parentStack;
    QStringList m_pathStack;
    bool m_modelMajorDFS;
};

void ResultsWindowController::resultsUpdated()
{
    m_window.g_treeView->clear();

    TreeWidgetItemGenerator tgen(*this, true);
    m_window.m_resultsCollection.dfs(
            ResultsCollector_t::KEY_ORDER_MODEL_SETTINGS, tgen);
    foreach(QTreeWidgetItem *i, tgen.items) {
        ResultTreeWidgetItem *ri = dynamic_cast<ResultTreeWidgetItem *>(i);
        assert(ri);
        m_pathToItem.insert(make_pair(ri->path(), i));
    }

    m_window.g_treeView->insertTopLevelItems(0, tgen.items);

    selectResult(m_window.m_resultsCollection.lastResultPath());

    m_window.show();
    m_window.raise();
    m_window.activateWindow();
}

ResultsWindowController::~ResultsWindowController()
{
}
