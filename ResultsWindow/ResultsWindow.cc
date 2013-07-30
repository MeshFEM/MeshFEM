////////////////////////////////////////////////////////////////////////////////
// ResultsWindow.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements the results collection viewer window.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/29/2013 19:31:19
////////////////////////////////////////////////////////////////////////////////
#include "ResultsWindow.hh"
#include "MeshlessFEM.hh"
#include "ResultsCollector.hh"

#include <QTreeWidget>
#include <QHeaderView>
#include <QList>
#include <QVBoxLayout>
#include <stack>
#include <cassert>

ResultsWindow::ResultsWindow(ResultsCollector_t &rc, QWidget *parent)
    : QWidget(parent), m_resultsCollection(rc)
{
    setWindowTitle("Results");
    QVBoxLayout *layout = new QVBoxLayout();
    g_treeView = new QTreeWidget();
    g_treeView->setColumnCount(1);
    g_treeView->header()->close();
    layout->addWidget(g_treeView);
    setLayout(layout);
    resultsUpdated();
}

class TreeWidgetItemGenerator {
public:
    void preVisit(const std::string &name) {
        QTreeWidgetItem *newItem = NULL;
        if (!m_parentStack.empty()) {
            newItem = new QTreeWidgetItem(m_parentStack.top(),
                    QStringList(QString::fromStdString(name)));
        }
        else {
            newItem = new QTreeWidgetItem((QTreeWidgetItem *) NULL,
                    QStringList(QString::fromStdString(name)));
        }
        m_parentStack.push(newItem);
        items.append(newItem);
    }

    void postVisit() {
        assert(!m_parentStack.empty());
        m_parentStack.pop();
    }

    // Note, these will be owned by the tree view, so they needn't be cleaned up
    // by a destructor.
    QList<QTreeWidgetItem *> items;

private:
    std::stack<QTreeWidgetItem *> m_parentStack;
};

void ResultsWindow::resultsUpdated()
{
    // TODO: perform tree diff and make minimal changes.
    g_treeView->clear();

    TreeWidgetItemGenerator tgen;
    m_resultsCollection.dfs(ResultsCollector_t::KEY_ORDER_MODEL_SETTINGS, tgen);
    g_treeView->insertTopLevelItems(0, tgen.items);
}

ResultsWindow::~ResultsWindow()
{

}
