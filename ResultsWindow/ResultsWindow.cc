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
#include "ResultTreeView.hh"
#include "ResultsWindowController.hh"

#include <QPushButton>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QHBoxLayout>

using namespace std;

ResultsWindow::ResultsWindow(ResultsCollector_t &rc, QWidget *parent)
    : QWidget(parent), m_resultsCollection(rc)
{
    m_controller = new ResultsWindowController(*this);

    setWindowTitle("Results Browser");
    QVBoxLayout *layout = new QVBoxLayout();
    g_treeView = new ResultTreeView();
    g_treeView->setColumnCount(1);
    g_treeView->header()->close();
    g_treeView->setSelectionMode(QAbstractItemView::ExtendedSelection);

    g_deleteButton = new QPushButton("Delete Selection");
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(g_deleteButton);

    layout->addWidget(g_treeView);
    layout->addLayout(buttonLayout);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);

    QObject::connect(g_treeView, SIGNAL(currentItemChanged(QTreeWidgetItem *, QTreeWidgetItem *)),
                     m_controller, SLOT(currentItemChanged(QTreeWidgetItem *, QTreeWidgetItem *)));
    QObject::connect(g_treeView, SIGNAL(itemSelectionChanged()),
                     m_controller, SLOT(itemSelectionChanged()));
    QObject::connect(g_treeView, SIGNAL(itemActivated(QTreeWidgetItem *, int)),
                     m_controller, SLOT(itemActivated(QTreeWidgetItem *, int)));
    QObject::connect(g_treeView, SIGNAL(itemChanged(QTreeWidgetItem *, int)),
                     m_controller, SLOT(itemChanged(QTreeWidgetItem *, int)));

    m_controller->resultsUpdated();
}

ResultsWindow::~ResultsWindow()
{
    delete m_controller;
}
