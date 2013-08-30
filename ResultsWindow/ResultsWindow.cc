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

#include <QtGui>
#include <QPushButton>
#include <QCheckBox>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <QListWidget>
#include <QLineEdit>

using namespace std;

ResultsWindow::ResultsWindow(ResultsCollector_t &rc, QWidget *parent)
    : QWidget(parent), m_controller(new ResultsWindowController(*this)),
      m_resultsCollection(rc)
{
    setWindowTitle("Results Browser");

    QVBoxLayout *layout = new QVBoxLayout();
    QTabWidget *viewTab = new QTabWidget();

    QVBoxLayout *resultsTreeLayout = new QVBoxLayout();
    resultsTreeLayout->setContentsMargins(0, 0, 0, 0);
    g_treeView = new ResultTreeView();
    g_treeView->setColumnCount(1);
    g_treeView->header()->close();
    g_treeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    resultsTreeLayout->addWidget(g_treeView);
    QWidget *resultsTreeWidget = new QWidget();
    resultsTreeWidget->setLayout(resultsTreeLayout);

    g_modelSettingsGrouping = new QCheckBox("Model -> Settings Grouping");
    g_modelSettingsGrouping->setChecked(true);
    resultsTreeLayout->addWidget(g_modelSettingsGrouping);
    viewTab->addTab(resultsTreeWidget, "Results Tree");

    QVBoxLayout *resultsFilterLayout = new QVBoxLayout();
    resultsFilterLayout->setContentsMargins(0, 0, 0, 0);
    g_filterView = new QListWidget();
    g_filterView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    resultsFilterLayout->addWidget(g_filterView);
    QHBoxLayout *searchLayout = new QHBoxLayout();
    g_searchField = new QLineEdit();
    g_searchButton = new QPushButton("Search");
    g_searchButton->setDefault(true);
    searchLayout->addWidget(g_searchField);
    searchLayout->addWidget(g_searchButton);
    resultsFilterLayout->addLayout(searchLayout);

    QWidget *resultsFilterWidget = new QWidget();
    resultsFilterWidget->setLayout(resultsFilterLayout);
    viewTab->addTab(resultsFilterWidget, "Search");

    g_deleteButton = new QPushButton("Delete");
    g_mshButton = new QPushButton("Dump .MSH");
    g_rawButton = new QPushButton("Dump raw");
    g_flipbookButton = new QPushButton("Flipbook");

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(g_deleteButton);
    buttonLayout->addWidget(g_mshButton);
    buttonLayout->addWidget(g_rawButton);
    buttonLayout->addWidget(g_flipbookButton);

    layout->addWidget(viewTab);
    layout->addLayout(buttonLayout);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);

    QObject::connect(g_treeView, SIGNAL(itemActivated(QTreeWidgetItem *, int)),
                     controller(), SLOT(itemActivated(QTreeWidgetItem *, int)));
    QObject::connect(g_treeView, SIGNAL(itemChanged(QTreeWidgetItem *, int)),
                     controller(), SLOT(itemChanged(QTreeWidgetItem *, int)));

    QObject::connect(g_treeView, SIGNAL(itemSelectionChanged()),
                     controller(), SLOT(selectionChanged()));
    QObject::connect(g_filterView, SIGNAL(itemSelectionChanged()),
                     controller(), SLOT(searchSelectionChanged()));

    QObject::connect(g_filterView, SIGNAL(itemActivated(QListWidgetItem *)),
                     controller(), SLOT(searchItemActivated(QListWidgetItem *)));
    QObject::connect(g_filterView, SIGNAL(itemChanged(QListWidgetItem *)),
                     controller(), SLOT(searchItemChanged(QListWidgetItem *)));

    QObject::connect(g_searchButton, SIGNAL(clicked()),
                     controller(), SLOT(runSearch()));

    QObject::connect(g_modelSettingsGrouping, SIGNAL(toggled(bool)),
                     controller(), SLOT(groupingCheckToggled(bool)));
    QObject::connect(g_deleteButton, SIGNAL(clicked()),
                     controller(), SLOT(deleteSelection()));
    QObject::connect(g_rawButton, SIGNAL(clicked()),
                     controller(), SLOT(dumpRaw()));

    m_controller->resultsUpdated();
}

ResultsWindow::~ResultsWindow()
{
}
