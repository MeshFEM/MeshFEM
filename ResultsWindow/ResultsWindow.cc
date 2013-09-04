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

    QHBoxLayout *r_buttonLayout = new QHBoxLayout();
    QPushButton *r_deleteButton = new QPushButton("Delete");
    QPushButton *r_mshButton = new QPushButton("Dump .MSH");
    QPushButton *r_rawButton = new QPushButton("Dump raw");
    QPushButton *r_flipbookButton = new QPushButton("Flipbook");
    r_buttonLayout->addWidget(r_deleteButton);
    r_buttonLayout->addWidget(r_mshButton);
    r_buttonLayout->addWidget(r_rawButton);
    r_buttonLayout->addWidget(r_flipbookButton);
    resultsTreeLayout->addLayout(r_buttonLayout);

    viewTab->addTab(resultsTreeWidget, "Results Tree");

    QVBoxLayout *resultsFilterLayout = new QVBoxLayout();
    resultsFilterLayout->setContentsMargins(0, 0, 0, 0);
    g_filterView = new QListWidget();
    g_filterView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    resultsFilterLayout->addWidget(g_filterView);
    QHBoxLayout *searchLayout = new QHBoxLayout();
    g_searchField = new QLineEdit();
    QPushButton *g_searchButton = new QPushButton("Search");
    searchLayout->addWidget(g_searchField);
    searchLayout->addWidget(g_searchButton);

    resultsFilterLayout->addLayout(searchLayout);

    QPushButton *s_deleteButton = new QPushButton("Delete");
    QPushButton *s_mshButton = new QPushButton("Dump .MSH");
    QPushButton *s_rawButton = new QPushButton("Dump raw");
    QPushButton *s_flipbookButton = new QPushButton("Flipbook");
    QHBoxLayout *s_buttonLayout = new QHBoxLayout();
    s_buttonLayout->addWidget(s_deleteButton);
    s_buttonLayout->addWidget(s_mshButton);
    s_buttonLayout->addWidget(s_rawButton);
    s_buttonLayout->addWidget(s_flipbookButton);
    resultsFilterLayout->addLayout(s_buttonLayout);

    QWidget *resultsFilterWidget = new QWidget();
    resultsFilterWidget->setLayout(resultsFilterLayout);
    viewTab->addTab(resultsFilterWidget, "Search");

    QWidget *modelsWidget = new QWidget();
    g_modelListView = new QListWidget();
    QVBoxLayout *modelLayout = new QVBoxLayout();
    modelLayout->addWidget(g_modelListView);
    modelLayout->setContentsMargins(0, 0, 0, 0);
    modelsWidget->setLayout(modelLayout);
    viewTab->addTab(modelsWidget, "Models");

    QWidget *settingsWidget = new QWidget();
    g_settingsListView = new QListWidget();
    QVBoxLayout *settingsLayout = new QVBoxLayout();
    settingsLayout->addWidget(g_settingsListView);
    settingsLayout->setContentsMargins(0, 0, 0, 0);
    settingsWidget->setLayout(settingsLayout);
    viewTab->addTab(settingsWidget, "Settings");

    layout->addWidget(viewTab);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);

    QObject::connect(viewTab, SIGNAL(currentChanged(int)),
                     controller(), SLOT(browserTabChanged(int)));
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

    QObject::connect(g_searchField, SIGNAL(returnPressed()),
                     g_searchButton, SLOT(click()));
    QObject::connect(g_searchButton, SIGNAL(clicked()),
                     controller(), SLOT(runSearch()));

    QObject::connect(g_modelSettingsGrouping, SIGNAL(toggled(bool)),
                     controller(), SLOT(groupingCheckToggled(bool)));

    QObject::connect(r_deleteButton, SIGNAL(clicked()),
                     controller(), SLOT(deleteSelection()));
    QObject::connect(r_rawButton, SIGNAL(clicked()),
                     controller(), SLOT(dumpRaw()));
    QObject::connect(s_deleteButton, SIGNAL(clicked()),
                     controller(), SLOT(deleteSelection()));
    QObject::connect(s_rawButton, SIGNAL(clicked()),
                     controller(), SLOT(dumpRaw()));

    m_controller->resultsUpdated();
}

ResultsWindow::~ResultsWindow()
{
}
