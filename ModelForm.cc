////////////////////////////////////////////////////////////////////////////////
// ModelForm.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        GUI for the modeling controls.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/17/2013 15:57:41
////////////////////////////////////////////////////////////////////////////////
#include "ModelForm.hh"

#include <QTreeView>
#include "CSGWindowController.hh"
#include "CSGTreeModel.hh"

ModelForm::ModelForm(CSGWindowController *controller, QWidget *parent)
{
    QTreeView *treeView = controller->csgTreeView();

    QVBoxLayout *modelSidebarLayout = new QVBoxLayout();
    modelSidebarLayout->addWidget(treeView);
    QPushButton *extractBoundaryButton = new QPushButton("Extract/Dump Boundary");
    treeView->setSizePolicy(QSizePolicy::MinimumExpanding,
                            QSizePolicy::MinimumExpanding);
    modelSidebarLayout->addWidget(treeView);
    modelSidebarLayout->addStretch(1);
    modelSidebarLayout->addWidget(extractBoundaryButton);
    modelSidebarLayout->setStretchFactor(treeView, 1);
    modelSidebarLayout->setStretchFactor(extractBoundaryButton, 0);

    setLayout(modelSidebarLayout);

    // Set up connections
    QObject::connect(treeView->selectionModel(),
                     SIGNAL(selectionChanged(const QItemSelection &,
                                             const QItemSelection &)),
                     controller, SLOT(csgTreeSelectionChanged(
                                        const QItemSelection &,
                                        const QItemSelection &)));
    QObject::connect(controller, SIGNAL(csgTreeApplyModifiedSelection(
                                        const QItemSelection &,
                                        QItemSelectionModel::SelectionFlags)),
                     treeView->selectionModel(), SLOT(select(
                                        const QItemSelection &,
                                        QItemSelectionModel::SelectionFlags)));
}
