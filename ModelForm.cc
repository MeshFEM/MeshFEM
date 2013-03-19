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
#include <QPushButton>
#include "CSGWindowController.hh"
#include "CSGTreeModel.hh"

ModelForm::ModelForm(CSGWindowController *controller, QWidget *parent)
{
    Q_UNUSED(parent)
    QTreeView *treeView = controller->csgTreeView();
    QVBoxLayout *layout = new QVBoxLayout();
    treeView->setSizePolicy(QSizePolicy::MinimumExpanding,
                            QSizePolicy::MinimumExpanding);
    layout->addWidget(treeView);
    layout->setStretchFactor(treeView, 1);
    layout->setContentsMargins(5, 5, 5, 5);

    setLayout(layout);

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
