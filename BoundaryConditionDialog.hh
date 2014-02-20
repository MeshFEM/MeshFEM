////////////////////////////////////////////////////////////////////////////////
// BoundaryConditionDialog.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Dialog box for configuring boundary conditions. The condition to be
//        configured is passed by reference to the constructor and will be
//        updated if this dialog is accepted (and if the changes are valid).
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/19/2014 17:31:43
////////////////////////////////////////////////////////////////////////////////
#ifndef BOUNDARY_CONDITION_DIALOG_HH
#define BOUNDARY_CONDITION_DIALOG_HH

#include "BoundaryConditions.hh"
#include "GlobalTypes.hh"

#include <QObject>
#include <QDialog>

class QLineEdit;
class QComboBox;
class QTabWidget;

class BoundaryConditionDialog : public QDialog
{
    Q_OBJECT

public:
    typedef BoundaryConditions<Vector> BCs;
    BoundaryConditionDialog(BCs &c, int condition = -1, QWidget *parent = 0);
    ~BoundaryConditionDialog();

private slots:
    void selectedConditionChanged(int newIdx);

private:
    QComboBox *g_conditionSelector;

    QLineEdit *g_regionMinX, *g_regionMaxX, *g_regionMinY, *g_regionMaxY;
    QLineEdit *g_tractionX, *g_tractionY, *g_pressure;
    QTabWidget *g_tractionPressureTab;

    // Reference to the condition we are editing.
    int m_selectedCondition;
    BCs &m_conditions;
};

#endif // BOUNDARY_CONDITION_DIALOG_HH
