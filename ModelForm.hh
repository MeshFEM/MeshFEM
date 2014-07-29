////////////////////////////////////////////////////////////////////////////////
// ModelForm.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        GUI for the modeling controls.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/17/2013 15:54:52
////////////////////////////////////////////////////////////////////////////////
#ifndef MODEL_FORM_HH
#define MODEL_FORM_HH

#include <QObject>
#include <QWidget>

class CSGWindowController;

class ModelForm : public QWidget
{
    Q_OBJECT

public:
    ModelForm(CSGWindowController *controller, QWidget *parent = NULL);

};

#endif // MODEL_FORM_HH
