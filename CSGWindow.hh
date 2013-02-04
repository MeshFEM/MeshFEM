////////////////////////////////////////////////////////////////////////////////
// CSGWindow.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      The main window for CSG operations and visualization.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/28/2013 14:52:21
////////////////////////////////////////////////////////////////////////////////
#ifndef CSGWINDOW_HH
#define CSGWINDOW_HH

#include <QMainWindow>
#include "GlobalTypes.hh"
#include "CSGWindowController.hh"
#include "MeshlessFEM.hh"

class CSGWindow : public QMainWindow
{
    Q_OBJECT

public:
    CSGWindow(MeshlessFEM_t &fem);
    ~CSGWindow() { delete controller; }
private:
    CSGWindowController *controller;
};

#endif // CSGWINDOW_HH
