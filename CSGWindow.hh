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

#include <QObject>
#include <QMainWindow>
#include "GlobalTypes.hh"
#include "MeshlessFEM.hh"
#include "ResultsCollector.hh"

#include "AnalysisSettings.hh"
#include "ViewSettings.hh"
#include "ViewSettingsWidget.hh"
#include "ResultsWindow/ResultsWindow.hh"
#include "CSGWindow.hh"

class CSGWindowController;
struct AnalysisSettings;
struct ViewSettings;
class  ViewSettingsWidget;

class CSGWindow : public QMainWindow
{
    Q_OBJECT

public:
    CSGWindow(MeshlessFEM_t &fem, AnalysisSettings &settings,
              SolverLibrary<Scalar> &solvers,
              ResultsCollector_t &results);
    CSGWindowController *getController() { return controller; }
    const CSGWindowController *getController() const { return controller; }
    ~CSGWindow();

private slots:
    void showViewSettings();
    void showResultsWindow();

private:
    CSGWindowController *controller;
    ViewSettings vsettings;
    ViewSettingsWidget *g_vsWidget;
    ResultsWindow *g_resultsWindow;
};

#endif // CSGWINDOW_HH
