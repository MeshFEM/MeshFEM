////////////////////////////////////////////////////////////////////////////////
// CSGFEM.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      The CGSFEM application.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/28/2013 14:57:06
////////////////////////////////////////////////////////////////////////////////
#include <QApplication>
#include "CSGWindow.hh"
#include "GlobalTypes.hh"
#include "SolverLibrary.hh"
#include "AnalysisSettings.hh"
#include "CSGWindowController.hh"

#ifdef HAS_MATLAB
#include "LazyMatlabInterfaces.hh"
#endif

#include <map>
#include <string>

#include <iostream>
#include <stdexcept>

using namespace std;

class MyApplication : public QApplication
{
public:
    MyApplication(int &argc, char **argv)
        : QApplication(argc, argv) { }

    bool notify(QObject *receiver, QEvent *event) {
        bool done = true;
        try {
            done = QApplication::notify(receiver, event);
        }
        catch (const exception &ex) {
            cout << "Exception caught during signal: " << ex.what() << endl;
        }
        return done;
    }
};

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on sucess)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    MyApplication app(argc, argv);
    CSGTree_t csgTree;
    AnalysisSettings settings;

    typedef CSGTree_t::Real Real;
#ifdef HAS_MATLAB
    LazyQMatlab matlab;
    SolverLibrary<Real> solvers(matlab);
#else
    SolverLibrary<Real> solvers();
#endif

    MeshlessFEM_t fem(csgTree, settings, solvers);
    ResultsCollector_t results;

    CSGWindow window(fem, settings, solvers, results);
    window.setWindowTitle("CSG Finite Element Structure Analysis");
    window.resize(1280, 768);
    window.show();

    if (argc > 1)
        window.getController()->loadCSG(QString(argv[1]));

    return app.exec();
}
