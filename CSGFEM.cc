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
#include "QMatlabInterface.hh"
#include "Solver.hh"
#include "AnalysisSettings.hh"

#include <map>
#include <string>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on sucess)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    CSGTree_t csgTree;
    AnalysisSettings settings;

    typedef CSGTree_t::Real Real;
    QMatlabInterface *matlabInterface = new QMatlabInterface();
    SolverLibrary<Real> solvers(matlabInterface);

    MeshlessFEM_t fem(csgTree, settings, solvers);

    CSGWindow window(fem, settings, solvers);
    window.setWindowTitle("CSG Finite Element Structure Analysis");
    window.resize(1280, 768);

    matlabInterface->show();
    window.show();

    return app.exec();
}
