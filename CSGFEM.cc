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
    CSGWindow window(csgTree);
    window.setWindowTitle("CSG Finite Element Structure Analysis");
    window.resize(1280, 768);
    window.show();

    return app.exec();
}
