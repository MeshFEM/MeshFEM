////////////////////////////////////////////////////////////////////////////////
// CSGFEM_cli.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Command line interface for the CSGFEM program.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/01/2013 12:25:38
////////////////////////////////////////////////////////////////////////////////
#include "AnalysisSettings.hh"
#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on sucess)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    AnalysisSettings settings;
    // po::options_description opts("Analysis Settings");
    po::options_description opts;
    settings.getOptions(opts);
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    cout << opts << endl;

    return 0;
}
