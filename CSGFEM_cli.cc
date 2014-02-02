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
#include <iostream>
#include <string>
#include <cstdlib>

#include "AnalysisSettings.hh"
#include "Solver.hh"
#include "GlobalTypes.hh"

#include <boost/program_options.hpp>
namespace po = boost::program_options;
using namespace std;

void usage(int exitVal, const po::options_description &visible_opts)
{
    cout << "Usage: CSGFEM_cli [options] input.csg output.msh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on sucess)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    AnalysisSettings settings;
    po::options_description analysis_opts("Analysis Settings");
    settings.getOptions(analysis_opts);

    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("input-file", po::value<string>(), "input CSG file")
        ("output-file", po::value<string>(), "output MSH file")
        ;

    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("output-file", 1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message");
    visible_opts.add(analysis_opts);

    po::options_description cli_opts;
    cli_opts.add(visible_opts).add(hidden_opts);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).
                  options(cli_opts).positional(p).run(), vm);
        po::notify(vm);
    }
    catch (std::exception &e) {
        cout << "Error: " << e.what() << endl << endl;
        usage(1, visible_opts);
    }

    if (vm.count("help"))
        usage(0, visible_opts);

    if ((vm.count("input-file") == 0) || (vm.count("output-file") == 0)) {
        cout << "Error: must specify input and output files" << endl;
        usage(1, visible_opts);
    }

    // MeshlessFEM_t fem(csgTree, settings, solver);

    return 0;
}
