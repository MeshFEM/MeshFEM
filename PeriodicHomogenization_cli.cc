////////////////////////////////////////////////////////////////////////////////
// PeriodicHomogenization_cli.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Commandline interface for periodic homogenization.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/01/2014 15:51:37
////////////////////////////////////////////////////////////////////////////////
#define DIM 3

#include "AnalysisSettings.hh"
#include "SolverLibrary.hh"
#include "MeshlessFEM3D.hh"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;
using namespace std;

void usage(int exitVal, const po::options_description &visible_opts)
{
    cout << "Usage: PeriodicHomogenization_cli [options]" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description analysis_opts("Analysis Settings");
    AnalysisSettings::getOptions(analysis_opts);

    // po::options_description hidden_opts("Hidden Arguments");
    // hidden_opts.add_options()
    //     ("modelFile",  po::value<string>(), "input model (CSG) file")
    //     ("outputFile", po::value<string>(), "output results file")
    //     ;
    // po::positional_options_description p;
    // p.add("modelFile",   1);
    // p.add("bcFile",      1);
    // p.add("outputFile",  1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("settings",      po::value<string>(), "settings file")
        ("msh",           po::value<string>(), ".msh output")
        ("dumpMatrices",                       "Dump matrices for debugging")
        ("settingsName",  po::value<string>(), "settings name")
        ("modelName",     po::value<string>(), "model name")
        ("time",                               "report timings")
        ;
    visible_opts.add(analysis_opts);

    po::options_description cli_opts;
    cli_opts.add(visible_opts); // .add(hidden_opts);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).
                  options(cli_opts).run(), vm);
                  // options(cli_opts).positional(p).run(), vm);
        po::notify(vm);
    }
    catch (std::exception &e) {
        cout << "Error: " << e.what() << endl << endl;
        usage(1, visible_opts);
    }

    if (vm.count("help"))
        usage(0, visible_opts);

    return vm;
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on sucess)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    po::variables_map args = parseCmdLine(argc, argv);

    AnalysisSettings settings;
    string settingsName("Default");
    if (args.count("settings") > 0) {
        string settingsPath = args["settings"].as<string>();
        ifstream settingsFile(settingsPath);
        if (!settingsFile.is_open())
            cout << "Couldn't open settings '" << settingsPath << '\'' << endl;
        else {
            settings.parseOptions(settingsFile);
            boost::filesystem::path spath(settingsPath);
            settingsName = boost::filesystem::basename(spath);
        }
    }

    if (args.count("settingsName"))
        settingsName = args["settingsName"].as<string>();

    bool dumpMatrices = args.count("dumpMatrices");

    SolverLibrary<Scalar> solvers(dumpMatrices);

    // MeshlessFEM_t fem(csgTree, settings, solvers);
    
    return 0;
}
