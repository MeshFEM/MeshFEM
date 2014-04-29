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
#include "SolverLibrary.hh"
#include "GlobalTypes.hh"
#include "MeshlessFEM2D.hh"
#include "ResultsCollector.hh"
#include "CSGFile.hh"
#include "BoundaryConditions.hh"
#ifdef HAS_MATLAB
#include "LazyMatlabInterfaces.hh"
#endif
#include "MSHWriter.hh"
#include "Timer.hh"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
namespace po = boost::program_options;
using namespace std;

void usage(int exitVal, const po::options_description &visible_opts)
{
    cout << "Usage: CSGFEM_cli model.csg bcond.bc output.res [options]" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description analysis_opts("Analysis Settings");
    AnalysisSettings::getOptions(analysis_opts);

    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("modelFile",  po::value<string>(), "input model (CSG) file")
        ("bcFile",     po::value<string>(), "boundary conditions file")
        ("outputFile", po::value<string>(), "output results file")
        ;

    po::positional_options_description p;
    p.add("modelFile",   1);
    p.add("bcFile",      1);
    p.add("outputFile",  1);

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

    if ((vm.count("modelFile") == 0) || (vm.count("outputFile") == 0)) {
        cout << "Error: must specify input and output files" << endl;
        usage(1, visible_opts);
    }

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
    po::variables_map vm = parseCmdLine(argc, argv);

    AnalysisSettings settings;
    string settingsName("Default");
    if (vm.count("settings") > 0) {
        string settingsPath = vm["settings"].as<string>();
        ifstream settingsFile(settingsPath);
        if (!settingsFile.is_open())
            cout << "Couldn't open settings '" << settingsPath << '\'' << endl;
        else {
            settings.parseOptions(settingsFile);
            boost::filesystem::path spath(settingsPath);
            settingsName = boost::filesystem::basename(spath);
        }
    }

    if (vm.count("settingsName"))
        settingsName = vm["settingsName"].as<string>();

    bool dumpMatrices = vm.count("dumpMatrices");

#ifdef HAS_MATLAB
    LazyQMatlab matlab;
    SolverLibrary<Scalar> solvers(matlab, dumpMatrices);
#else
    SolverLibrary<Scalar> solvers(dumpMatrices);
#endif

    CSGTree_t csgTree;
    string modelPath = vm["modelFile"].as<string>();
    boost::filesystem::path mpath(modelPath);
    string modelName = boost::filesystem::basename(mpath);

    Timer *timer = NULL;
    if (vm.count("time"))
        timer = new Timer();
    
    if (vm.count("modelName"))
        modelName = vm["modelName"].as<string>();

    if (timer) timer->startSection("Setup");
    if (timer) timer->start("parseCSGFile");
    parseCSGFile(modelPath.c_str(), csgTree);
    if (timer) timer->stop("parseCSGFile");

    if (timer) timer->start("FEM Grid");
    MeshlessFEM_t fem(csgTree, settings, solvers);
    if (timer) timer->stop("FEM Grid");

    if (timer) timer->start("readConditions");
    string bcPath = vm["bcFile"].as<string>();
    fem.boundaryConditions().readConditions(bcPath);
    if (timer) timer->stop("readConditions");

    if (timer) timer->start("Prepare Results Collector");
    ResultsCollector_t rc;
    rc.addSettings(settingsName, settings);
    rc.addModel(modelName, csgTree, csgTree.boundingBox());
    if (timer) timer->stop("Prepare Results Collector");
    if (timer) timer->stopSection("Setup");

    fem.simulate(&rc, timer);

    if (timer) timer->startSection("Output");
    if (timer) timer->start(".res");
    string lastResultPath = rc.lastResultPath();
    string outPath = vm["outputFile"].as<string>();
    rc.writeResult(lastResultPath, outPath);
    if (timer) timer->stop(".res");

    if (vm.count("msh")) {
        if (timer) timer->start(".msh");
        string path = vm["msh"].as<string>();
        MSHWriter<MeshlessFEM_t::ElementGrid> writer(path, fem.elementGrid());
        ResultsCollector_t::ConstRPtr r = rc.getResultWithPath(lastResultPath);
        r->addToMSH(writer, "Simulation");
        if (timer) timer->stop(".msh");
    }

    if (timer) timer->stopSection("Output");
    if (timer) timer->report(cout);

    return 0;
}
