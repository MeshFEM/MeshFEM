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
#include "LevelSet.hh"
#include "CSGTree.hh"
#include "CSGFile.hh"
#include "MSHWriter.hh"
#include "WireNetwork.hh"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iomanip>

namespace po = boost::program_options;
using namespace std;

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: PeriodicHomogenization_cli modelFile [options]" << endl;
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
        ;
    po::positional_options_description p;
    p.add("modelFile",   1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("settings",      po::value<string>(), "settings file")
        ("msh",           po::value<string>(), ".msh output")
        ("dumpMatrices",                       "Dump matrices for debugging")
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

    if ((vm.count("modelFile") == 0)) {
        cout << "Error: must specify input file" << endl;
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
    if (args.count("settings") > 0) {
        string settingsPath = args["settings"].as<string>();
        ifstream settingsFile(settingsPath);
        if (!settingsFile.is_open())
            cout << "Couldn't open settings '" << settingsPath << '\'' << endl;
        else {
            // Merge in settings file's options
            // Already specified options read from command line remain unchanged
            // (so they override the settings file)
            po::options_description opts;
            AnalysisSettings::getOptions(opts);
            po::store(po::parse_config_file(settingsFile, opts), args);
        }
    }

    settings.readOptions(args);

    Timer *timer = NULL;
    if (args.count("time"))
        timer = new Timer();

    bool dumpMatrices = args.count("dumpMatrices");

    SolverLibrary<Scalar> solvers(dumpMatrices);

    // SchwarzP<Vector> schwarzP(BBox_t(M_PI * Vector(0.0, 0.0, 0.0),
    //                                  M_PI * Vector(2.0, 2.0, 2.0)));
    // SchwarzP<Vector> model(BBox_t(M_PI * Vector(-1.0, -1.0, -1.0),
    //                               M_PI * Vector( 1.0,  1.0,  1.0)));
    // Sphere<Vector> sphere(BBox_t(Vector(-1.0, -1.0, -1.0),
    //                              Vector( 1.0,  1.0,  1.0)),
    //                       Vector(0.0, 0.0, 0.0), 2.0);
    // WireNetwork<Vector> model(BBox_t(Vector(-5.0, -5.0, -5.0), Vector(5.0, 5.0, 5.0)), 
    //             "examples/wires/star.wire", 1.0);
    // WireNetwork<Vector> model(
    //         BBox_t(Vector(0.0, 0.0, 0.0), Vector(10.0, 10.0, 10.0)),
    //         "examples/wires/brick5.wire",
    //         0.5);
    // string modelPath = args["modelFile"].as<string>();
    CSGTree_t model;
    parseCSGFile(args["modelFile"].as<string>(), model);

    // typedef MeshlessFEM3D<LevelSet_t> MeshlessFEM3D_t;
    typedef MeshlessFEM3D<CSGTree_t> MeshlessFEM3D_t;
    if (timer) timer->start("Setup");
    MeshlessFEM3D_t fem(model, settings, solvers);
    if (timer) timer->stop("Setup");

    // cout << fem.getElasticityTensor();
    // cout << endl;

    typedef MSHWriter<MeshlessFEM3D_t::ElementGrid> MSHWriter_t;
    MSHWriter_t *msh = NULL;
    if (args.count("msh")) {
        msh = new MSHWriter_t(args["msh"].as<string>(), fem.elementGrid());

        MeshlessFEM3D_t::SField overlaps(fem.elementGrid().numElements());
        for (size_t e = 0; e < fem.elementGrid().numElements(); ++e)
            overlaps[e] = fem.elementGrid().elementOverlap(e);
        msh->addField("cellOverlaps", overlaps, MSHWriter_t::PER_ELEMENT);

        MeshlessFEM3D_t::SField signedDistances(fem.elementGrid().numNodes());
        for (size_t n = 0; n < fem.elementGrid().numNodes(); ++n)
            signedDistances[n] = model.signedDistance(fem.elementGrid().nodePosition(n));
        msh->addField("signedDistances", signedDistances, MSHWriter_t::PER_NODE);
    }

    if (timer) timer->startSection("Periodic Homogenization");
    MeshlessFEM3D_t::ETensor Eh = fem.periodicHomogenize(timer, msh);
    if (timer) timer->stopSection("Periodic Homogenization");

    cout << setprecision(16);
    cout << "Homogenized elasticity tensor:" << endl;
    cout << Eh << endl << endl;;

    MeshlessFEM3D_t::ETensor::DType Dinv = Eh.D().inverse();

    cout << "Homogenized compliance tensor:" << endl;
    cout << Dinv << endl << endl;
    Eigen::Matrix<Scalar, 6, 1> moduli(1.0 / Dinv.diagonal().array());
    std::cout << "Approximate Young moduli:\t" << moduli[0] << "\t" << moduli[1] << "\t"
              << moduli[2] << endl;
    std::cout << "Approximate shear moduli:\t" << moduli[3] << "\t" << moduli[4] << "\t"
              << moduli[5] << endl;
    if (timer) timer->report(cout);
    
    return 0;
}
