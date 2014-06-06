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
#include <vector>

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
        ("parameterStep", po::value<double>()->default_value(0.1),
                "(fractional) ammount by which to attempt to change elastic coefficients")
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
            // Merge settings file's contents into args.
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
    CSGTree_t model;
    parseCSGFile(args["modelFile"].as<string>(), model);

    // typedef MeshlessFEM3D<LevelSet_t> MeshlessFEM3D_t;
    typedef MeshlessFEM3D<CSGTree_t> MeshlessFEM3D_t;
    if (timer) timer->start("Setup");
    MeshlessFEM3D_t fem(model, settings, solvers);
    if (timer) timer->stop("Setup");

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

    cout << setprecision(16);
    if (timer) timer->startSection("Periodic Homogenization");
    vector<MeshlessFEM3D_t::VField> w_ij;
    std::cout << "Running homogenization on "
              << settings.Int("Nz") << " x "
              << settings.Int("Ny") << " x "
              << settings.Int("Nx") << " grid with "
              << fem.elementGrid().numElements()  << " elements and "
              << fem.elementGrid().numNodes()  << " nodes (qp = "
              << settings.Int("quadraturePoints") << ")" << std::endl;
    fem.solveCellProblems(w_ij, timer, msh);
    MeshlessFEM3D_t::ETensor Eh =
                fem.homogenizedElasticityTensor(w_ij, timer, msh);
    MeshlessFEM3D_t::ETensor ETargetinv(Eh.inverse());
    // // Try to double x Young's modulus
    // ETargetinv.D(0, 0) /= 2.0;
    // ETargetinv.D(0, 1) /= 2.0;
    // ETargetinv.D(0, 2) /= 2.0;
    
    // Try to reduce all Poisson ratios
    Scalar parameterStep = args["parameterStep"].as<double>();
    cout << "Parameter step: " << parameterStep << endl;
    Scalar currentPoisson = -ETargetinv.D(0, 1) / ETargetinv.D(1, 1);
    Scalar targetPoisson = currentPoisson + parameterStep * (-0.5 - currentPoisson);
    cout << "currentPoisson, targetPoisson:\t" << currentPoisson << "\t"
         << targetPoisson << endl;

    ETargetinv.D(0, 1) = -targetPoisson * ETargetinv.D(1, 1);
    ETargetinv.D(0, 2) = -targetPoisson * ETargetinv.D(2, 2);
    ETargetinv.D(1, 2) = -targetPoisson * ETargetinv.D(2, 2);

    // // Try to double all Young's moduli
    // ETargetinv.D(0, 0) /= 2.0;
    // ETargetinv.D(1, 1) /= 2.0;
    // ETargetinv.D(2, 2) /= 2.0;
    // ETargetinv.D(0, 1) /= 2.0;
    // ETargetinv.D(0, 2) /= 2.0;
    // ETargetinv.D(1, 2) /= 2.0;

    fem.homogenizedElasticityTensorShapeDerivative(ETargetinv.inverse(), w_ij, timer, msh);
    
    if (timer) timer->stopSection("Periodic Homogenization");

    cout << "Homogenized elasticity tensor:" << endl;
    cout << Eh << endl << endl;;

    cout << "Tensor Diff:" << endl << Eh - ETargetinv.inverse() << endl << endl;;

    MeshlessFEM3D_t::ETensor Einv = Eh.inverse();

    cout << "Homogenized compliance tensor:" << endl;
    cout << Einv << endl << endl;
    Eigen::Matrix<Scalar, 6, 1> moduli(1.0 / Einv.diag().array());
    cout << "Approximate Young moduli:\t" << moduli[0] << "\t" << moduli[1] << "\t"
         << moduli[2] << endl;
    cout << "Approximate shear moduli:\t" << moduli[3] << "\t" << moduli[4] << "\t"
         << moduli[5] << endl;

    cout << "v_yx, v_zx, v_zy:\t" << -Einv.D(0, 1) / Einv.D(1, 1) << "\t"
                                  << -Einv.D(0, 2) / Einv.D(2, 2) << "\t"
                                  << -Einv.D(1, 2) / Einv.D(2, 2) << endl;
    cout << "v_xy, v_xz, v_yz:\t" << -Einv.D(1, 0) / Einv.D(0, 0) << "\t"
                                  << -Einv.D(2, 0) / Einv.D(0, 0) << "\t"
                                  << -Einv.D(2, 1) / Einv.D(1, 1) << endl;

    if (timer) timer->report(cout);
    
    return 0;
}
