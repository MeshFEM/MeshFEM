//
// Created by Davi Colli Tozoni on 7/30/18.
//

#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/LinearElasticityWithContact.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/util.h>
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;
using namespace std;

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: NormalContactForceFunctionTest [options] mesh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
            ("mesh",       po::value<string>(),                     "input mesh")
            ;
    po::positional_options_description p;
    p.add("mesh",                1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
            ("material,m",           po::value<string>()->default_value(""), "simulation material")
            ("boundaryConditions,b", po::value<string>(),                    "boundary conditions")
            ("degree,d",             po::value<int>()->default_value(2),     "FEM degree (1 or 2)")
            ;

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

    bool fail = false;
    if (vm.count("mesh") == 0) {
        cout << "Error: must specify input mesh" << endl;
        fail = true;
    }


    if (vm.count("boundaryConditions") == 0) {
        cout << "Error: must specify boundary conditions" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N, size_t _Deg>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices,
             const vector<MeshIO::IOElement> &inElements) {
    typedef LinearElasticity::Mesh<_N, _Deg> Mesh;
    using Simulator = LinearElasticityWithContact::Simulator<Mesh>;
    Simulator sim(inElements, inVertices);

    const string &materialPath = args["material"].as<string>();

    string bcPath;
    if (args.count("boundaryConditions")) bcPath = args["boundaryConditions"].as<string>();

    // Read homogenous material from .material file (or use default material
    // if no file is given).
    Materials::Constant<_N> mat;
    if (materialPath != "")
        mat.setFromFile(materialPath);
    LinearElasticity::ETensorStoreGetter<_N> store(mat.getTensor());
    for (size_t i = 0; i < sim.mesh().numElements(); ++i)
        sim.mesh().element(i)->configure(store);

    // Apply boundary conditions to simulation
    bool noRigidMotion;
    vector<PeriodicPairDirichletCondition<_N>> pps;
    ComponentMask pinTranslationComponents;
    auto bconds = readBoundaryConditions<_N>(bcPath, sim.mesh().boundingBox(), noRigidMotion, pps, pinTranslationComponents);
    sim.applyBoundaryConditions(bconds);


    // Construct function
    NormalContactForceFunction<Real, Mesh> function(sim.mesh(), 1e-2);

    // Choose u
    std::vector<Real> u(_N * sim.mesh().numNodes(), 0.0);

    // Evaluate function at u
    std::vector<Real> result = function.evaluate(u);

    // Loop through all elements of u, computing finite difference
    Real perturbation = 1e-10;
    TripletMatrix<Triplet<Real>> approximatedJacobian(result.size(), _N * sim.mesh().numNodes());
    for (unsigned j = 0; j < u.size(); j++) {

        // Compute perturbed u
        std::vector<Real> perturbedU = u;
        perturbedU[j] = u[j] + perturbation;

        // Evaluate normal force on perturbed u
        std::vector<Real> perturbedResult = function.evaluate(perturbedU);

        for (unsigned i = 0; i < result.size(); i++) {
            Real relativeDifference = (perturbedResult[i] - result[i]) / perturbation;

            if (abs(relativeDifference) > 1e-15) {
                //std::cout << "result = " << result[i] << std::endl;
                //std::cout << "perturbed = " << perturbedResult[i] << std::endl;
                std::cout << "Relative difference = " << relativeDifference << std::endl;
                approximatedJacobian.addNZ(i, j, relativeDifference);
            }
        }
    }

    size_t approximatedJacobianSize = approximatedJacobian.nz.size();

    // Add each element of negative approximated Jacobian to actual jacobian matrix. And sum repeated.
    // Result should be empty matrix
    TripletMatrix<Triplet<Real>> jacobian = function.jacobian(u);
    size_t jacobianSize = jacobian.nz.size();

    for (auto t : approximatedJacobian.nz) {
        jacobian.addNZ(t.i, t.j, -t.v);
    }
    jacobian.sumRepeated();

    // Verify that all entries are super small or zero
    int errorCount = 0;
    for (auto t : jacobian.nz) {
        if (abs(t.v) > 1e-6) {
            std::cout << "[Warning!!] Element (" << t.i << ", " << t.j << ") differs in jacobian by " << t.v
                      << std::endl;
            errorCount++;
        }
    }

    std::cout << "Number of nz elements in Jacobian: " << jacobianSize << std::endl;
    std::cout << "Number of nz elements in approximate Jacobian: " << approximatedJacobianSize << std::endl;
    std::cout << "Error count: " << errorCount << std::endl;

    return;
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    po::variables_map args = parseCmdLine(argc, argv);

    vector<MeshIO::IOVertex>  inVertices;
    vector<MeshIO::IOElement> inElements;
    string meshPath = args["mesh"].as<string>();

    auto type = load(meshPath, inVertices, inElements, MeshIO::FMT_GUESS, MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    // Look up and run appropriate simulation instantiation.
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(args, inVertices, inElements);

    return 0;
}
