#include <iostream>
#include <MeshFEM/Poisson.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/MeshIO.hh>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

[[ noreturn ]] void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: Simulate_cli [options] mesh" << endl;
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
        ("boundaryConditions,b", po::value<string>(),                    "boundary conditions")
        ("outputMSH,o",          po::value<string>(),                    "output mesh")
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
    if (vm.count("outputMSH") == 0) {
        cout << "Error: must specify output msh file" << endl;
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
    PoissonMesh<_N, _Deg, VectorND<_N>> poissonMesh(inElements, inVertices);
    bool dummy;
    auto bconds = readBoundaryConditions<_N>(args["boundaryConditions"].as<string>(),
                                             poissonMesh.boundingBox(), dummy);
    poissonMesh.applyBoundaryConditions(bconds);

    std::vector<Real> x;
    poissonMesh.solve(x);
    MSHFieldWriter writer(args["outputMSH"].as<string>(), poissonMesh);

    ScalarField<Real> u(poissonMesh.numNodes());
    assert(u.domainSize() == x.size());
    for (size_t i = 0; i < x.size(); ++i) u[i] = x[i];
    writer.addField("u", u, DomainType::PER_NODE);

    VectorField<Real, _N> gradU(poissonMesh.numElements());
    auto grads = poissonMesh.gradUAverage(x);
    assert(gradU.domainSize() == grads.size());
    for (size_t i = 0; i < grads.size(); ++i)
        gradU(i) = grads[i];
    writer.addField("grad u", gradU, DomainType::PER_ELEMENT);
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

    auto type = load(meshPath, inVertices, inElements, MeshIO::FMT_GUESS,
                     MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    // Look up and run appropriate Poisson instantiation
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(args, inVertices, inElements);

    return 0;
}
