#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
#include "LinearElasticity.hh"
#include "Materials.hh"
#include "PeriodicHomogenization.hh"
#include "GlobalBenchmark.hh"
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
using namespace PeriodicHomogenization;

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: PeriodicHomogenization_cli [options] mesh" << endl;
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
        ("material,m", po::value<string>(), "base material")
        ("degree,d",   po::value<int>()->default_value(2), "degree of finite elements")
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

    int d = vm["degree"].as<int>();
    if (d < 1 || d > 2) {
        cout << "Error: FEM Degree must be 1 or 2" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N>
using HMG = LinearElasticity::HomogenousMaterialGetter<Materials::Constant>::template Getter<_N>;

template<size_t _N, size_t _FEMDegree>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements) {
    auto &mat = HMG<_N>::material;
    if (args.count("material")) mat.setFromFile(args["material"].as<string>());

    typedef LinearElasticity::Mesh<_N, _FEMDegree, HMG> Mesh;
    typedef LinearElasticity::Simulator<Mesh> Simulator;
    Simulator sim(inElements, inVertices);
    typedef typename Simulator::ETensor ETensor;
    typedef typename Simulator::VField  VField;

    BENCHMARK_START_TIMER_SECTION("Cell Problems");
    std::vector<VField> w_ij;
    solveCellProblems(w_ij, sim);
    BENCHMARK_STOP_TIMER_SECTION("Cell Problems");

    // MSHFieldWriter writer("phomog.msh", sim.mesh());
    // for (size_t i = 0; i < w_ij.size(); ++i) {
    //     VField rhs(sim.constantStrainLoad(-Simulator::SMatrix::CanonicalBasis(i)));
    //     // NOTE: constant strain load on vertex nodes is actually zero in deg 2!
    //     writer.addField("load " + to_string(i), sim.dofToNodeField(rhs), MSHFieldWriter::PER_NODE);
    //     writer.addField("w_ij " + to_string(i), w_ij[i], MSHFieldWriter::PER_NODE);
    // }

    BENCHMARK_START_TIMER_SECTION("Compute Tensor");
    // ETensor Eh = homogenizedElasticityTensor(w_ij, sim);
    ETensor Eh = homogenizedElasticityTensorDisplacementForm(w_ij, sim);
    BENCHMARK_STOP_TIMER_SECTION("Compute Tensor");

    cout << setprecision(16) << endl;
    cout << "Homogenized elasticity tensor:" << endl;
    cout << Eh << endl << endl;

    ETensor S = Eh.inverse();
    cout << "Homogenized compliance tensor:" << endl;
    cout << S << endl;
    vector<Real> moduli(flatLen(_N));

    // Shear moduli are multiplied by 4 in flattened compliance tensor...
    for (size_t i = 0; i < flatLen(_N); ++i)
        moduli[i] = ((i < _N) ? 1.0 : 0.25) / S.D(i, i);

    vector<Real> poisson;
    if (_N == 2) poisson = { -S.D(0, 1) / S.D(1, 1),   // v_yx
                             -S.D(1, 0) / S.D(0, 0) }; // v_xy
    else         poisson = { -S.D(0, 1) / S.D(1, 1),   // v_yx
                             -S.D(0, 2) / S.D(2, 2),   // v_zx
                             -S.D(1, 2) / S.D(2, 2),   // v_zy
                             -S.D(1, 0) / S.D(0, 0),   // v_xy
                             -S.D(2, 0) / S.D(0, 0),   // v_xz
                             -S.D(2, 1) / S.D(1, 1) }; // v_zy

    if (_N == 2)  {
        cout << "Approximate Young moduli:\t"  << moduli[0] << "\t" << moduli[1] << endl;
        cout << "Approximate shear modulus:\t" << moduli[2] << endl;

        cout << "v_yx, v_xy:\t" << poisson[0] << "\t" << poisson[1] << endl;
    }
    else {
        cout << "Approximate Young moduli:\t" << moduli[0] << "\t" << moduli[1] << "\t"
             << moduli[2] << endl;
        cout << "Approximate shear moduli:\t" << moduli[3] << "\t" << moduli[4] << "\t"
             << moduli[5] << endl;

        cout << "v_yx, v_zx, v_zy:\t" << poisson[0] << "\t" << poisson[1] << "\t" << poisson[2] << endl;
        cout << "v_xy, v_xz, v_yz:\t" << poisson[3] << "\t" << poisson[4] << "\t" << poisson[5] << endl;
    }

    cout << "Anisotropy:\t" << Eh.anisotropy() << endl;

    BENCHMARK_REPORT();
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
    else    throw std::runtime_error("Mesh must be triangle or tet.");

    // Look up and run appropriate homogenizer instantiation.
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(args, inVertices, inElements);

    return 0;
}
