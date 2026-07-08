////////////////////////////////////////////////////////////////////////////////
// test_ipc_simulation.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A C++ version of the IPC simulations set up in `python/demos/IPCSimulation`.
//
//  This is especially helpful for diagnosing crashes in a simpler setting than
//  with the Python bindings. Also, the simulation results are compared against
//  "ground truth" values stored within the repository.
*///////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/ElasticSolid.hh>
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/IPCIntegration/IPCObjectiveTerm.hh>
#include <MeshFEM/DynamicSimulator.hh>
#include <MeshFEMCore/GlobalBenchmark.hh>

// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>

#include <Eigen/Dense>

// For reading compressed data files...
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/lzma.hpp>
#include <boost/iostreams/device/file.hpp>
#include <MeshFEMCore/Utilities/load_dense_matrix.hh>

// Relative tolerance for comparing the simulation result to the ground truth.
// (Note that parallelism-induced rounding differences can accumulate over time,
// so we need a relatively loose tolerance.)
constexpr double SOLUTION_TOLERANCE = 1e-10;

using namespace MeshFEM;
namespace bio = boost::iostreams;

// Equivalent of sim_utils.getBBoxVars(es, BBoxFace.MIN_Y, tol=GROUND_TOL)
template<class Mesh>
std::vector<size_t> getBBoxVarsMinY(const Mesh &mesh, double tol = 1e-8) {
    constexpr size_t dim = Mesh::EmbeddingSpace::RowsAtCompileTime;
    double minY = std::numeric_limits<double>::max();
    for (auto n : mesh.nodes())
        minY = std::min(minY, n->p[1]);

    std::vector<size_t> varIdxs;
    for (auto n : mesh.nodes()) {
        if (std::abs(n->p[1] - minY) < tol) {
            for (size_t c = 0; c < dim; ++c)
                varIdxs.push_back(dim * n.index() + c);
        }
    }
    return varIdxs;
}

template<size_t N, size_t  Deg>
void run_test() {
    using VNd    = VecN_T<double, N>;
    using Mesh   = FEMMesh<N, Deg, VNd>;
    using Energy = CommonNeoHookeanEnergy<double, N>;
    using ES     = ElasticSolid<N, Deg, VNd, Energy>;

    const std::string path = std::string(MESHFEM_DIR) + "/misc/examples/meshes/"
        + ((N == 2) ? "square_hole_subdiv.msh" : "bunny_coarse.msh");
    
    constexpr double RHO        = (N == 2) ? 4e-2 : 1e-4;
    constexpr double GROUND_TOL = (N == 2) ? 1e-8 : 4;
    constexpr double T          = (N == 2) ? 2.0  : 10;
    constexpr double dt         = (N == 2) ? 0.001 : 0.1;

    constexpr double E  = 2.0;
    constexpr double nu = 0.4;

    auto lambdaFromENu = [](double E, double nu, bool is3D = true) { return is3D ? (E * nu / ((1 + nu) * (1 - 2 * nu))) : ((nu * E) / (1.0 - nu * nu)); };
    auto     muFromENu = [](double E, double nu)                   { return E / (2 * (1 + nu)); };

    const double lambda = lambdaFromENu(E, nu, (N == 3));
    const double mu     =     muFromENu(E, nu);

    auto m = std::shared_ptr<Mesh>(Mesh::load(path));

    auto es = std::make_shared<ES>(Energy(lambda, mu), m);
    es->setMassDensity(RHO);

    auto contact = std::make_shared<IPCObjectiveTerm<double>>(es, es->getCollisionMesh());
    contact->sparsityPatternUpdateThreshold = 10;

    DynamicSimulator<double>::Terms terms;
    terms.emplace_back(std::make_shared<Loads::Gravity<ES>>(es));
    terms.emplace_back(contact);

    DynamicSimulator<double> ds(es, terms, /*useLumpedMass = */ true, dt);

    ds.setFixedVars(getBBoxVarsMinY(*m, GROUND_TOL));

    ds.method = TimesteppingMethod::BackwardEuler;
    ds.getOptimizer().options.verbose = 0;

    BENCHMARK_RESET();
    ds.run(0.0, T);
    BENCHMARK_REPORT();

    bio::filtering_istream in;
    in.push(bio::lzma_decompressor());
    const std::string ground_truth_file = std::string(MESHFEM_DIR) + "/tests/data/ipc_sim_result_" + std::to_string(N) + "_" + std::to_string(Deg) + ".txt.xz";
    in.push(bio::file_source(ground_truth_file, std::ios_base::binary));
    auto x_gt = load_matrix_from_stream<double>(in, es->numVars(), 1);
    REQUIRE((es->getVars() - x_gt).norm() < SOLUTION_TOLERANCE * x_gt.norm());
}

TEST_CASE("IPC Simulation", "[ipc_simulation]" ) {
    SECTION("2D, Deg 1") { run_test<2, 1>(); }
    SECTION("2D, Deg 2") { run_test<2, 2>(); }
    SECTION("3D, Deg 1") { run_test<3, 1>(); }
}

TEST_CASE("IPC Simulation Slow (Skipped by Default)", "[.][ipc_simulation_slow]" ) {
    SECTION("3D, Deg 2") { run_test<3, 2>(); }
}
