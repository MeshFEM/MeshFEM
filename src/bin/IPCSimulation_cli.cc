#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/ElasticSolid.hh>
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/IPCIntegration/IPCObjectiveTerm.hh>
#include <MeshFEM/DynamicSimulator.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

// 2D, degree-2 triangle mesh types (mirrors DIM=2, DEG=2 in the Python script)
using Vec2d  = Eigen::Matrix<double, 2, 1>;
using Mesh   = FEMMesh<2, 2, Vec2d>;
using Energy = CommonNeoHookeanEnergy<double, 2>;
using ES     = ElasticSolid<2, 2, Vec2d, Energy>;

// Equivalent of sim_utils.getBBoxVars(es, BBoxFace.MIN_Y, tol=GROUND_TOL)
std::vector<size_t> getBBoxVarsMinY(const Mesh &mesh, double tol = 1e-8) {
    constexpr size_t dim = 2;
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

int main() {
    const std::string PATH  = "/home/hmohammadian/workspace/github_repos/MeshFEM_dev/misc/examples/meshes/square_hole_subdiv.msh";
    
    constexpr double RHO        = 4e-2;
    constexpr double GROUND_TOL = 1e-8;
    constexpr double T          = 2.0;
    constexpr double dt         = 0.001;

    // Lamé parameters from Young's modulus E=2, Poisson's ratio nu=0.4 (2D formula)
    constexpr double E      = 2.0;
    constexpr double nu     = 0.4;
    const double     lambda = nu * E / (1.0 - nu * nu);
    const double     mu     = E / (2.0 * (1.0 + nu));

    auto m = std::shared_ptr<Mesh>(Mesh::load(PATH));

    Energy energy(lambda, mu);
    auto es = std::make_shared<ES>(energy, m);
    es->setMassDensity(RHO);

    auto g       = std::make_shared<Loads::Gravity<ES>>(es);
    auto eo      = std::static_pointer_cast<ElasticObject<double>>(es);
    auto contact = std::make_shared<IPCObjectiveTerm<double>>(es, eo->getCollisionMesh());

    std::vector<std::shared_ptr<NewtonObjectiveTermBase>> terms = { g, contact };
    DynamicSimulator<double> ds(es, terms, /*useLumpedMass=*/true, dt);

    ds.setFixedVars(getBBoxVarsMinY(*m, GROUND_TOL));

    contact->sparsityPatternUpdateThreshold = 10;
    ds.method = TimesteppingMethod::BackwardEuler;
    ds.getOptimizer().options.verbose = 0;

    BENCHMARK_RESET();
    ds.run(0.0, T);
    std::cout << "Simulation completed successfully." << std::endl;
    BENCHMARK_REPORT();

    return 0;
}
