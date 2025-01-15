#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include <ipc/ipc.hpp>
#include <ipc/collisions/collisions.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/potentials/barrier_potential.hpp>

double compute_collision_tightInclusion_stepsize(const ipc::CollisionMesh &cm, const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1, double dhat) {
    BENCHMARK_START_TIMER_SECTION("candidates.build");
    auto candidateCache = std::make_unique<ipc::Candidates>();
    ipc::Candidates &candidates = *candidateCache;
    candidates.build(
        cm,
        V0,
        V1,
        /* inflation_radius = */ dhat / 2, ipc::BroadPhaseMethod::HASH_GRID);
    BENCHMARK_STOP_TIMER_SECTION("candidates.build");

    double dmin = 0.0;
    double ccd_tolerance = 2e-8;
    size_t max_iteration = 1e6;
    double alpha = candidates.compute_collision_free_stepsize(
        cm, V0, V1, /* dmin = */ dmin, /* tolerance = */ ccd_tolerance, /* max_iterations = */ max_iteration);
    return alpha;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " 2D_debug_ccd_directory" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    Eigen::MatrixXi E, F;
    std::vector<Eigen::MatrixXd> x0_inputs, x1_inputs;

    // Load the CCD input meshes from `{inputPath}/debug_ccd_{counter}_0.obj`
    // and `{inputPath}/debug_ccd_{counter}_1.obj`.
    // Currently we only support the CCD output for 2D simulations, since 3D
    // would require loading both edges and faces.
    size_t counter = 0;
    while (true) {
        std::string x0Path = inputPath + "/debug_ccd_" + std::to_string(counter) + "_0.obj";
        std::string x1Path = inputPath + "/debug_ccd_" + std::to_string(counter) + "_1.obj";
        std::vector<MeshIO::IOVertex> vertices;
        std::vector<MeshIO::IOElement> elements;

        try {
            MeshIO::load(x0Path, vertices, elements);
            x0_inputs.push_back(getV(vertices));
            if (E.size() == 0) E = getF(elements);
            if (E != getF(elements)) {
                std::cerr << "Mesh connectivity mismatch" << std::endl;
                return 1;
            }

            MeshIO::load(x1Path, vertices, elements);
            x1_inputs.push_back(getV(vertices));
            if (E != getF(elements)) {
                std::cerr << "Mesh connectivity mismatch" << std::endl;
                return 1;
            }
            counter++;
        }
        catch (...) { break; }
    }

    auto cm = ipc::CollisionMesh(x0_inputs[0], E, F);

    auto bbox_diag = [](const Eigen::MatrixXd &V) {
        Eigen::RowVector3d min = V.colwise().minCoeff();
        Eigen::RowVector3d max = V.colwise().maxCoeff();
        return (max - min).norm();
    };

    double dhat = bbox_diag(x0_inputs[0]) * 1e-3;

    std::cout.precision(19);
    for (size_t i = 0; i < x0_inputs.size(); ++i) {
        double stepSize = compute_collision_tightInclusion_stepsize(cm, x0_inputs[i], x1_inputs[i], dhat);
        std::cout << "Step size for collision " << i << ": " << stepSize << std::endl;
    }

    return 0;
}
