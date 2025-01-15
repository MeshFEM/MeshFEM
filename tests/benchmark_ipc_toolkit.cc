#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include <ipc/ipc.hpp>
#include <ipc/collisions/collisions.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/potentials/barrier_potential.hpp>

#include <MeshFEM/Parallelism.hh>

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> readMatrix(const char *filename) {
    int cols = 0, rows = 0;
    std::vector<T> buff;

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(filename);
    while (! infile.eof()) {
        std::string line;
        getline(infile, line);

        int temp_cols = 0;
        std::stringstream stream(line);
		T val;
        while((stream >> val)) {
			buff.push_back(val);
			++temp_cols;
		}

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    if (rows * cols != buff.size()) {
        std::cout << "rows: " << rows << ", cols: " << cols << ", buff.size(): " << buff.size() << std::endl;
        throw std::runtime_error("Read error from " + std::string(filename));
    }
	return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(buff.data(), rows, cols);
};

double compute_collision_tightInclusion_stepsize(const ipc::CollisionMesh &cm, const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1, double dhat) {
    std::cout << "dhat: " << dhat << std::endl;
    BENCHMARK_START_TIMER_SECTION("candidates.build");
    auto candidateCache = std::make_unique<ipc::Candidates>();
    ipc::Candidates &candidates = *candidateCache;
    candidates.build(cm, V0, V1, /* inflation_radius = */ dhat / 2, ipc::BroadPhaseMethod::HASH_GRID);
    BENCHMARK_STOP_TIMER_SECTION("candidates.build");

    BENCHMARK_START_TIMER_SECTION("compute_collision_free_stepsize");
    double dmin = 0.0;
    double ccd_tolerance = 2e-8;
    size_t max_iteration = 1e6;
    std::cout << "candidates.compute_collision_free_stepsize with candidate size " << candidates.size() << " and step length " << (V0 - V1).norm() << std::endl;
    double alpha = candidates.compute_collision_free_stepsize(
        cm, V0, V1, /* dmin = */ dmin, /* tolerance = */ ccd_tolerance, /* max_iterations = */ max_iteration);
    BENCHMARK_STOP_TIMER_SECTION("compute_collision_free_stepsize");
    return alpha;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " 2D_debug_ccd_directory" << std::endl;
        return 1;
    }
    std::string inputPath = argv[1];

    set_max_num_tbb_threads(1);

    // Load the CCD input data from the files:
    //      `{inputPath}/cm_edges.txt`
    //      `{inputPath}/cm_faces.txt`
    //      `{inputPath}/debug_ccd_{counter}_x[01].txt`
    Eigen::MatrixXi E = readMatrix<int>((inputPath + "/cm_edges.txt").c_str());
    Eigen::MatrixXi F = readMatrix<int>((inputPath + "/cm_faces.txt").c_str());
    std::vector<Eigen::MatrixXd> x0_inputs, x1_inputs;
    size_t counter = 0;
    while (true) {
        std::string x0Path = inputPath + "/debug_ccd_" + std::to_string(counter) + "_x0.txt";
        std::string x1Path = inputPath + "/debug_ccd_" + std::to_string(counter) + "_x1.txt";

        if (!std::ifstream(x0Path).good()) break;

        x0_inputs.push_back(readMatrix<double>(x0Path.c_str()));
        x1_inputs.push_back(readMatrix<double>(x1Path.c_str()));
        counter++;
    }
    if (counter == 0) {
        std::cerr << "No input files found in " << inputPath << std::endl;
        return 1;
    }

    auto cm = ipc::CollisionMesh(x0_inputs[0], E, F);
    size_t numObstacleVertices = 4;
    size_t obstacleVertexOffset = x0_inputs[0].rows() - numObstacleVertices;
    cm.can_collide = [&](size_t vi, size_t vj) {
        return (vi < obstacleVertexOffset) || (vj < obstacleVertexOffset);
    };

    auto bbox_diag = [](const Eigen::MatrixXd &V) {
        Eigen::RowVector3d min = V.colwise().minCoeff();
        Eigen::RowVector3d max = V.colwise().maxCoeff();
        return (max - min).norm();
    };

    // double dhat = bbox_diag(x0_inputs[0]) * 1e-3;
    double dhat = 6.92820323e-05;

    std::cout.precision(19);
    for (size_t i = 0; i < x0_inputs.size(); ++i) {
        double stepSize = compute_collision_tightInclusion_stepsize(cm, x0_inputs[i], x1_inputs[i], dhat);
        std::cout << "Step size for collision " << i << ": " << stepSize << std::endl;
    }

    BENCHMARK_REPORT();

    return 0;
}
