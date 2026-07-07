////////////////////////////////////////////////////////////////////////////////
// benchmark_dense_linalg.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark low-level dense linear algebra routines.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/28/2025 13:02:01
*///////////////////////////////////////////////////////////////////////////////
#include <MeshFEMCore/Parallelism.hh>
#include <MeshFEMCore/GlobalBenchmark.hh>
#include <catamari/dense_factorizations.hpp>
#include <tbb/tbb.h>

using namespace MeshFEM;

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num_threads>" << std::endl;
        return 1;
    }
    size_t num_threads = std::stoul(argv[1]);
    set_max_num_tbb_threads(num_threads);

#if __linux__
    PinningObserver thread_pinner;
#endif

    int maxSize = 3000;
    int numSizes = 100;
    double scale = pow(maxSize, 1.0 / numSizes);
    std::vector<int> sizes(numSizes + 1);
    sizes[0] = 1;
    {
        double s = 1;
        for (int i = 1; i <= numSizes; ++i) {
            s = s * scale;
            sizes[i] = s;
        }
    }
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    int block_size = 64;
    int tile_size = 128;

    int s_max = sizes.back();
    srand(0);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(s_max, s_max);
    A = (A.transpose() * A).eval();

    tbb::task_group_context tgc;

    // Warm-up and verify
    for (int s : sizes) {
        catamari::BlasMatrixView<double> matrix;
        Eigen::MatrixXd A_ss = A.block(0, 0, s, s);
        matrix.data = A_ss.data();
        matrix.height = s;
        matrix.width = s;
        matrix.leading_dim = s;
        // Print(matrix, "A block", std::cout);
        catamari::Int num_pivots = catamari::LowerCholeskyFactorization(block_size, &matrix);
        if (num_pivots < s) throw std::runtime_error("Non-SPD");
        // Print(matrix, "L", std::cout);

        Eigen::MatrixXd L = A_ss;

        A_ss = A.block(0, 0, s, s);
        matrix.data = A_ss.data();

#if 1
        num_pivots = catamari::CholeskyFlowgraph<double>(tgc, matrix, tile_size, block_size).run(matrix);
        if (num_pivots < s) throw std::runtime_error("Non-SPD TBB");
#else
        catamari::LowerCholeskyFactorizationOpenMP(tile_size, block_size, &matrix);
#endif
        A_ss.triangularView<Eigen::StrictlyUpper>().setZero();
        L.triangularView<Eigen::StrictlyUpper>().setZero();

        double relerr = (A_ss - L).norm() / L.norm();
        if (relerr > 1e-10) {
            std::cerr << "Cholesky factorization relative error: " << relerr << " at size " << s << std::endl;
            // std::cout << A_ss << std::endl << std::endl;
            // std::cout << L << std::endl << std::endl;
            return 1;
        }
    }

    for (int s : sizes) {
        Eigen::MatrixXd A_ss = Eigen::MatrixXd::Identity(s, s);
        catamari::BlasMatrixView<double> matrix;
        matrix.data = A_ss.data();
        matrix.height = s;
        matrix.width = s;
        matrix.leading_dim = s;
        size_t numTrials = 50;
        double time = 0;

        // std::unique_ptr<catamari::CholeskyFlowgraph<double>> flowgraph;

        for (size_t i = 0; i < numTrials; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
#if 1
            // Reusing the flowgraph ends up not being appreciably faster--and
            // appears to cause weird performance regressions at certain thread
            // counts on Apple Silicon...
            //      if (!flowgraph) flowgraph = std::make_unique<catamari::CholeskyFlowgraph<double>>(tgc, matrix, tile_size, block_size);
            //      flowgraph->run(matrix);
            catamari::CholeskyFlowgraph<double>(tgc, matrix, tile_size, block_size).run(matrix);
            // catamari::LowerCholeskyFactorizationOpenMP(tile_size, block_size, &matrix);
#else
            catamari::LowerCholeskyFactorization(block_size, &matrix);
#endif
            auto end = std::chrono::high_resolution_clock::now();
            time += std::chrono::duration<double>(end - start).count();
        }
        std::cout << s << "," << time / numTrials << std::endl;
    }
    return 0;
}
