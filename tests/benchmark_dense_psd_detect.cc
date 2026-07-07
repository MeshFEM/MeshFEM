////////////////////////////////////////////////////////////////////////////////
// benchmark_dense_psd_detect.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark various strategies for quickly checking/ruling out if a matrix
//  is indefinite (and thus needs a Hessian projection).
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  05/19/2025 16:37:32
*///////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <MeshFEM/Utilities/DensePSDDetect.hh>
#include <MeshFEMCore/Parallelism.hh>

using namespace MeshFEM;

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <tbb/tbb.h>

// Helper to fill a symmetric matrix
void fill_symmetric_matrix(double* a, int N, int id) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            a[i + j*N] = a[j + i*N] = std::sin((id + 1) * (i + 1) * (j + 1));
        }
    }
}

void test_dsyevd() {
    static constexpr int N = 12; // Matrix size
    const int num_tasks = 1000;

    std::vector<DenseEighRealSolver<N>> solver_for_thread(get_max_num_tbb_threads());

    // Parallel computation
    tbb::parallel_for(0, num_tasks, [&](int i) {
        std::vector<double> a(N * N);

        fill_symmetric_matrix(a.data(), N, i);

        auto &solver = solver_for_thread[tbb::this_task_arena::current_thread_index()];

        solver.compute(Eigen::Map<Eigen::Matrix<double, N, N>>(a.data()));

        if (i % 200 == 0) {
            Eigen::Matrix<double, N, N> A_eigen;
            A_eigen = Eigen::Map<Eigen::Matrix<double, N, N>>(a.data());

            Eigen::SelfAdjointEigenSolver<decltype(A_eigen)> Hes(A_eigen);
            std::cout << "Matrix " << i
                      << " first eigenvalue: " << solver.eigenvalues()[0]
                      << ", " << Hes.eigenvalues()[0] << "\n";
            std::cout << "Eigenvectors from Lapack:" << std::endl;
            std::cout << solver.eigenvectors() << std::endl << std::endl;
            std::cout << "Eigenvectors from Eigen:" << std::endl;
            std::cout << Hes.eigenvectors() << std::endl << std::endl;
        }
    });

    std::cout << "Done.\n";
}


template<size_t N>
void run() {
    using Matrix = Eigen::Matrix<double, N, N>;
    using Vector = Eigen::Matrix<double, N, 1>;

    // Generate a symmetric matrix with random eigenvalues
    Vector eigs = Vector::Random(N); // [-1, 1]
    eigs.array() += 1.1; // Shift to [0.1, 2.1]

    // Randomly set one of the eigenvalues to be negative
    int idx = rand() % (2 * N);
    if (idx < N) eigs[idx] *= -1;

    // QR decomposition to get an orthogonal matrix
    Eigen::HouseholderQR<Matrix> qr(Matrix::Random(N, N));
    Matrix Q = qr.householderQ();
    Matrix A = Q * eigs.asDiagonal() * Q.transpose();

    // Check if `A` is positive semidefinite
    PSDResult gershgorinResult = isPSDGershgorin(A);
    bool choleskyResult = isPSDCholesky(A);

    bool eigenDecompResult = isPSDEigenDecomp(A);

    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Gershgorin result: " << static_cast<int>(gershgorinResult) << std::endl;
    std::cout << "Cholesky result: " << choleskyResult << std::endl;
    std::cout << "eigenDecompResult: " << eigenDecompResult << std::endl;

    const size_t numTests = 10000;
    {
        size_t numInconclusive = 0;
        BENCHMARK_SCOPED_TIMER_SECTION timer("Gershgorin " + std::to_string(N));
        for (size_t i = 0; i < numTests; ++i) {
            PSDResult result = isPSDGershgorin(A);
            numInconclusive += (result == PSDResult::Maybe);
        }
        std::cout << "Gershgorin inconclusive: " << numInconclusive << std::endl;
    }

    {
        size_t numPSD = 0;
        BENCHMARK_SCOPED_TIMER_SECTION timer("Cholesky " + std::to_string(N));
        for (size_t i = 0; i < numTests; ++i) {
            numPSD += isPSDCholesky(A);
        }
        std::cout << "Cholesky numPSD: " << numPSD << std::endl;
    }

    {
        size_t numPSD = 0;
        BENCHMARK_SCOPED_TIMER_SECTION timer("EigenDecomp " + std::to_string(N));
        for (size_t i = 0; i < numTests; ++i) {
            numPSD += isPSDEigenDecomp(A);
        }
        std::cout << "EigenDecomp numPSD: " << numPSD << std::endl;
    }
}

int main(int argc, const char *argv[]) {
    set_max_num_tbb_threads(1);
    test_dsyevd();

    run<6>();  // 2D degree 1 triangles
    run<18>(); // 2D degree 2 triangles
    run<12>(); // 3D degree 1 tetrahedra
    run<30>(); // 3D degree 2 tetrahedra

    BENCHMARK_REPORT();
    return 0;
}
