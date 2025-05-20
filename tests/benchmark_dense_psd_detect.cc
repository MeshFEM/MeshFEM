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
#include <catamari.hpp>
#include <Eigen/Dense>
#include <MeshFEM/Utilities/DensePSDDetect.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>

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
    run<6>();  // 2D degree 1 triangles
    run<18>(); // 2D degree 2 triangles
    run<12>(); // 3D degree 1 tetrahedra
    run<30>(); // 3D degree 2 tetrahedra

    using Psi = NeoHookeanEnergy<double, 3>;
    using Psi_ap = AutoHessianProjection<Psi>;
    using Psi_ap2 = AutoHessianProjection<Psi_ap>;

    Psi_ap psi_ap;
    // Psi_ap2 psi_ap2;

    BENCHMARK_REPORT();
    return 0;
}
