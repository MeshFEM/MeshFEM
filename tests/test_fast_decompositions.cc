////////////////////////////////////////////////////////////////////////////////
// test_fast_decompositions.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Rather exhaustive tests for our fast 3x3 and 2x2 matrix decompositions.
//  Confirms properties like factor orthogonality and evaluates the
//  backward error when operating on a large set of uniform-random matrices
//  as well as matrices generated to be near-degenerate (e.g., with
//  eigenvalues or singular values clustered close together or close to zero).
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  08/28/2025 13:35:07
*///////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/Utilities/fast_3x3_decompositions.hh>
#include <MeshFEM/Utilities/fast_2x2_decompositions.hh>
#include <MeshFEM/Utilities/fast_3x2_decompositions.hh>

// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>

#include <iostream>

using namespace MeshFEM;

// Test the cubic polynomial solver in the case of three real roots
// (used for the polar decomposition code).
void test_cubicroot() {
    static constexpr size_t numTests = 1000;
    for (size_t i = 0; i < numTests; ++i) {
        Eigen::Vector3d roots = Eigen::Vector3d::Random();

        // Sort roots by absolute value in descending order
        std::sort(roots.data(), roots.data() + 3, [](double a, double b) { return std::abs(a) > std::abs(b); });

        // prod_i (x - r_i) = x^3 + a*x^2 + b*x + c
        Real a = -roots.sum();
        Real b = roots[0] * roots[1] + roots[0] * roots[2] + roots[1] * roots[2];
        Real c = -roots.prod();
        Real x = fast_decompositions::cubic_max_abs_root(a, b, c);

        // Check if the roots are real
        REQUIRE(std::abs(x - roots[0]) < 1e-10 * std::abs(roots[0]));
    }
}

template<size_t N>
void test_polar() {
    using MNd = MatN_T<double, N>;
    static constexpr size_t numTests = 1000;
    for (size_t i = 0; i < numTests; ++i) {
        MNd A = MNd::Random();
        MNd R, S;
        fast_decompositions::polar(A, R, S);

        REQUIRE((R.transpose() * R - MNd::Identity()).norm() < 1e-10);
        REQUIRE((R * S - A).norm() / A.norm() < 1e-10); // Check backward error.
        REQUIRE(R.determinant() > 0); // Check that R is a proper rotation (not a reflection).
    }
}

template<typename T>
T random_logspaced(T low, T high) {
    double alpha = (double)rand() / RAND_MAX;
    return low * std::pow(high / low, alpha);
}

template<size_t N>
void test_eigs() {
    using MNd = MatN_T<double, N>;
    using VNd = VecN_T<double, N>;
    const double tol = (N == 3) ? 1e-11 : 1e-8;

    // // Specifically problematic inputs we've discovered (for debugging).
    // {
    //     MNd A;
    //     A <<  1.04521642853942182e-13,  2.6407339355839125e-30, -7.49858870371328175e-30,
    //          -5.19735587091900743e-30, 1.04521642853942145e-13,  8.97109828726837732e-31,
    //                                 0,                       0,  1.04521642853942145e-13;
    //     MNd Q;
    //     VNd lambda;
    //     fast_decompositions::sym_eigensolver(A, lambda, Q);
    //     std::cout << "Problematic Q: " << std::endl << Q << std::endl << std::endl;
    // }

    // Exactly diagonal inputs, in both orderings: the eigenvectors must be permuted to
    // match the sorted eigenvalues, so a descending diagonal cannot pair with the identity.
    {
        const double vals[] = {3.0, -1.0, 0.0, 2.5, -4.25, 1e-13, -1e-13};
        for (double d0 : vals) {
            for (double d1 : vals) {
                MNd A = MNd::Zero();
                for (size_t j = 0; j < N; ++j) A(j, j) = (j == 0) ? d0 : d1;
                MNd Q;
                VNd lambda;
                fast_decompositions::sym_eigensolver(A, lambda, Q);
                REQUIRE((Q.transpose() * Q - MNd::Identity()).norm() < tol);
                const double Anorm = A.norm();
                REQUIRE((Q * lambda.asDiagonal() * Q.transpose() - A).norm() <= tol * std::max(Anorm, 1.0));
                for (size_t j = 1; j < N; ++j) REQUIRE(lambda[j] >= lambda[j - 1]); // ascending, as documented
            }
        }
    }

    static constexpr size_t numTests = 1e7; // increase this for more exhaustive but slower testing; has been tested at 1e9.
    for (size_t i = 0; i < numTests; ++i) {
        MNd A = MNd::Random();
        A = (A + A.transpose()).eval();
        MNd Q;
        VNd lambda;
        fast_decompositions::sym_eigensolver(A, lambda, Q);

        std::cout.precision(16);
        // std::cout << "Q: " << std::endl << Q << std::endl << std::endl;
        // std::cout << "lambda: " << lambda.transpose() << std::endl << std::endl;

        // std::cout << "Reconstructed A: " << std::endl << (Q * lambda.asDiagonal() * Q.transpose()) << std::endl << std::endl;
        // std::cout << "Original A: " << std::endl << A << std::endl << std::endl;
        // std::cout << "Absolute backward error: " << (Q * lambda.asDiagonal() * Q.transpose() - A).norm() << std::endl;
        REQUIRE((Q.transpose() * Q - MNd::Identity()).norm() < tol);
        REQUIRE((Q * lambda.asDiagonal() * Q.transpose() - A).norm() / A.norm() < tol); // Check backward error.
    }

    // Test more difficult degenerate and near-degenerate cases.
    // Using random Q matrices, try cases where the eigenvalues
    // cluster close to one another and also around zero.
    for (size_t i = 0; i < numTests; ++i) {
        MNd Q;
        // Use QR to get random orthogonal matrices
        Q = MNd::Random().householderQr().householderQ();
        VNd lambda;
        lambda[0] = random_logspaced(1e-16, 1e-8);
        if (rand() % 1000 == 0) lambda[0] = 0; // Occasionally test exact zero eigenvalue.
        for (size_t j = 1; j < N; ++j) {
            lambda[j] = lambda[j - 1] * random_logspaced(1 - 1e-8, 1 - 1e-16); // Each subsequent eigenvalue in [s_{i-1}*1e-10, s_{i-1}]
            if (rand() % 100 == 0) // Occasionally test repeated eigenvalues.
                lambda[j] = lambda[j - 1];
        }
        MNd A = Q * lambda.asDiagonal() * Q.transpose();

        MNd Q2;
        VNd lambda2;
        fast_decompositions::sym_eigensolver(A, lambda2, Q2);


        Real Q_orthogonality_error_near_degenerate_case = (Q2.transpose() * Q2 - MNd::Identity()).norm();
        Real abs_backward_error_near_degenerate_case = (Q2 * lambda2.asDiagonal() * Q2.transpose() - A).norm();

        if (!(Q_orthogonality_error_near_degenerate_case < tol) || // Negation is used to catch `NaN` too...
            !(abs_backward_error_near_degenerate_case <= tol * A.norm())) {

            std::cout.precision(18);
            std::cout << "Original A: " << std::endl << A << std::endl << std::endl;
            std::cout << "Q: " << std::endl << Q << std::endl << std::endl;
            std::cout << "lambda: " << lambda.transpose() << std::endl << std::endl;

            std::cout << "Computed Eigenvalue Decomposition:" << std::endl;
            std::cout << "Q2: " << std::endl << Q2 << std::endl << std::endl;
            std::cout << "lambda2: " << lambda2.transpose() << std::endl << std::endl;
            std::cout << "Q2^T Q2: " << std::endl << (Q2.transpose() * Q2) << std::endl << std::endl;
            std::cout << "Q orthogonality error: " << Q_orthogonality_error_near_degenerate_case << std::endl << std::endl;
            std::cout << "Reconstructed A: " << std::endl << (Q2 * lambda2.asDiagonal() * Q2.transpose()) << std::endl << std::endl;
            std::cout << "Absolute backward error: " << (Q2 * lambda2.asDiagonal() * Q2.transpose() - A).norm() << std::endl;
            std::cout << "A norm: " << A.norm() << std::endl;
        }

        REQUIRE(Q_orthogonality_error_near_degenerate_case < tol);
        REQUIRE(abs_backward_error_near_degenerate_case <= tol * A.norm()); // Check backward error.
    }
}

template<size_t N>
void test_svd() {
    using MNd = MatN_T<double, N>;
    using VNd = VecN_T<double, N>;

    const double tol = (N == 3) ? 1e-9 : 0.5e-14;

    // // Specifically problematic inputs we've discovered (for debugging).
    // {
    //     MNd A;
    //     A << -3.94473647360167662e-16,  -1.1624128097566198e-17,  -3.3884401797112678e-16,
    //           9.26248386030959244e-17,  4.96377381885518148e-16, -1.24859846855218819e-16,
    //           3.26145716118725673e-16, -1.55029744658660994e-16, -3.74372271220711362e-16;
    //     MNd U2, V2;
    //     VNd sigma2;
    //     fast_decompositions::svd(A, U2, sigma2, V2);
    //     std::cout << "Problematic V: " << std::endl << V2 << std::endl << std::endl;
    // }

    static constexpr size_t numTests = 1e7; // increase this for more exhaustive but slower testing; has been tested at 1e9.
    // Uniformly random tests
    for (size_t i = 0; i < numTests; ++i) {
        MNd A = MNd::Random();
        MNd U, V;
        VNd sigma;
        fast_decompositions::svd(A, U, sigma, V);

        // std::cout.precision(16);
        // std::cout << "U: " << std::endl << U << std::endl << std::endl;
        // std::cout << "V: " << std::endl << V << std::endl << std::endl;
        // std::cout << "sigma: " << sigma.transpose() << std::endl << std::endl;

        // std::cout << "Reconstructed A: " << std::endl << (U * sigma.asDiagonal() * V.transpose()) << std::endl << std::endl;
        // std::cout << "Original A: " << std::endl << A << std::endl << std::endl;
        // std::cout << "Absolute backward error: " << (U * sigma.asDiagonal() * V.transpose() - A).norm() << std::endl;
        REQUIRE((U.transpose() * U - MNd::Identity()).norm() < tol);
        REQUIRE((V.transpose() * V - MNd::Identity()).norm() < tol);
        REQUIRE((U * sigma.asDiagonal() * V.transpose() - A).norm() / A.norm() < tol); // Check backward error.
        REQUIRE(sigma[0] >= sigma[1]);
        if (N == 3) REQUIRE(sigma[1] >= sigma[2]);
    }

    // Test more difficult degenerate and near-degenerate cases.
    // Using random U and V matrices, try cases where the singular values
    // cluster close to one another and also around zero.
    for (size_t i = 0; i < numTests; ++i) {
        MNd U, V;
        // Use QR to get random orthogonal matrices
        U = MNd::Random().householderQr().householderQ();
        V = MNd::Random().householderQr().householderQ();
        VNd sigma;
        sigma[0] = random_logspaced(1e-16, 1e-8);
        if (rand() % 1000 == 0) sigma[0] = 0; // Occasionally test exact zero singular value.
        for (size_t j = 1; j < N; ++j) {
            sigma[j] = sigma[j - 1] * random_logspaced(1 - 1e-8, 1 - 1e-16); // Each subsequent singular value in [s_{i-1}*1e-10, s_{i-1}]
            if (rand() % 100 == 0) // Occasionally test repeated singular values.
                sigma[j] = sigma[j - 1];
        }
        MNd A = U * sigma.asDiagonal() * V.transpose();

        MNd U2, V2;
        VNd sigma2;

        fast_decompositions::svd(A, U2, sigma2, V2);

        Real U_orthogonality_error_near_degenerate_case = (U2.transpose() * U2 - MNd::Identity()).norm();
        Real V_orthogonality_error_near_degenerate_case = (V2.transpose() * V2 - MNd::Identity()).norm();
        Real abs_backward_error_near_degenerate_case = (U2 * sigma2.asDiagonal() * V2.transpose() - A).norm();

        if (!(U_orthogonality_error_near_degenerate_case < tol) || // Negation is used to catch `NaN` too...
            !(V_orthogonality_error_near_degenerate_case < tol) ||
            !(abs_backward_error_near_degenerate_case <= tol * A.norm())) {

            std::cout.precision(18);
            std::cout << "Failure case:" << std::endl;
            std::cout << "Original A: " << std::endl << A << std::endl << std::endl;
            std::cout << "U: " << std::endl << U << std::endl << std::endl;
            std::cout << "V: " << std::endl << V << std::endl << std::endl;
            std::cout << "sigma: " << sigma.transpose() << std::endl << std::endl;

            std::cout << "Computed SVD:" << std::endl;
            std::cout << "U2: " << std::endl << U2 << std::endl << std::endl;
            std::cout << "V2: " << std::endl << V2 << std::endl << std::endl;
            std::cout << "sigma2: " << sigma2.transpose() << std::endl << std::endl;
            std::cout << "Reconstructed A: " << std::endl << (U2 * sigma2.asDiagonal() * V2.transpose()) << std::endl << std::endl;
            std::cout << "Absolute backward error: " << (U2 * sigma2.asDiagonal() * V2.transpose() - A).norm() << std::endl;
            std::cout << "A norm: " << A.norm() << std::endl;
        }

        REQUIRE(U_orthogonality_error_near_degenerate_case < tol);
        REQUIRE(V_orthogonality_error_near_degenerate_case < tol);
        REQUIRE(abs_backward_error_near_degenerate_case <= tol * A.norm());
        REQUIRE(sigma2[0] >= sigma2[1]);
        if (N == 3) REQUIRE(sigma2[1] >= sigma2[2]);
    }
}

template<size_t M, size_t N>
void test_rectangular_svd() {
    using Mat = Eigen::Matrix<double, M, N>;
    using MNd = MatN_T<double, N>;
    using VNd = VecN_T<double, N>;

    const double tol = 0.5e-9;

    static constexpr size_t numTests = 1e7; // increase this for more exhaustive but slower testing; has been tested at 1e9.
    // Uniformly random tests
    for (size_t i = 0; i < numTests; ++i) {
        Mat A = Mat::Random();
        Mat U;
        MNd V;
        VNd sigma;
        fast_decompositions::svd(A, U, sigma, V);

        // std::cout.precision(16);
        // std::cout << "U: " << std::endl << U << std::endl << std::endl;
        // std::cout << "V: " << std::endl << V << std::endl << std::endl;
        // std::cout << "sigma: " << sigma.transpose() << std::endl << std::endl;

        // std::cout << "Reconstructed A: " << std::endl << (U * sigma.asDiagonal() * V.transpose()) << std::endl << std::endl;
        // std::cout << "Original A: " << std::endl << A << std::endl << std::endl;
        // std::cout << "Absolute backward error: " << (U * sigma.asDiagonal() * V.transpose() - A).norm() << std::endl;
        REQUIRE((U.transpose() * U - MNd::Identity()).norm() < tol);
        REQUIRE((V.transpose() * V - MNd::Identity()).norm() < tol);
        REQUIRE((U * sigma.asDiagonal() * V.transpose() - A).norm() / A.norm() < tol); // Check backward error.
        REQUIRE(sigma[0] >= sigma[1]);
    }

#if 1
    // Test more difficult degenerate and near-degenerate cases.
    // Using random U and V matrices, try cases where the singular values
    // cluster close to one another and also around zero.
    for (size_t i = 0; i < numTests; ++i) {
        Mat U;
        MNd V;
        // Use QR to get random orthogonal matrices
        U = MatN_T<double, 3>(Mat::Random().householderQr().householderQ()).template leftCols<2>();
        V = MNd::Random().householderQr().householderQ();
        VNd sigma;
        sigma[0] = random_logspaced(1e-16, 1e-8);
        if (rand() % 1000 == 0) sigma[0] = 0; // Occasionally test exact zero singular value.
        for (size_t j = 1; j < N; ++j) {
            sigma[j] = sigma[j - 1] * random_logspaced(1 - 1e-8, 1 - 1e-16); // Each subsequent singular value in [s_{i-1}*1e-10, s_{i-1}]
            if (rand() % 100 == 0) // Occasionally test repeated singular values.
                sigma[j] = sigma[j - 1];
        }
        Mat A = U * sigma.asDiagonal() * V.transpose();

        Mat U2;
        MNd V2;
        VNd sigma2;

        fast_decompositions::svd(A, U2, sigma2, V2);

        Real U_orthogonality_error_near_degenerate_case = (U2.transpose() * U2 - MNd::Identity()).norm();
        Real V_orthogonality_error_near_degenerate_case = (V2.transpose() * V2 - MNd::Identity()).norm();
        Real abs_backward_error_near_degenerate_case = (U2 * sigma2.asDiagonal() * V2.transpose() - A).norm();

        if (!(U_orthogonality_error_near_degenerate_case < tol) || // Negation is used to catch `NaN` too...
            !(V_orthogonality_error_near_degenerate_case < tol) ||
            !(abs_backward_error_near_degenerate_case <= tol * A.norm())) {

            std::cout.precision(18);
            std::cout << "Failure case:" << std::endl;
            std::cout << "Original A: " << std::endl << A << std::endl << std::endl;
            std::cout << "U: " << std::endl << U << std::endl << std::endl;
            std::cout << "V: " << std::endl << V << std::endl << std::endl;
            std::cout << "sigma: " << sigma.transpose() << std::endl << std::endl;

            std::cout << "Computed SVD:" << std::endl;
            std::cout << "U2: " << std::endl << U2 << std::endl << std::endl;
            std::cout << "V2: " << std::endl << V2 << std::endl << std::endl;
            std::cout << "sigma2: " << sigma2.transpose() << std::endl << std::endl;
            std::cout << "Reconstructed A: " << std::endl << (U2 * sigma2.asDiagonal() * V2.transpose()) << std::endl << std::endl;
            std::cout << "Absolute backward error: " << (U2 * sigma2.asDiagonal() * V2.transpose() - A).norm() << std::endl;
            std::cout << "A norm: " << A.norm() << std::endl;
        }

        REQUIRE(U_orthogonality_error_near_degenerate_case < tol);
        REQUIRE(V_orthogonality_error_near_degenerate_case < tol);
        REQUIRE(abs_backward_error_near_degenerate_case <= tol * A.norm());
        REQUIRE(sigma2[0] >= sigma2[1]);
    }
#endif

}
TEST_CASE("fast 3x3 decompositions", "[fast_3x3_decompositions]" ) {
    test_eigs<3>();
    test_svd<3>();
    test_cubicroot();
    test_polar<3>();

    {
        // Benchmarking comparison of `fast_decompositions::svd` against Eigen's SVD
        static constexpr size_t runs = 10000000;

        std::vector<Eigen::Matrix3d> A_mats;
        A_mats.reserve(runs);
        for (size_t i = 0; i < runs; ++i)
            A_mats.push_back(Eigen::Matrix3d::Random());

        auto start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix3d U, V;
        Eigen::Vector3d sigma;
        for (size_t i = 0; i < runs; ++i)
            fast_decompositions::svd(A_mats[i], U, sigma, V);
        double duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "fast_decompositions::svd (3x3) took " << duration << "s for " << runs << " runs." << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < runs; ++i) {
            Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::NoQRPreconditioner> svd;
            svd.compute(A_mats[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
            U = svd.matrixU();
            V = svd.matrixV();
            sigma = svd.singularValues();
        }
        duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Eigen's 3x3 SVD took " << duration << "s for " << runs << " runs." << std::endl;
    }

    {
        // Benchmarking comparison of `fast_decompositions::sym_eigensolver` against Eigen's eigensolver
        static constexpr size_t runs = 10000000;
        std::vector<Eigen::Matrix3d> A_mats;
        A_mats.reserve(runs);
        for (size_t i = 0; i < runs; ++i) {
            Eigen::Matrix3d A = Eigen::Matrix3d::Random();
            A = (A + A.transpose()).eval(); // Make symmetric
            A_mats.push_back(A);
        }

        auto start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix3d Q;
        Eigen::Vector3d lambda;
        for (size_t i = 0; i < runs; ++i)
            fast_decompositions::sym_eigensolver(A_mats[i], lambda, Q);

        std::cout << std::endl;

        double duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "fast_decompositions::sym_eigensolver (3x3) took " << duration << "s for " << runs << " runs." << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < runs; ++i) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
            es.computeDirect(A_mats[i]);
        }
        duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Eigen's computeDirect 3x3 eigensolver took " << duration << "s for " << runs << " runs." << std::endl;
    }
}

TEST_CASE("fast 2x2 decompositions", "[fast_2x2_decompositions]" ) {
    test_eigs<2>();
    test_svd<2>();
    test_polar<2>();

    {
        // Benchmarking comparison of `fast_decompositions::svd` against Eigen's SVD
        static constexpr size_t runs = 10000000;
        std::vector<Eigen::Matrix2d> A_mats;

        A_mats.reserve(runs);
        for (size_t i = 0; i < runs; ++i)
            A_mats.push_back(Eigen::Matrix2d::Random());

        auto start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix2d U, V, V_sum;
        Eigen::Vector2d sigma;
        for (size_t i = 0; i < runs; ++i) {
            fast_decompositions::svd(A_mats[i], U, sigma, V);
            V_sum += V; // prevent compiler from optimizing away computations...
        }

        double duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "fast_decompositions::svd (2x2) took " << duration << "s for " << runs << " runs." << std::endl;
        std::cout << "V_sum trace: " << V_sum.trace() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < runs; ++i) {
            Eigen::JacobiSVD<Eigen::Matrix2d, Eigen::NoQRPreconditioner> svd;
            svd.compute(A_mats[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        }
        duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Eigen's 2x2 SVD took " << duration << "s for " << runs << " runs." << std::endl;
    }

    {
        // Benchmarking comparison of `fast_decompositions::sym_eigensolver` against Eigen's eigensolver
        static constexpr size_t runs = 10000000;

        std::vector<Eigen::Matrix2d> A_mats;

        A_mats.reserve(runs);
        for (size_t i = 0; i < runs; ++i) {
            Eigen::Matrix2d A = Eigen::Matrix2d::Random();
            A = (A + A.transpose()).eval(); // Make symmetric
            A_mats.push_back(A);
        }

        auto start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix2d Q, Q_sum;
        Eigen::Vector2d lambda, lambda_sum;
        for (size_t i = 0; i < runs; ++i) {
            fast_decompositions::sym_eigensolver(A_mats[i], lambda, Q);
            Q_sum += Q;
            lambda_sum += lambda;
        }
        double duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "fast_decompositions::sym_eigensolver (2x2) took " << duration << "s for " << runs << " runs." << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < runs; ++i) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es;
            es.compute(A_mats[i]);
        }
        duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Eigen's 2x2 eigensolver took " << duration << "s for " << runs << " runs." << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < runs; ++i) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es;
            es.computeDirect(A_mats[i]);
            Q_sum += es.eigenvectors();
            lambda_sum += es.eigenvalues();
        }
        duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Eigen's 2x2 computeDirect eigensolver took " << duration << "s for " << runs << " runs." << std::endl;

        std::cout << "Q_sum: " << Q_sum << std::endl;
        std::cout << "lambda: " << lambda.transpose() << std::endl;
    }
}

TEST_CASE("fast 3x2 decompositions", "[fast_3x2_decompositions]" ) {
    test_rectangular_svd<3, 2>();
}

TEST_CASE("fast 3x3 decompositions float", "[fast_3x3_decompositions_float]" ) {
    {
        // Benchmarking comparison of `fast_decompositions::svd` against Eigen's SVD
        Eigen::Matrix3f A = Eigen::Matrix3f::Random();
        static constexpr size_t runs = 10000000;

        auto start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix3f U, V;
        Eigen::Vector3f sigma;
        for (size_t i = 0; i < runs; ++i) {
            A(0, 0) += 1e-8; // make sure compiler doesn't optimize away computations
            fast_decompositions::svd(A, U, sigma, V);
        }
        double duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "fast_decompositions::svd (3x3, float) took " << duration << "s for " << runs << " runs." << std::endl;

        A.setRandom();
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < runs; ++i) {
            Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::NoQRPreconditioner> svd;
            svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            U = svd.matrixU();
            V = svd.matrixV();
            sigma = svd.singularValues();
        }
        duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Eigen's 3x3 SVD (float) took " << duration << "s for " << runs << " runs." << std::endl;
    }
}
