////////////////////////////////////////////////////////////////////////////////
// test_analytic_hessian_projections.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Validate analytical Hessian projection routines for energy densities psi(F)
//  against a brute-force projection via a numerical eigendecomposition of the
//  exact Hessian.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  08/29/2025 17:37:26
*///////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#include <MeshFEM/EnergyDensities/IsoCRLEFixed.hh>
#include <MeshFEM/GlobalBenchmark.hh>

// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>

// Tests for energy densities with isotropic material properties parametrized
// by (lambda, mu)
template<class Psi>
void run_tests() {
    static constexpr size_t N = Psi::N;
    using MNd = typename Psi::Matrix;

    size_t numTests = 1e7;
    for (size_t i = 0; i < numTests; ++i) {
        // Generate random material properties.
        auto lambdaFromENu = [](double E, double nu, bool is3D = true) { return is3D ? (E * nu / ((1 + nu) * (1 - 2 * nu))) : ((nu * E) / (1.0 - nu * nu)); };
        auto     muFromENu = [](double E, double nu)                   { return E / (2 * (1 + nu)); };

        Real E = 1.0, nu;
        Real alpha = double(rand()) / RAND_MAX;
        nu = (1 - alpha) * -0.99 + alpha * ((N == 3) ? 0.499 : 1);
        Real lambda = lambdaFromENu(E, nu, N == 3);
        Real mu     = muFromENu(E, nu);

        Psi psi(lambda, mu);

        MNd F;
        F.setRandom();
        F *= 1.5; // We mostly care about compression anyway...
        if (F.determinant() < 0) F.col(0) = -F.col(0); // Ensure det(F) > 0

        psi.setDeformationGradient(F, EvalLevel::HessianWithDisabledProjection);
        auto H_exact = psi.d2energy();
        psi.setDeformationGradient(F);
        auto H_proj = psi.d2energy();

        Eigen::SelfAdjointEigenSolver<decltype(H_exact)> Hes(H_exact);
        auto H_brute_proj = Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(0.0).asDiagonal() * Hes.eigenvectors().transpose();

        Real tol = 1e-8;
        Real projectionError = (H_brute_proj - H_proj).norm() / H_exact.norm();
        if (!(projectionError < tol)) {
            std::cout << "Energy density: " << Psi::name() << std::endl;
            std::cout << "F = \n" << F << std::endl;
            std::cout << "E = " << E << ", nu = " << nu << std::endl;
            std::cout << "H_exact = \n" << H_exact << std::endl;
            std::cout << "H_proj = \n" << H_proj << std::endl;
            std::cout << "H_brute_proj = \n" << H_brute_proj << std::endl;
            std::cout << "Brute-force projection vs analytical projection relative error: " << projectionError << std::endl;
            std::cout << "Relative change from projection: " << (H_brute_proj - H_exact).norm() / H_exact.norm() << std::endl;
        }
        REQUIRE(projectionError < tol);
    }
}

TEST_CASE("analytical hessian projections", "[hessian_projection]" ) {
    run_tests<CommonNeoHookeanEnergy<double, 2>>();
    run_tests<CommonNeoHookeanEnergy<double, 3>>();
}
