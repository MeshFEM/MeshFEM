#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/IsoCRLEWithHessianProjection.hh>
#include <catch2/catch.hpp>

#include "EDensityTestUtils.hh"

template<size_t N, class Psi>
Eigen::Matrix<Real, N * N, N * N> evalHessian(const Psi &psi) {
    Eigen::Matrix<Real, N * N, N * N> H;

    Eigen::Matrix<Real, N, N> dF;
    dF.setZero();
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            dF(i, j) = 1;
            auto delta_de = psi.delta_denergy(dF);
            // column major flattening order
            H.col(i + j * N) = Eigen::Map<const Eigen::Matrix<double, N * N, 1>>(delta_de.data());
            dF(i, j) = 0;
        }
    }

    if ((H - H.transpose()).squaredNorm() > 1e-10 * H.squaredNorm())
        throw std::runtime_error("Asymmetric probed Hessian");
    return H;
}

template<size_t N>
void testIsoCRLEWithHessianProjection() {
    auto lambdaFromENu = [](double E, double nu) { return (N == 3) ? (E * nu / ((1 + nu) * (1 - 2 * nu))) : ((nu * E) / (1.0 - nu * nu)); };
    auto     muFromENu = [](double E, double nu) { return E / (2 * (1 + nu)); };
    using MNd = Eigen::Matrix<double, N, N>;

    Real E = 1.0;
    Real nu = 0.35;
    ElasticityTensor<Real, N> et(E, nu);

    IsoCRLEWithHessianProjection<Real, N> psi(lambdaFromENu(E, nu), muFromENu(E, nu));
    psi.projectionEnabled = false;
    // Test exact agreement with the full anisotropic CorotatedLinearElasticity implementation
    // in the isotropic case with Hessian projection disabled.
    CorotatedLinearElasticity<Real, N> psi_groundTruth(et);

    compareFEnergies(psi_groundTruth, psi);

    // Verify that that the analytic Hessian projection agrees with a brute force implementation
    constexpr size_t ntests = 10000;
    Eigen::SelfAdjointEigenSolver<decltype(evalHessian<N>(psi))> es;

    for (size_t i = 0; i < ntests; ++i) {
        psi.setDeformationGradient(getPositiveF<MNd>());
        psi.projectionEnabled = true;
        auto H_proj = evalHessian<N>(psi);
        es.compute(H_proj);
        Eigen::VectorXd lambda_analytic = es.eigenvalues();

        psi.projectionEnabled = false;
        auto H_full = evalHessian<N>(psi);
        es.compute(H_full);
        Eigen::VectorXd lambda_ground_truth = es.eigenvalues().cwiseMax(0.0);

        REQUIRE((lambda_ground_truth - lambda_analytic).norm() < 1e-10 *  lambda_ground_truth.norm());
    }
}

TEST_CASE("Energy Densities", "[edensites]") {
    SECTION("IsoCRLEWithHessianProjection 3D") { testIsoCRLEWithHessianProjection<3>(); }
    SECTION("IsoCRLEWithHessianProjection 2D") { testIsoCRLEWithHessianProjection<2>(); }
}
