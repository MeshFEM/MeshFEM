#include "MeshFEM/ElasticityTensor.hh"
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/IsoCRLEWithHessianProjection.hh>
#include <MeshFEM/EnergyDensities/TangentElasticityTensor.hh>
#include <catch2/catch.hpp>
#include <random>

#include "EDensityTestUtils.hh"

template<class Psi_F>
void testCWrapper(Psi_F psi_F) {
    compareEnergies(EnergyDensityCBasedFromFBased<Psi_F>(psi_F), psi_F);
}

template<class Psi_C>
void testFWrapper(Psi_C psi_C) {
    compareEnergies(psi_C, EnergyDensityFBasedFromCBased<Psi_C>(psi_C));
}

template<class Psi_F>
void testFCWrapperComposition(Psi_F psi_F) {
    compareFEnergies(psi_F, EnergyDensityFBasedFromCBased<EnergyDensityCBasedFromFBased<Psi_F>>(psi_F));
}

template<class Psi_C>
void testCFWrapperComposition(Psi_C psi_C) {
    compareCEnergies(psi_C, EnergyDensityCBasedFromFBased<EnergyDensityFBasedFromCBased<Psi_C>>(psi_C));
}

template<class Psi>
void testTangentElasticityTensor() {
    static constexpr size_t N = Psi::N;
    for (size_t i = 0; i < 1000; ++i) {
        ElasticityTensor<Real, N> et;
        auto D = ElasticityTensor<Real, N>::DType::Random().eval();
        symmetrize(D);
        et.setD(D);

        auto etProbed = tangentElasticityTensor(Psi(et));
        REQUIRE((et - etProbed).frobeniusNormSq() < 1e-10);
    }
}

TEST_CASE("Energy Density Adaptors", "[edensity_adaptors]") {
    Real E = 1.0;
    Real nu = 0.35;
    Real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    Real mu     = E / (2 * (1 + nu));
    Real lambdaPlaneStress = E * nu / (1.0 - nu * nu);
    ElasticityTensor<Real, 2> et2d(E, nu);
    ElasticityTensor<Real, 3> et3d(E, nu);

    SECTION("C Wrapper 2D")           { testCWrapper(            NeoHookeanEnergy<Real, 2>(lambdaPlaneStress, mu)); }
    SECTION("C Wrapper 3D")           { testCWrapper(            NeoHookeanEnergy<Real, 3>(lambda           , mu)); }
    SECTION("Composition F(C(F)) 2D") { testFCWrapperComposition(NeoHookeanEnergy<Real, 2>(lambdaPlaneStress, mu)); }
    SECTION("Composition F(C(F)) 3D") { testFCWrapperComposition(NeoHookeanEnergy<Real, 3>(lambda           , mu)); }

    SECTION("F Wrapper 2D")           { testFWrapper(            StVenantKirchhoffEnergyCBased<Real, 2>(et2d)); }
    SECTION("F Wrapper 3D")           { testFWrapper(            StVenantKirchhoffEnergyCBased<Real, 3>(et3d)); }
    SECTION("Composition C(F(C)) 2D") { testCFWrapperComposition(StVenantKirchhoffEnergyCBased<Real, 2>(et2d)); }
    SECTION("Composition C(F(C)) 3D") { testCFWrapperComposition(StVenantKirchhoffEnergyCBased<Real, 3>(et3d)); }

    SECTION("C Wrapper 2D")           { testCWrapper(            CorotatedLinearElasticity<Real, 2>(et2d)); }
    SECTION("C Wrapper 3D")           { testCWrapper(            CorotatedLinearElasticity<Real, 3>(et3d)); }
    SECTION("Composition F(C(F)) 2D") { testFCWrapperComposition(CorotatedLinearElasticity<Real, 2>(et2d)); }
    SECTION("Composition F(C(F)) 3D") { testFCWrapperComposition(CorotatedLinearElasticity<Real, 3>(et3d)); }

    SECTION("Membrane energy") {
        // Test the 2D C-based ==> Membrane wrapper
        compareEnergies(StVenantKirchhoffEnergyCBased<Real, 2>(et2d),
                        StVenantKirchhoffMembraneEnergy<Real> (et2d));
        // Test the 2D F-based  ==> Membrane wrapper against the 2D C-based ==> Membrane Wrapper
        compareFEnergies(StVenantKirchhoffMembraneEnergy<Real> (et2d),
                         EnergyDensityFBasedMembraneFromFBased<StVenantKirchhoffEnergy<Real, 2>>(et2d));

        // The same tests, using the generic membrane adaptor interface.
        compareEnergies(StVenantKirchhoffEnergyCBased<Real, 2>(et2d),
                        MembraneEnergyDensityFrom2x2Density<StVenantKirchhoffEnergyCBased<Real, 2>>(et2d));
        compareFEnergies(StVenantKirchhoffMembraneEnergy<Real> (et2d),
                         MembraneEnergyDensityFrom2x2Density<StVenantKirchhoffEnergy<Real, 2>>(et2d));
    }

    SECTION("AutoHessianProjection 2D") {
        AutoHessianProjection<CorotatedLinearElasticity<Real, 2>> psi(et2d);
        psi.projectionEnabled = true;
        compareFEnergies(psi, IsoCRLEWithHessianProjection<Real, 2>(lambdaPlaneStress, mu));

        psi.projectionEnabled = false;
        compareFEnergies(psi, CorotatedLinearElasticity<Real, 2>(et2d));
    }

    SECTION("AutoHessianProjection 3D") {
        AutoHessianProjection<CorotatedLinearElasticity<Real, 3>> psi(et3d);
        psi.projectionEnabled = true;
        compareFEnergies(psi, IsoCRLEWithHessianProjection<Real, 3>(lambda, mu));

        psi.projectionEnabled = false;
        compareFEnergies(psi, CorotatedLinearElasticity<Real, 3>(et3d));
    }

    SECTION("TangentElasticityTensor 2D") {
        testTangentElasticityTensor<CorotatedLinearElasticity<Real, 2>>();
        testTangentElasticityTensor<StVenantKirchhoffEnergyCBased<Real, 2>>();
        testTangentElasticityTensor<StVenantKirchhoffEnergy<Real, 2>>();
    }

    SECTION("TangentElasticityTensor 3D") {
        testTangentElasticityTensor<CorotatedLinearElasticity<Real, 3>>();
        testTangentElasticityTensor<StVenantKirchhoffEnergyCBased<Real, 3>>();
        testTangentElasticityTensor<StVenantKirchhoffEnergy<Real, 3>>();
    }

    SECTION("TangentElasticityTensor Isotropic 2D") {
        std::default_random_engine gen;
        std::uniform_real_distribution<> Edist(0.1, 2000),
                                         nudist(-0.99, 0.499);

        for (size_t i = 0; i < 1000; ++i) {
            Real E_rand = Edist(gen);
            Real nu_rand = nudist(gen);

            // Note: even in the plane stress (2D) case, we need to specify the
            // (lambda, mu) for the volumetric material since NeoHookeanEnergy
            // imposes the plane stress conditions internally.
            Real lambda_3d = (nu_rand * E_rand) / ((1.0 + nu_rand) * (1.0 - 2.0 * nu_rand));
            Real mu_3d = E_rand / (2.0 + 2.0 * nu_rand);

            NeoHookeanEnergy<Real, 2> psi(lambda_3d, mu_3d);
            auto etProbed = tangentElasticityTensor(psi);
            ElasticityTensor<Real, 2> et;
            et.setIsotropic(E_rand, nu_rand);
            REQUIRE((et - etProbed).frobeniusNormSq() < 1e-10);
        }
    }

    SECTION("TangentElasticityTensor Isotropic 3D") {
        for (size_t i = 0; i < 1000; ++i) {
            auto lamMu = Eigen::Vector2d::Random().eval();
            NeoHookeanEnergy<Real, 3> psi(lamMu[0], lamMu[1]);
            auto etProbed = tangentElasticityTensor(psi);
            ElasticityTensor<Real, 3> et;
            et.setIsotropicLame(lamMu[0], lamMu[1]);
            REQUIRE((et - etProbed).frobeniusNormSq() < 1e-10);
        }
    }
}
