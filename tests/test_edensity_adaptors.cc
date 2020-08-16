#include "MeshFEM/ElasticityTensor.hh"
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/IsoCRLEWithHessianProjection.hh>
#include <catch2/catch.hpp>

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

TEST_CASE("Energy Density Adaptors", "[edensity_adaptors]") {
    Real E = 1.0;
    Real nu = 0.35;
    Real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    Real mu     = E / (2 * (1 + nu));
    Real lambdaPlaneStress = E * nu / (1.0 - nu * nu);

    SECTION("C Wrapper 2D")           { testCWrapper(            NeoHookeanEnergy<Real, 2>(lambdaPlaneStress, mu)); }
    SECTION("C Wrapper 3D")           { testCWrapper(            NeoHookeanEnergy<Real, 3>(lambda           , mu)); }
    SECTION("Composition F(C(F)) 2D") { testFCWrapperComposition(NeoHookeanEnergy<Real, 2>(lambdaPlaneStress, mu)); }
    SECTION("Composition F(C(F)) 3D") { testFCWrapperComposition(NeoHookeanEnergy<Real, 3>(lambda           , mu)); }

    SECTION("F Wrapper 2D")           { testFWrapper(            StVenantKirchhoffEnergyCBased<Real, 2>(ElasticityTensor<Real, 2>(E, nu))); }
    SECTION("F Wrapper 3D")           { testFWrapper(            StVenantKirchhoffEnergyCBased<Real, 3>(ElasticityTensor<Real, 3>(E, nu))); }
    SECTION("Composition C(F(C)) 2D") { testCFWrapperComposition(StVenantKirchhoffEnergyCBased<Real, 2>(ElasticityTensor<Real, 2>(E, nu))); }
    SECTION("Composition C(F(C)) 3D") { testCFWrapperComposition(StVenantKirchhoffEnergyCBased<Real, 3>(ElasticityTensor<Real, 3>(E, nu))); }

    SECTION("C Wrapper 2D")           { testCWrapper(            CorotatedLinearElasticity<Real, 2>(ElasticityTensor<Real, 2>(E, nu))); }
    SECTION("C Wrapper 3D")           { testCWrapper(            CorotatedLinearElasticity<Real, 3>(ElasticityTensor<Real, 3>(E, nu))); }
    SECTION("Composition F(C(F)) 2D") { testFCWrapperComposition(CorotatedLinearElasticity<Real, 2>(ElasticityTensor<Real, 2>(E, nu))); }
    SECTION("Composition F(C(F)) 3D") { testFCWrapperComposition(CorotatedLinearElasticity<Real, 3>(ElasticityTensor<Real, 3>(E, nu))); }

    SECTION("Membrane energy") {
        compareEnergies(StVenantKirchhoffEnergyCBased<Real, 2>(ElasticityTensor<Real, 2>(E, nu)),
                        StVenantKirchhoffMembraneEnergy<Real> (ElasticityTensor<Real, 2>(E, nu)));
    }

    SECTION("AutoHessianProjection 2D") {
        AutoHessianProjection<CorotatedLinearElasticity<Real, 2>> psi(ElasticityTensor<Real, 2>(E, nu));
        psi.projectionEnabled = true;
        compareFEnergies(psi, IsoCRLEWithHessianProjection<Real, 2>(lambdaPlaneStress, mu));

        psi.projectionEnabled = false;
        compareFEnergies(psi, CorotatedLinearElasticity<Real, 2>(ElasticityTensor<Real, 2>(E, nu)));
    }

    SECTION("AutoHessianProjection 3D") {
        AutoHessianProjection<CorotatedLinearElasticity<Real, 3>> psi(ElasticityTensor<Real, 3>(E, nu));
        psi.projectionEnabled = true;
        compareFEnergies(psi, IsoCRLEWithHessianProjection<Real, 3>(lambda, mu));

        psi.projectionEnabled = false;
        compareFEnergies(psi, CorotatedLinearElasticity<Real, 3>(ElasticityTensor<Real, 3>(E, nu)));
    }
}
