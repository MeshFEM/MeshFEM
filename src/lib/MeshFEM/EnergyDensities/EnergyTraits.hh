#ifndef ENERGYTRAITS_HH
#define ENERGYTRAITS_HH

#include <MeshFEM/Concepts.hh>

struct UninitializedDeformationTag { }; // Tag used to avoid copying cached deformation quantities when unnecessary.

// Sometimes the cached quantities needed to evaluate the Hessian are expensive
// to compute, and we would like to avoid these costs when we only need the
// energy or gradient. We therefore allow the caller to specify exactly what
// quantities they need when setting the energy density's deformation.
enum class EvalLevel : int { EnergyOnly = 0, Gradient = 1, Hessian = 3, Full = 3, HessianWithDisabledProjection = 4 };

// We support three types of energy densities:
//      F-based energies  psi(F  ) where F   is a 2x2 or 3x3 deformation gradient
//      C-based energies  psi(C  ) where C   is a 2x2 or 3x3 Cauchy deformation tensor (F^T F)
//      Membrane energies psi(F32) where F32 is a 3x2 deformation gradient.
enum class EDensityType { FBased, CBased, Membrane };

namespace Concepts {
    struct CRLinearElaticEnergy { static constexpr const char *name() { return "CorotatedLinearElasticity"; } };
    struct   LinearElaticEnergy { static constexpr const char *name() { return          "LinearElasticity"; } };
    struct     NeoHookeanEnergy { static constexpr const char *name() { return                "NeoHookean"; } };
    struct           StVKEnergy { static constexpr const char *name() { return         "StVenantKirchhoff"; } };
}

template<class _Energy> struct isCRLinearElastic : public models_concept<Concepts::CRLinearElaticEnergy, _Energy> { };
template<class _Energy> struct isLinearElastic   : public models_concept<Concepts::  LinearElaticEnergy, _Energy> { };
template<class _Energy> struct isNeoHookean      : public models_concept<Concepts::    NeoHookeanEnergy, _Energy> { };
template<class _Energy> struct isStVK            : public models_concept<Concepts::          StVKEnergy, _Energy> { };

#endif /* end of include guard: ENERGYTRAITS_HH */
