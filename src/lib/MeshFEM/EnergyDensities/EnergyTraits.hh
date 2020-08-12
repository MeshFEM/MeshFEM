#ifndef ENERGYTRAITS_HH
#define ENERGYTRAITS_HH

#include <MeshFEM/Concepts.hh>

struct UninitializedDeformationTag { }; // Tag used to avoid copying cached deformation quantities when unnecessary.

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
