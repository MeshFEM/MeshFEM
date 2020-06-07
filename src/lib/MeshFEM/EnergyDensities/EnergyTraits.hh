#ifndef ENERGYTRAITS_HH
#define ENERGYTRAITS_HH

#include <MeshFEM/Concepts.hh>

struct UninitializedDeformationTag { }; // Tag used to avoid copying cached deformation quantities when unnecessary.

struct CRLinearElaticEnergyConcept { static constexpr const char *name() { return "CorotatedLinearElasticity"; } };
struct   LinearElaticEnergyConcept { static constexpr const char *name() { return          "LinearElasticity"; } };
struct     NeoHookeanEnergyConcept { static constexpr const char *name() { return                "NeoHookean"; } };

template<class _Energy> struct isLinearElastic   : public models_concept<  LinearElaticEnergyConcept, _Energy> { };
template<class _Energy> struct isCRLinearElastic : public models_concept<CRLinearElaticEnergyConcept, _Energy> { };
template<class _Energy> struct isNeoHookean      : public models_concept<    NeoHookeanEnergyConcept, _Energy> { };

#endif /* end of include guard: ENERGYTRAITS_HH */
