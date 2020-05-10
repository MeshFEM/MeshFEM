#ifndef ENERGYTRAITS_HH
#define ENERGYTRAITS_HH

#include <MeshFEM/Concepts.hh>

struct UninitializedDeformationTag { }; // Tag used to avoid copying cached deformation quantities when unnecessary.

struct LinearElaticEnergyConcept { };
struct   NeoHookeanEnergyConcept { };

template<class _Energy> struct isLinearElastic : public models_concept<LinearElaticEnergyConcept, _Energy> { };
template<class _Energy> struct isNeoHookean    : public models_concept<  NeoHookeanEnergyConcept, _Energy> { };

#endif /* end of include guard: ENERGYTRAITS_HH */
