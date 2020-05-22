#ifndef ENERGYTRAITS_HH
#define ENERGYTRAITS_HH

#include <MeshFEM/Concepts.hh>
#include <MeshFEM_export.h>

struct UninitializedDeformationTag { }; // Tag used to avoid copying cached deformation quantities when unnecessary.

struct CRLinearElaticEnergyConcept { MESHFEM_EXPORT static constexpr char name[] = "CorotatedLinearElasticity"; };
struct   LinearElaticEnergyConcept { MESHFEM_EXPORT static constexpr char name[] =          "LinearElasticity"; };
struct     NeoHookeanEnergyConcept { MESHFEM_EXPORT static constexpr char name[] =                "NeoHookean"; };

constexpr char CRLinearElaticEnergyConcept::name[];
constexpr char   LinearElaticEnergyConcept::name[];
constexpr char     NeoHookeanEnergyConcept::name[];

template<class _Energy> struct isLinearElastic   : public models_concept<  LinearElaticEnergyConcept, _Energy> { };
template<class _Energy> struct isCRLinearElastic : public models_concept<CRLinearElaticEnergyConcept, _Energy> { };
template<class _Energy> struct isNeoHookean      : public models_concept<    NeoHookeanEnergyConcept, _Energy> { };

#endif /* end of include guard: ENERGYTRAITS_HH */
