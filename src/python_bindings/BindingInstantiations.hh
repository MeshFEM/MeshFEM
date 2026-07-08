////////////////////////////////////////////////////////////////////////////////
// BindingInstantiations.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Collects in one place all Mesh/ElasticObject/ElasticSheet/etc. binding
//  generation (so corresponding bindings of functions taking these objects are
//  also generated).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/08/2020 19:04:12
////////////////////////////////////////////////////////////////////////////////
#ifndef BINDING_INSTANTIATIONS_HH
#define BINDING_INSTANTIATIONS_HH

#include <utility>

#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/ElasticSolid.hh>
#include <MeshFEM/ElasticSheet.hh>

#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/IsoCRLEWithHessianProjection.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/EnergyDensities/TensionFieldNeoHookean.hh>

#include <MeshFEM/EnergyDensities/SymmetricDirichlet.hh>
#include <MeshFEM/EnergyDensities/AutodiffEDensity.hh>

#include "MeshBindings.hh"

namespace MeshFEM {

namespace impl {
    // MeshBinder that generates ElasticSolid bindings with a particular energy density.
    template<class ESolidBinder, template<typename, size_t> class _Energy_T>
    struct ESolidMeshBinder {
        ESolidMeshBinder(ESolidBinder &b) : m_b(b) { }

        template<class Mesh>
        std::enable_if_t<Mesh::EmbeddingDimension == Mesh::K> // Only K-D meshes embedded in K-D
        bind(py::module &module, py::module &detail_module) {
            m_b.template bind<ElasticSolid<Mesh::K, Mesh::Deg, typename Mesh::EmbeddingSpace,
                                           _Energy_T<typename Mesh::Real, Mesh::K>>>(module, detail_module);
        }

        template<class Mesh>
        std::enable_if_t<Mesh::EmbeddingDimension != Mesh::K>
        bind(py::module /* &module */, py::module &/* detail_module */) {
            // Meshes embedded in higher dimensions (e.g., a triangle mesh in 3D) cannot be used as elastic solids.
        }

    private:
        ESolidBinder &m_b;
    };
}

template<typename _Real, size_t _N>
using StVenantKirchhoffEnergyHP = AutoHessianProjection<StVenantKirchhoffEnergy<_Real, _N>>;

template<typename _Real, size_t _N>
using NeoHookeanEnergyHP = AutoHessianProjection<NeoHookeanEnergy<_Real, _N>>;

template<template<typename, size_t> class Energy, class ESBinder>
void generateElasticSolidBindingsForEnergy(py::module &m, py::module &detail_module, ESBinder &&b) {
    generateMeshSpecificBindings(m, detail_module, impl::ESolidMeshBinder<ESBinder, Energy>(b));
}

template<class ESBinder>
void generateElasticSolidBindings(py::module &m, py::module &detail_module, ESBinder &&b) {
    // For each energy, generate an elastic solid binding
    // for each mesh we bind (neglecting meshes embedded in higher dimensions).

#if 1
    // generateMeshSpecificBindings(m, detail_module, impl::ESolidMeshBinder<ESBinder,          LinearElasticEnergy>(b));
    // generateMeshSpecificBindings(m, detail_module, impl::ESolidMeshBinder<ESBinder,             NeoHookeanEnergy>(b));
    // generateMeshSpecificBindings(m, detail_module, impl::ESolidMeshBinder<ESBinder,    CorotatedLinearElasticity>(b));
    // generateMeshSpecificBindings(m, detail_module, impl::ESolidMeshBinder<ESBinder,      StVenantKirchhoffEnergy>(b));

    // generateMeshSpecificBindings(m, detail_module, impl::ESolidMeshBinder<ESBinder, IsoCRLEWithHessianProjection>(b));
    // generateMeshSpecificBindings(m, detail_module, impl::ESolidMeshBinder<ESBinder,    StVenantKirchhoffEnergyHP>(b));
    generateElasticSolidBindingsForEnergy<    NeoHookeanEnergyHP>(m, detail_module, b);
    generateElasticSolidBindingsForEnergy<CommonNeoHookeanEnergy>(m, detail_module, b);
    generateElasticSolidBindingsForEnergy<    SymmetricDirichlet>(m, detail_module, b);
    generateElasticSolidBindingsForEnergy<SymmetricDirichletDerivativeFree>(m, detail_module, b);
#endif
}

template<template<typename, size_t> class Energy, class ESBinder>
void generateElasticSheetBindingsForEnergy(py::module &m, py::module &detail_module, ESBinder &&b) {
    b.template bind<ElasticSheet<Energy<double, 2>>>(m, detail_module);
}

template<class ESBinder>
void generateElasticSheetBindings(py::module &m, py::module &detail_module, ESBinder &&b) {
    // b.template bind<ElasticSheet<StVenantKirchhoffEnergyCBased<double, 2>>>(m, detail_module);
    // b.template bind<ElasticSheet<             NeoHookeanEnergy<double, 2>>>(m, detail_module);
    // b.template bind<ElasticSheet<   OptionalTensionFieldEnergy<double   >>>(m, detail_module);
    generateElasticSheetBindingsForEnergy<NeoHookeanEnergy>(m, detail_module, b);
}

// Generate instantiations for all of MeshFEM's default energy densities defined above;
// user-defined energies will need to be bound separately by directly calling
//      `generateElasticSolidBindingsForEnergy` or
//      `generateElasticSheetBindingsForEnergy`.
template<class EOBinder>
void generateElasticObjectBindings(py::module &m, py::module &detail_module, EOBinder &&b) {
    generateElasticSolidBindings<EOBinder>(m, detail_module, std::forward<EOBinder>(b));
    generateElasticSheetBindings<EOBinder>(m, detail_module, std::forward<EOBinder>(b));
}

} // namespace MeshFEM

#endif /* end of include guard: BINDING_INSTANTIATIONS_HH */
