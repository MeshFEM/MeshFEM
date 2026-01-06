////////////////////////////////////////////////////////////////////////////////
// CustomEnergyInstantiationsBinder.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Helper class to ease the generation of python bindings for user-defined
//  energy types.
//
//  See MeshFEMDemos/python_bindings/custom_energy_demo.cc for example usage.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  12/10/2025 22:27:10
*///////////////////////////////////////////////////////////////////////////////
#ifndef CUSTOMENERGYINSTANTIATIONSBINDER_HH
#define CUSTOMENERGYINSTANTIATIONSBINDER_HH

#include "EnergyBinding.hh"
#include "MeshEnergyBinder.hh"
#include "LoadBinding.hh"
#include "ElasticSolidBinding.hh"
#include "ElasticSheetBinding.hh"
#include "ParametrizationBinding.hh"
#include <MeshFEM/Elements/ParametrizationElement.hh>

// Bind instantiations of Elastic Solid, Sheet, and parametrization objects
// for a given energy density.
struct CustomEnergyInstantiationsBinder {
    CustomEnergyInstantiationsBinder(py::module &m)
        : m_module(m),
          m_e(m.def_submodule("energy")),
          m_e_d(m_e.def_submodule("detail")),
          m_l(m.def_submodule("loads")),
          m_l_d(m_l.def_submodule("detail")),
          m_es(m.def_submodule("elastic_solid")),
          m_es_d(m_es.def_submodule("detail")),
          m_esh(m.def_submodule("elastic_sheet")),
          m_esh_d(m_esh.def_submodule("detail")),
          m_param(m.def_submodule("parametrization")),
          m_param_d(m_param.def_submodule("detail"))
    {
        py::module::import("mesh_energy");
        py::module::import("elastic_object");
        py::module::import("sparse_matrices");
    }

    template<template<typename Real_, size_t Dim_> class EnergyType, bool bindSolid = true, bool bindSheet = true, bool bindParametrization = true, bool bindLoads = true>
    void bindParameterless(const std::string &name) {
        generateEnergyBindingsParameterless<EnergyType>(name, m_e, m_e_d);
        m_bindInstantiations<EnergyType, bindSolid, bindSheet, bindParametrization, bindLoads>(name);
    }

    template<template<typename Real_, size_t Dim_> class EnergyType, bool bindSolid = true, bool bindSheet = true, bool bindParametrization = true, bool bindLoads = true>
    void bindYoungPoisson(const std::string &name) {
        generateEnergyBindingsYoungPoisson<EnergyType>(name, m_e, m_e_d);
        m_bindInstantiations<EnergyType, bindSolid, bindSheet, bindParametrization, bindLoads>(name);
    }

private:
    template<template<typename Real_, size_t Dim_> class EnergyType, bool bindSolid = true, bool bindSheet = true, bool bindParametrization = true, bool bindLoads = true>
    void m_bindInstantiations(const std::string &name) {
        if constexpr (bindSolid) generateElasticSolidBindingsForEnergy<EnergyType>( m_es,  m_es_d);
        if constexpr (bindSheet) generateElasticSheetBindingsForEnergy<EnergyType>(m_esh, m_esh_d, ElasticSheetBinder());
        if constexpr (bindParametrization) bindParametrizationMeshEnergy<EnergyType<double, 2>>(m_param, m_param_d);
        LoadBinder lb;
        if constexpr (bindLoads && bindSolid) generateMeshSpecificBindings(m_l, m_l_d, impl::ESolidMeshBinder<LoadBinder, EnergyType>(lb));
        if constexpr (bindLoads && bindSheet) generateElasticSheetBindingsForEnergy<EnergyType>(m_l, m_l_d, lb);
    }

    py::module &m_module; // top-level module
    py::module m_e, m_e_d, m_l, m_l_d, m_es, m_es_d, m_esh, m_esh_d, m_param, m_param_d; // submodules

};

#endif /* end of include guard: CUSTOMENERGYINSTANTIATIONSBINDER_HH */
