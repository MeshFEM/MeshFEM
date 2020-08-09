#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/MassMatrix.hh>
#include <MeshFEM/ElasticObjectLoads.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include "MeshEntities.hh"
#include "EquilibriumBinding.hh"
#include "LoadBindings.hh"

template<template<typename, size_t> class _Energy_T, size_t _K, size_t _Degree>
void bindElasticObject(py::module &module, py::module &detail_module)
{
    static constexpr size_t K   = _K;
    static constexpr size_t N   = _K;
    static constexpr size_t Deg = _Degree;
    using Vector = VectorND<N>;
    using Energy = _Energy_T<Real, N>;

    using EO   = ElasticObject<K, Deg, Vector, Energy>;
    using Mesh = typename EO::Mesh;

    module.def("ElasticObject", [](const Mesh &m, const Energy &e) { return std::make_shared<EO>(e, m); }, py::arg("mesh"), py::arg("energy"));

    const std::string name = getElasticObjectName<Energy, K, Deg, Vector>();
    py::class_<EO, std::shared_ptr<EO>> pyEO(detail_module, name.c_str());
    pyEO
      .def_property_readonly_static("dimension",   [](py::object /* self */) { return N; })
      .def_property_readonly_static("degree",      [](py::object /* self */) { return Deg; })
      .def_property_readonly_static("energy_name", [](py::object /* self */) { return getEnergyName<Energy>(); })
      .def("mesh",                   &EO::mesh)
      .def("numVars",                &EO::numVars)
      .def("numElements",            &EO::numElements)
      .def("setIdentityDeformation", &EO::setIdentityDeformation)
      .def("getVars",                &EO::getVars)
      .def("setVars",                &EO::setVars, py::arg("vars"))
      .def("applyRigidTransform",    &EO::applyRigidTransform, py::arg("R"), py::arg("t"))
      .def("prepareRigidMotionPins", &EO::prepareRigidMotionPins)
      .def("energy",                 &EO::energy)
      .def("gradient",               &EO::gradient)
      .def("hessian",                py::overload_cast<>(&EO::hessian, py::const_))
      .def("hessianSparsityPattern", &EO::hessianSparsityPattern)
      .def("massMatrix", [](const EO &e, bool lumped) {
                    return MassMatrix::construct_vector_valued<>(e.mesh(), lumped);
              }, py::arg("lumped") = false)
      .def("deformedVertices",       &EO::deformedVertices)
      .def("getEnergyDensity",       &EO::getEnergyDensity, py::arg("ei"))
      .def("visualizationGeometry", [](const EO &obj) {
            FEMMesh<Mesh::K, 1, typename Mesh::EmbeddingSpace> visMesh(getF(obj.mesh()), obj.deformedVertices());
            return getVisualizationGeometry(visMesh);
         })
     ;

    addComputeEquilibriumBinding<EO>(pyEO, detail_module, name);
    addLoadBindings<EO>(pyEO, detail_module, name);
}

PYBIND11_MODULE(elastic_object, m)
{
    py::module detail_module = m.def_submodule("detail");

    py::module::import("mesh");
    py::module::import("energy");
    py::module::import("sparse_matrices");
    py::module::import("py_newton_optimizer");
    py::module::import("loads");


    bindElasticObject<      LinearElasticEnergy, 2, 1>(m, detail_module);
    bindElasticObject<      LinearElasticEnergy, 3, 1>(m, detail_module);
    bindElasticObject<      LinearElasticEnergy, 3, 2>(m, detail_module);
    bindElasticObject<      LinearElasticEnergy, 2, 2>(m, detail_module);
    bindElasticObject<         NeoHookeanEnergy, 2, 1>(m, detail_module);
    bindElasticObject<         NeoHookeanEnergy, 3, 1>(m, detail_module);
    bindElasticObject<         NeoHookeanEnergy, 3, 2>(m, detail_module);
    bindElasticObject<         NeoHookeanEnergy, 2, 2>(m, detail_module);
    bindElasticObject<CorotatedLinearElasticity, 2, 1>(m, detail_module);
    bindElasticObject<CorotatedLinearElasticity, 3, 1>(m, detail_module);
    bindElasticObject<CorotatedLinearElasticity, 3, 2>(m, detail_module);
    bindElasticObject<CorotatedLinearElasticity, 2, 2>(m, detail_module);
    bindElasticObject<  StVenantKirchhoffEnergy, 2, 1>(m, detail_module);
    bindElasticObject<  StVenantKirchhoffEnergy, 3, 1>(m, detail_module);
    bindElasticObject<  StVenantKirchhoffEnergy, 3, 2>(m, detail_module);
    bindElasticObject<  StVenantKirchhoffEnergy, 2, 2>(m, detail_module);
}
