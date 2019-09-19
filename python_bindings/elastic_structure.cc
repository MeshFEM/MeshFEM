#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/ElasticStructure.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/Utilities/TemplateName.hh>


template<typename _Energy, size_t _Dimension, size_t _Degree>
std::string
getElasticStructureClassName()
{
    return getElasticStructureTypeName<_Energy, _Dimension, _Degree>() + "ElasticStructure";
}

template<template<typename, size_t> class _Energy, size_t _Dimension, size_t _Degree>
void
bindElasticStructure(py::module& /* module */, py::module& detail_module)
{
    using Energy = _Energy<double, _Dimension>;
    using EStructure = ElasticStructure<double, Energy, _Dimension, _Degree>;

    // We are using shared pointer as holder instead of unique pointers since some function takes
    // shared pointer as arguments
    py::class_<EStructure, std::shared_ptr<EStructure>>(
      detail_module, getElasticStructureClassName<Energy, _Dimension, _Degree>().c_str())
      .def_property_readonly_static("dimension", [](py::object /* self */) { return _Dimension; })
      .def_property_readonly_static("degree", [](py::object /* self */) { return _Degree; })
      .def_property_readonly_static("energy_name",
                                    [](py::object /* self */) { return getEnergyName<Energy>(); })
      .def_property_readonly_static(
        "class_name",
        [](py::object /* self */) {
            return getElasticStructureClassName<Energy, _Dimension, _Degree>();
        })
      .def(py::init<const Energy&, const typename EStructure::Mesh&>(),
           py::arg("energy"),
           py::arg("mesh"))
      .def(py::init<const Energy&,
                    const typename EStructure::Mesh&,
                    Real>(),
           py::arg("energy"),
           py::arg("mesh"),
           py::arg("volume"))
      .def("numVars", &EStructure::numVars)
      .def("numElements", &EStructure::numElements)
      .def("setIdentityDeformationGradient", &EStructure::setIdentityDeformationGradient)
      .def("getNodeFluctuationDisplacementVarIndices",
           &EStructure::getNodeFluctuationDisplacementVarIndices,
           py::arg("node_index"))
      .def("getNodeIndicesForVertices",
           &EStructure::getNodeIndicesForVertices,
           py::arg("vertex_indices"))
      .def("setNodeFluctuationDisplacement",
           &EStructure::setNodeFluctuationDisplacement,
           py::arg("node_index"),
           py::arg("dim"),
           py::arg("value"))
      .def("getVars", &EStructure::getVars)
      .def("setVars", &EStructure::setVars, py::arg("vars"))
      .def("energy", &EStructure::energy)
      .def("getStressTensor", &EStructure::getStressTensor)
      .def("gradient", &EStructure::gradient)
    //   .def("gradient", py::overload_cast<Eigen::VectorXd&>(&EStructure::gradient, py::const_))
      .def("hessian", py::overload_cast<>(&EStructure::hessian, py::const_))
      .def("hessian", py::overload_cast<SuiteSparseMatrix&>(&EStructure::hessian, py::const_))
      .def("laplacian", &EStructure::laplacian, py::arg("addM") = 0)
      .def("hessianSparsityPattern", &EStructure::hessianSparsityPattern)
      .def("vertices",
           [&](const EStructure& m) {
               Eigen::Matrix<double, Eigen::Dynamic, _Dimension> V(m.numVertices(), _Dimension);
               for (const auto& v : m.mesh().vertices())
                   V.row(v.index()) = m.getNodePosition(v.node().index());
               return V;
           })
      .def("elements",
           [&](const EStructure& elastic_structure) {
               std::vector<std::array<size_t, _Dimension + 1>> elements;
               elements.reserve(elastic_structure.numElements());
               std::array<size_t, _Dimension + 1> current_element;
               for (const auto& e : elastic_structure.mesh().elements())
               {
                   for (const auto& v : e.vertices())
                   {
                       current_element[v.localIndex()] = v.index();
                   }
                   elements.push_back(current_element);
               }
               return elements;
           })
      .def("boundary_elements",
           [&](const EStructure& elastic_structure) {
               std::vector<std::array<size_t, _Dimension>> elements;
               elements.reserve(elastic_structure.mesh().numBoundaryElements());
               std::array<size_t, _Dimension> current_element;
               for (const auto& e : elastic_structure.mesh().boundaryElements())
               {
                   for (const auto& v : e.vertices())
                   {
                       current_element[v.localIndex()] = v.volumeVertex().index();
                   }
                   elements.push_back(current_element);
               }
               return elements;
           })
      .def("is_tet_mesh", [&](const EStructure& elastic_structure) {
          return (elastic_structure.mesh().element(0).vertices().size() == 4);
      });
}


PYBIND11_MODULE(elastic_structure, m)
{
    py::module detail_module = m.def_submodule("detail");

    bindElasticStructure<LinearElasticEnergy, 2, 1>(m, detail_module);
    bindElasticStructure<LinearElasticEnergy, 3, 1>(m, detail_module);
    bindElasticStructure<LinearElasticEnergy, 3, 2>(m, detail_module);
    bindElasticStructure<LinearElasticEnergy, 2, 2>(m, detail_module);
    bindElasticStructure<NeoHookeanEnergy, 2, 1>(m, detail_module);
    bindElasticStructure<NeoHookeanEnergy, 3, 1>(m, detail_module);
    bindElasticStructure<NeoHookeanEnergy, 3, 2>(m, detail_module);
    bindElasticStructure<NeoHookeanEnergy, 2, 2>(m, detail_module);
}
