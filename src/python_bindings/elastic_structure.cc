#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/ElasticStructure.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

#include "MeshFEM/MassMatrix.hh"
#include "MeshEntities.hh"

template<template<typename, size_t> class _Energy_T, size_t _K, size_t _Degree>
void
bindElasticStructure(py::module& module, py::module& detail_module)
{
    static constexpr size_t Dimension = _K; // Volumetric elasticity means embedding space equals the simplex dimension
    using Energy = _Energy_T<double, Dimension>;
    using EStructure = ElasticStructure<double, Energy, Dimension, _Degree>;
    using Mesh       = typename EStructure::Mesh;
    using EmbeddingSpace = Eigen::Matrix<Real, Dimension, 1>;

    module.def("ElasticStructure", [](const Mesh &m, const Energy &e, Real vol) {
                if (vol <= 0.0) vol = m.boundingBox().volume();
                return std::make_shared<EStructure>(e, m, vol);
            }, py::arg("mesh"), py::arg("energy"), py::arg("volume") = 0.0);


    // We are using shared pointer as holder instead of unique pointers since some function takes
    // shared pointer as arguments
    py::class_<EStructure, std::shared_ptr<EStructure>>(
      detail_module, getElasticStructureClassName<Energy, Dimension, _Degree, EmbeddingSpace>().c_str())
      .def_property_readonly_static("dimension", [](py::object /* self */) { return Dimension; })
      .def_property_readonly_static("degree", [](py::object /* self */) { return _Degree; })
      .def_property_readonly_static("energy_name",
                                    [](py::object /* self */) { return getEnergyName<Energy>(); })
      .def_property_readonly_static(
        "class_name",
        [](py::object /* self */) {
            return getElasticStructureClassName<Energy, Dimension, _Degree, EmbeddingSpace>();
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
      .def("mesh",    &EStructure::mesh)
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
      .def("massMatrix", [](const EStructure &e, bool lumped) {
                    return MassMatrix::construct_vector_valued<1>(e.mesh(), lumped);
              }, py::arg("lumped") = false)
      .def("deformedVertices",
           [&](const EStructure& m) {
               Eigen::Matrix<double, Eigen::Dynamic, Dimension> V(m.numVertices(), Dimension);
               for (const auto& v : m.mesh().vertices())
                   V.row(v.index()) = m.getNodePosition(v.node().index());
               return V;
           })
      .def("getNodePositions",
           [&](const EStructure& m) {
               size_t num_nodes = m.mesh().numNodes();
               Eigen::Matrix<double, Eigen::Dynamic, Dimension> V(num_nodes, Dimension);
               for (size_t i = 0; i < num_nodes; i++)
                   V.row(i) = m.getNodePosition(i);
               return V;
           })
     .def("visualizationGeometry", [](const EStructure &e) {
            std::vector<MeshIO::IOVertex > vertices;
            std::vector<MeshIO::IOElement> elements;

            const auto &m = e.mesh();
            for (const auto &v : m.vertices())
                vertices.emplace_back(e.getNodePosition(v.index()));
            for (const auto &ee : m.elements()) {
                elements.emplace_back();
                for (const auto &v : ee.vertices())
                    elements.back().push_back(v.index());
            }

            FEMMesh<Mesh::K, 1, typename Mesh::EmbeddingSpace> visMesh(elements, vertices);
            return getVisualizationGeometry(visMesh);
         })
     ;
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
