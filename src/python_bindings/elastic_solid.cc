#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/ElasticSolid.hh>
#include <MeshFEM/MassMatrix.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include "MeshEntities.hh"

#include "EquilibriumBinding.hh"
#include "BindingInstantiations.hh"

struct ElasticSolidBinder {
    template<class ES>
    static void bind(py::module &module, py::module &detail_module) {
        static constexpr size_t K   = ES::K;
        static constexpr size_t N   = ES::K;
        static constexpr size_t Deg = ES::Deg;
        using Vector = VectorND<N>;
        using Energy = typename ES::Energy;
        using MXNd   = Eigen::Matrix<Real, Eigen::Dynamic, N>;
        using Mesh = typename ES::Mesh;

        module.def("ElasticSolid", [](const Mesh &m, const Energy &e) { return std::make_shared<ES>(e, m); }, py::arg("mesh"), py::arg("energy"));

        const std::string name = getElasticSolidName<Energy, K, Deg, Vector>();
        py::class_<ES, std::shared_ptr<ES>> pyEO(detail_module, name.c_str());
        pyEO
          .def_property_readonly_static("dimension",   [](py::object /* self */) { return N; })
          .def_property_readonly_static("degree",      [](py::object /* self */) { return Deg; })
          .def_property_readonly_static("energy_name", [](py::object /* self */) { return getEnergyName<Energy>(); })
          .def("mesh",                   &ES::mesh)
          .def("numVars",                &ES::numVars)
          .def("numElements",            &ES::numElements)
          .def("setIdentityDeformation", &ES::setIdentityDeformation)
          .def("getVars",                &ES::getVars)
          .def("setVars",                &ES::setVars, py::arg("vars"))
          .def("applyRigidTransform",    &ES::applyRigidTransform, py::arg("R"), py::arg("t"))
          .def("prepareRigidMotionPins", &ES::prepareRigidMotionPins)
          .def("energy",                 &ES::energy)
          .def("gradient",               &ES::gradient)
          .def("hessian",                py::overload_cast<>(&ES::hessian, py::const_))
          .def("hessianSparsityPattern", &ES::hessianSparsityPattern)
          .def("massMatrix", [](const ES &e, bool lumped) {
                        return MassMatrix::construct_vector_valued<>(e.mesh(), lumped);
                  }, py::arg("lumped") = false)
          .def("getDeformedPositions",   &ES::deformedPositions)
          .def("getRestPositions",       &ES::restPositions)
          .def("getNodeDisplacements",   &ES::nodeDisplacements)
          .def("getEnergyDensity",       &ES::getEnergyDensity, py::arg("ei"), py::return_value_policy::reference)
          .def("visualizationGeometry", [](const ES &obj) {
                FEMMesh<Mesh::K, 1, typename Mesh::EmbeddingSpace> visMesh(getF(obj.mesh()), obj.deformedVertices());
                return getVisualizationGeometry(visMesh);
             })
          .def("visualizationField", [](const ES &es, const Eigen::VectorXd &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("visualizationField", [](const ES &es, const MXNd            &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
         ;

        addComputeEquilibriumBinding<ES>(pyEO, detail_module, name);
    }
};

PYBIND11_MODULE(elastic_solid, m)
{
    py::module detail_module = m.def_submodule("detail");

    py::module::import("mesh");
    py::module::import("energy");
    py::module::import("sparse_matrices");
    py::module::import("py_newton_optimizer");
    py::module::import("loads");

    generateElasticSolidBindings(m, detail_module, ElasticSolidBinder());
}
