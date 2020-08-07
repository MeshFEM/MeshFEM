#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/ElasticSheet.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include "MeshEntities.hh"
#include "EquilibriumBinding.hh"
#include "LoadBindings.hh"

template<class Psi_C>
void bindElasticSheet(py::module &module, py::module &detail_module)
{
    using Energy = Psi_C;
    using ES   = ElasticSheet<Psi_C>;
    using Mesh = typename ES::Mesh;
    using MX3d   = Eigen::Matrix<Real, Eigen::Dynamic, 3>;

    module.def("ElasticSheet", [](const std::shared_ptr<Mesh> &m, const Energy &e) { return std::make_shared<ES>(m, e); }, py::arg("mesh"), py::arg("energy"));

    const std::string name = "ElasticSheet" + getEnergyName<Energy>();
    py::class_<ES, std::shared_ptr<ES>> pyES(detail_module, name.c_str());

    using EType = typename ES::EnergyType;
    py::enum_<EType>(pyES, "EnergyType")
        .value("Full"    ,  EType::Full)
        .value("Membrane" , EType::Membrane)
        .value("Bending",   EType::Bending)
        ;

    pyES
      .def("mesh",                     py::overload_cast<>(&ES::mesh), py::return_value_policy::reference)
      .def("numVars",                  &ES::numVars)
      .def("thetaOffset",              &ES::thetaOffset)
      .def("setIdentityDeformation",   &ES::setIdentityDeformation)
      .def("getVars",                  &ES::getVars)
      .def("getThetas",                &ES::getThetas)
      .def("setThetas",                &ES::setThetas)
      .def("setDeformedPositions",     &ES::setDeformedPositions)
      .def("getDeformedPositions",     &ES::deformedPositions)
      .def("initializeMidedgeNormals", &ES::initializeMidedgeNormals, py::arg("minimizeBending") = true)
      .def("updateSourceFrame",        &ES::updateSourceFrame)
      .def("setVars",                  &ES::setVars, py::arg("vars"))
      .def("getII",                    &ES::getII)
      .def("getRestII",                &ES::getRestII)
      .def("getB",                     &ES::getB)
      .def("getC",                     &ES::getC)
      .def("getPrincipalCurvatures",   &ES::getPrincipalCurvatures)
      .def("getAlphas",                &ES::getAlphas)
      .def("getGammas",                &ES::getGammas)
      .def("getSourceAlphas",          &ES::getSourceAlphas)
      .def("energy",                   &ES::energy,   py::arg("etype") = EType::Full)
      .def("gradient",                 &ES::gradient, py::arg("updatedSource") = false,   py::arg("etype") = EType::Full)
      .def("hessian",                  [](const ES &es, EType etype) { auto H = es.hessianSparsityPattern(); es.hessian(H, etype); return H; }, py::arg("etype") = EType::Full)
      .def("hessianSparsityPattern",   &ES::hessianSparsityPattern)
      // .def("massMatrix", [](const ES &e, bool lumped) {
      //               return MassMatrix::construct_vector_valued<>(e.mesh(), lumped);
      //         }, py::arg("lumped") = false)
      .def("deformedPositions",      &ES::deformedPositions)
      .def("midedgeNormals",         &ES::midedgeNormals)
      .def("midedgeReferenceFrames", &ES::midedgeReferenceFrames)
      .def("sourceReferenceFrames"  ,&ES::sourceReferenceFrames)
      .def("edgeMidpoints",          &ES::edgeMidpoints)
      .def("getEnergyDensity",       &ES::getEnergyDensity, py::arg("ei"))
      .def("visualizationGeometry", [](const ES &obj) {
            FEMMesh<Mesh::K, 1, typename Mesh::EmbeddingSpace> visMesh(getF(obj.mesh()), obj.deformedPositions());
            return getVisualizationGeometry(visMesh);
         })
      .def("visualizationField", [](const ES &es, const Eigen::VectorXd &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
      .def("visualizationField", [](const ES &es, const MX3d            &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))

      .def("normalInferenceProblem", [](ES &es) -> std::unique_ptr<NewtonProblem> { return std::make_unique<NormalInferenceProblem<ES>>(es); })

      .def_property("thickness", &ES::getThickness, &ES::setThickness)
      ;

    addComputeEquilibriumBinding<ES>(pyES, detail_module, name);
    addLoadBindings<ES>(pyES, detail_module, name);
}

PYBIND11_MODULE(elastic_sheet, m)
{
    py::module detail_module = m.def_submodule("detail");
    py::module::import("mesh");
    py::module::import("energy");
    py::module::import("sparse_matrices");
    py::module::import("py_newton_optimizer");
    py::module::import("loads");

    bindElasticSheet<StVenantKirchhoffEnergyCBased<double, 2>>(m, detail_module);
}
