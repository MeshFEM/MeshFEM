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

#include "BindingInstantiations.hh"
#include "EquilibriumBinding.hh"

struct ElasticSheetBinder {
    template<class ES>
    static void bind(py::module &module, py::module &detail_module) {
        using Energy = typename ES::Psi_2x2;
        using Mesh = typename ES::Mesh;
        using MX3d   = Eigen::Matrix<Real, Eigen::Dynamic, 3>;

        using CreaseEdges = typename ES::CreaseEdges;
        module.def("ElasticSheet", [](const std::shared_ptr<Mesh> &m, const Energy &e, const CreaseEdges &creases) {
                return std::make_shared<ES>(m, e, creases); }, py::arg("mesh"), py::arg("energy"), py::arg("creaseEdges") = CreaseEdges());

        py::class_<ES, std::shared_ptr<ES>> pyES(detail_module, NameMangler<ES>::name().c_str());

        using EType = typename ES::EnergyType;
        py::enum_<EType>(pyES, "EnergyType")
            .value("Full"    ,  EType::Full)
            .value("Membrane" , EType::Membrane)
            .value("Bending",   EType::Bending)
            ;

        using HPType = typename ES::HessianProjectionType;
        py::enum_<HPType>(pyES, "HPType")
            .value("Off"    ,         HPType::Off)
            .value("MembraneFBased" , HPType::MembraneFBased)
            .value("FullXBased",      HPType::FullXBased)
            ;

        pyES
          .def("mesh",                     py::overload_cast<>(&ES::mesh), py::return_value_policy::reference)
          .def("numVars",                  &ES::numVars)
          .def("numThetas",                &ES::numThetas)
          .def("numCreases",               &ES::numCreases)
          .def("thetaOffset",              &ES::thetaOffset)
          .def("creaseAngleOffset",        &ES::creaseAngleOffset)
          .def("setIdentityDeformation",   &ES::setIdentityDeformation)
          .def("getVars",                  &ES::getVars)
          .def("getThetas",                &ES::getThetas)
          .def("setThetas",                &ES::setThetas)
          .def("getCreaseAngles",          &ES::getCreaseAngles)
          .def("setCreaseAngles",          &ES::setCreaseAngles)
          .def("setDeformedPositions",     &ES::setDeformedPositions)
          .def("getDeformedPositions",     &ES::deformedPositions)
          .def("getRestPositions",         &ES::restPositions)
          .def("getNodeDisplacements",     &ES::nodeDisplacements)
          .def("applyRigidTransform",      &ES::applyRigidTransform, py::arg("R"), py::arg("t"))
          .def("prepareRigidMotionPins",   &ES::prepareRigidMotionPins)
          .def("filterRMPinArtifacts",     &ES::filterRMPinArtifacts, py::arg("pinVertices"))
          .def("initializeMidedgeNormals", &ES::initializeMidedgeNormals, py::arg("minimizeBending") = true)
          .def("updateSourceFrame",        &ES::updateSourceFrame)
          .def("setVars",                  &ES::setVars, py::arg("vars"))
          .def("getII",                    &ES::getII)
          .def("getRestII",                &ES::getRestII)
          .def("getB",                     &ES::getB)
          .def("getC",                     &ES::getC)
          .def("getMembraneGreenStrains",  &ES::getMembraneGreenStrains)
          .def("vertexGreenStrains",       &ES::vertexGreenStrains)
          .def("getPrincipalCurvatures",   &ES::getPrincipalCurvatures)
          .def("getAlphas",                &ES::getAlphas)
          .def("getGammas",                &ES::getGammas)
          .def("getSourceAlphas",          &ES::getSourceAlphas)
          .def("energy",                   [](const ES &es, EType etype) { return es.energy(etype); }, py::arg("etype") = EType::Full)
          .def("gradient",                 [](const ES &es, bool us, EType etype) { return es.gradient(us, etype); }, py::arg("updatedSource") = false, py::arg("etype") = EType::Full)
          .def("elementEnergy",            [](const ES &es, size_t ei, EType etype) { return es.elementEnergy(ei, etype); }, py::arg("ei"), py::arg("etype") = EType::Full)
          .def("elementGradient",          [](const ES &es, size_t ei, bool us, EType etype) { return es.elementGradient(ei, us, etype); }, py::arg("ei"), py::arg("updatedSource") = false, py::arg("etype") = EType::Full)
          .def("hessian",                  [](const ES &es, EType etype, bool projectionMask) { auto H = es.hessianSparsityPattern(); es.hessian(H, etype, projectionMask); return H; }, py::arg("etype") = EType::Full, py::arg("projectionMask") = false)
          .def("hessianSparsityPattern",   &ES::hessianSparsityPattern)
          // .def("massMatrix", [](const ES &e, bool lumped) {
          //               return MassMatrix::construct_vector_valued<>(e.mesh(), lumped);
          //         }, py::arg("lumped") = false)
          .def("midedgeNormals",         &ES::midedgeNormals)
          .def("midedgeReferenceFrames", &ES::midedgeReferenceFrames)
          .def("sourceReferenceFrames"  ,&ES::sourceReferenceFrames)
          .def("edgeMidpoints",          &ES::edgeMidpoints)
          .def("restEdgeMidpoints",      &ES::restEdgeMidpoints)
          .def("getEnergyDensity",       &ES::getEnergyDensity, py::arg("ei"))
          .def("visualizationGeometry", [](const ES &obj, double normalCreaseAngle) {
                FEMMesh<Mesh::K, 1, typename Mesh::EmbeddingSpace> visMesh(getF(obj.mesh()), obj.deformedPositions());
                return getVisualizationGeometry(visMesh, normalCreaseAngle);
             }, py::arg("normalCreaseAngle") = M_PI)
          .def("visualizationField", [](const ES &es, const Eigen::VectorXd &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("visualizationField", [](const ES &es, const MX3d            &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))

          .def("normalInferenceProblem", [](ES &es) -> std::unique_ptr<NewtonProblem> { return std::make_unique<NormalInferenceProblem<ES>>(es); })

          .def_property("thickness", &ES::getThickness, &ES::setThickness)
          // For debugging purposes, drop the bending energy term.
          .def_property("disableBending", &ES::getDisabledBending, &ES::setDisabledBending)
          .def_property("hessianProjectionType", &ES::getHessianProjectionType, &ES::setHessianProjectionType)

          .def("referenceConfigSampler",   &ES::referenceConfigSampler)
          .def("deformationSamplerMatrix", &ES::deformationSamplerMatrix)
          ;

        const std::string name = NameMangler<ES>::name();
        addComputeEquilibriumBinding<ES>(pyES, detail_module, name);
   }
};

PYBIND11_MODULE(elastic_sheet, m)
{
    py::module detail_module = m.def_submodule("detail");
    py::module::import("mesh");
    py::module::import("energy");
    py::module::import("sparse_matrices");
    py::module::import("py_newton_optimizer");
    py::module::import("loads");

    generateElasticSheetBindings(m, detail_module, ElasticSheetBinder());
}
