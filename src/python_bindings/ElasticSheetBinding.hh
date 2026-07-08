#ifndef ELASTICSHEETBINDING_HH
#define ELASTICSHEETBINDING_HH

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <MeshFEM/ElasticSheet.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include "MeshEntities.hh"

#include "BindingInstantiations.hh"

namespace MeshFEM {

struct ElasticSheetBinder {
    template<class ES>
    static void bind(py::module &module, py::module &detail_module) {
        using Energy = typename ES::Psi_2x2;
        using Mesh   = typename ES::Mesh;
        using MX3d   = Eigen::Matrix<Real, Eigen::Dynamic, 3>;

        using Real   = typename ES::Real;
        using EO     = ElasticObject<Real>;

        using CreaseEdges = typename ES::CreaseEdges;
        module.def("ElasticSheet", [](const std::shared_ptr<Mesh> &m, const Energy &e, const CreaseEdges &creases) {
                return std::make_shared<ES>(m, e, creases); }, py::arg("mesh"), py::arg("energy"), py::arg("creaseEdges") = CreaseEdges());

        py::class_<ES, EO, std::shared_ptr<ES>> pyES(detail_module, NameMangler<ES>::name().c_str());

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
          .def_property_readonly_static("dimension",   [](py::object /* self */) { return 3; })
          .def("mesh",                     py::overload_cast<>(&ES::mesh), py::return_value_policy::reference_internal)
          .def("numThetas",                &ES::numThetas)
          .def("numCreases",               &ES::numCreases)
          .def("thetaOffset",              &ES::thetaOffset)
          .def("creaseAngleOffset",        &ES::creaseAngleOffset)
          .def("getThetas",                &ES::getThetas)
          .def("setThetas",                &ES::setThetas)
          .def("getCreaseAngles",          &ES::getCreaseAngles)
          .def("setCreaseAngles",          &ES::setCreaseAngles)
          .def("programFlatRestCurvature", &ES::programFlatRestCurvature)
          .def("programRestCurvature",     &ES::programRestCurvature)
          .def("setDeformedPositions",     &ES::setDeformedPositions)
          .def("getDeformedPositions",     &ES::deformedPositions)
          .def("getRestPositions",         &ES::restPositions)
          .def("applyRigidTransform",      &ES::applyRigidTransform, py::arg("R"), py::arg("t"))
          .def("prepareRigidMotionPins",   &ES::prepareRigidMotionPins)
          .def("filterRMPinArtifacts",     &ES::filterRMPinArtifacts, py::arg("pinVertices"))
          .def("initializeMidedgeNormals", &ES::initializeMidedgeNormals, py::arg("inferCreaseAngles") = true, py::arg("minimizeBending") = true)
          .def("updateSourceFrame",        &ES::updateSourceFrame)
          .def("getII",                    &ES::getII, py::arg("ei"))
          .def("getRestII",                &ES::getRestII, py::arg("ei"))
          .def("getC",                     &ES::getC, py::arg("ei"))
          .def("getMembraneGreenStrain",   &ES::getMembraneGreenStrain, py::arg("ei"))
          .def("vertexGreenStrains",       &ES::vertexGreenStrains)
          .def("getElementVolumetricStrain",  &ES::getElementVolumetricStrain, py::arg("ei"), py::arg("z"))
          .def("getElementCauchyStress",      &ES::getElementCauchyStress, py::arg("ei"), py::arg("z"))
          .def("getVertexVolumetricStrains",  &ES::getVertexVolumetricStrains, py::arg("z"))
          .def("getVertexCauchyStresses",     &ES::getVertexCauchyStresses, py::arg("z"))
          .def("vertexGreenStrains",       &ES::vertexGreenStrains)
          .def("getPrincipalCurvatures",   &ES::getPrincipalCurvatures)
          .def("getAlphas",                &ES::getAlphas)
          .def("getGammas",                &ES::getGammas)
          .def("getSourceAlphas",          &ES::getSourceAlphas)
          // The following overloads of the EO bindings are needed for ES-specific arguments.
          .def("energy",                   [](const ES &es, EType etype) { return es.energy(etype); }, py::arg("etype") = EType::Full)
          .def("gradient",                 [](const ES &es, bool us, VariableMask vmask, EType etype) { return es.gradient(us, vmask, etype); }, py::arg("updatedSource") = false, py::arg("vmask") = VariableMask::Defo, py::arg("etype") = EType::Full)
          .def("hessian",                  [](const ES &es, bool p, VariableMask vmask, EType etype) { return es.hessian(p, vmask, etype); }, py::arg("projectionMask") = false, py::arg("vmask") = VariableMask::Defo, py::arg("etype") = EType::Full)

          .def("midedgeNormals",         &ES::midedgeNormals)
          .def("midedgeReferenceFrames", &ES::midedgeReferenceFrames)
          .def("sourceReferenceFrames",  &ES::sourceReferenceFrames)
          .def("edgeMidpoints",          &ES::edgeMidpoints)
          .def("restEdgeMidpoints",      &ES::restEdgeMidpoints)

          .def("setMaterials", [](ES &es, const std::vector<Energy> &psis, const std::vector<size_t> &materialForElement = {}) { es.setMaterials(psis, materialForElement); }, py::arg("psis"), py::arg("materialForElement") = nullptr)
          .def("materialForElement", &ES::materialForElement, py::return_value_policy::reference_internal)

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

          .def_readwrite("angleVarRelativeMomentOfInertia", &ES::angleVarRelativeMomentOfInertia)
          ;

        const std::string name = NameMangler<ES>::name();
   }
};

} // namespace MeshFEM

#endif /* end of include guard: ELASTICSHEETBINDING_HH */
