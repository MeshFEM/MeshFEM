#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <MeshFEM/ElasticSheet.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include "MeshEntities.hh"
#include <MeshFEM/Elements/DihedralAngle.hh>

#include "BindingInstantiations.hh"

struct ElasticSheetBinder {
    template<class ES>
    static void bind(py::module &module, py::module &detail_module) {
        using Energy = typename ES::Psi_2x2;
        using Mesh   = typename ES::Mesh;
        using MX3d   = Eigen::Matrix<Real, Eigen::Dynamic, 3>;
        using Mat    = typename ES::Material;

        using Real   = typename ES::Real;
        using EO     = ElasticObject<Real>;

        using VariableMask = typename ES::VariableMask;

        using CreaseEdges = typename ES::CreaseEdges;
        module.def("ElasticSheet", [](const std::shared_ptr<Mesh> &m, const Energy &e, const CreaseEdges &creases) {
                return std::make_shared<ES>(m, e, creases); }, py::arg("mesh"), py::arg("energy"), py::arg("creaseEdges") = CreaseEdges());

        py::class_<ES, EO, std::shared_ptr<ES>> pyES(detail_module, NameMangler<ES>::name().c_str());
        py::class_<Mat>(pyES, "Material")
            .def("set", &Mat::set)
            .def_property_readonly("psi",     &Mat::psi)
            .def_property_readonly("etensor", &Mat::etensor)
            .def("setProjectionEnabled",      &Mat::setProjectionEnabled)
            ;


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
          .def("getNodeDisplacements",     &ES::nodeDisplacements)
          .def("applyRigidTransform",      &ES::applyRigidTransform, py::arg("R"), py::arg("t"))
          .def("prepareRigidMotionPins",   &ES::prepareRigidMotionPins)
          .def("filterRMPinArtifacts",     &ES::filterRMPinArtifacts, py::arg("pinVertices"))
          .def("initializeMidedgeNormals", &ES::initializeMidedgeNormals, py::arg("minimizeBending") = true)
          .def("updateSourceFrame",        &ES::updateSourceFrame)
          .def("getII",                    &ES::getII)
          .def("getRestII",                &ES::getRestII)
          .def("getC",                     &ES::getC)
          .def("getMembraneGreenStrains",  &ES::getMembraneGreenStrains)
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
          .def("hessian",                  [](const ES &es, EType etype, bool projectionMask, VariableMask vmask) { auto H = es.hessianSparsityPattern(); es.hessian(H, etype, projectionMask, vmask); return H; }, py::arg("etype") = EType::Full, py::arg("projectionMask") = false, py::arg("vmask") = VariableMask::Defo)
          .def("elementEnergy",            [](const ES &es, size_t ei, EType etype) { return es.elementEnergy(ei, etype); }, py::arg("ei"), py::arg("etype") = EType::Full)
          .def("elementGradient",          [](const ES &es, size_t ei, bool us, EType etype) { return es.elementGradient(ei, us, etype); }, py::arg("ei"), py::arg("updatedSource") = false, py::arg("etype") = EType::Full)
          .def("midedgeNormals",         &ES::midedgeNormals)
          .def("midedgeReferenceFrames", &ES::midedgeReferenceFrames)
          .def("sourceReferenceFrames",  &ES::sourceReferenceFrames)
          .def("edgeMidpoints",          &ES::edgeMidpoints)
          .def("restEdgeMidpoints",      &ES::restEdgeMidpoints)

          .def("elementPsi",        &ES::elementPsi, py::arg("ei"))
          .def("elementETensor",    &ES::elementETensor, py::arg("ei"))

          .def("numMaterials", &ES::numMaterials)
          .def("getMaterial",  [](ES &es, size_t mi) { return es.material(mi); }, py::return_value_policy::reference_internal)
          .def("setMaterials", &ES::setMaterials, py::arg("psiList"))
          .def("clearMaterialAssignments", &ES::clearMaterialAssignments)
          .def_property("elementMaterialAssignments", &ES::elementMaterialAssignments, &ES::setElementMaterialAssignments)

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
          ;

        const std::string name = NameMangler<ES>::name();
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
    py::module::import("elastic_object");

    generateElasticSheetBindings(m, detail_module, ElasticSheetBinder());

    // Standalone binding of PlateBendingElement for validation.
    using PBE = elements::PlateBending<double, elements::AngleFunctionSin>;
    using CPos = typename PBE::CornerPositions;
    using ET = typename PBE::ETensor;
    using V3d = Eigen::Vector3d;
    auto elementData = [](const CPos &x) {
        using LEE = LinearlyEmbeddedElement<Simplex::Triangle, 1, V3d>;
        LEE e;
        e.embed(x.row(0).transpose(), x.row(1).transpose(), x.row(2).transpose());
        return elements::EmbeddedMembraneElementData<LEE, LEE>(e);
    };
    py::class_<PBE>(m, "PlateBendingElement")
        .def(py::init<double>(), py::arg("thickness"))
        .def("energy", [&elementData](const PBE &e, const ET &C, const CPos &X, const CPos &x, const Eigen::Vector3d &gamma) {
                    const auto ref_edata = elementData(X);
                    return e.energy(C, e.getII(x, gamma, ref_edata), Eigen::Matrix2d::Zero(), ref_edata);
                }, py::arg("C"), py::arg("X"), py::arg("x"), py::arg("gamma"))
        .def("gradient", [&elementData](const PBE &e, const ET &C, const CPos &X, const CPos &x, const Eigen::Vector3d &gamma) {
                    const auto ref_edata = elementData(X);
                    return e.gradient(C, x, gamma, e.getII(x, gamma, ref_edata), Eigen::Matrix2d::Zero(), ref_edata);
                }, py::arg("C"), py::arg("X"), py::arg("x"), py::arg("gamma"))
        .def("hessian", [&elementData](const PBE &e, const ET &C, const CPos &X, const CPos &x, const Eigen::Vector3d &gamma) {
                    const auto ref_edata = elementData(X);
                    return e.hessian(C, x, gamma, e.getII(x, gamma, ref_edata), Eigen::Matrix2d::Zero(), ref_edata);
                }, py::arg("C"), py::arg("X"), py::arg("x"), py::arg("gamma"))
    ;

    // Standalone binding of DihedralAngle for validation.
    using DA = elements::DihedralAngle<double>;
    py::class_<DA>(m, "DihedralAngle")
        .def(py::init<>())
        .def("configure", &DA::configure, py::arg("stencilPts"))
        .def("value",    &DA::value)
        .def("gradient", &DA::gradient)
        .def("hessian",  &DA::hessian)
    ;
}
