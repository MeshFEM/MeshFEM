#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include "../PanelizationHingeEnergy.hh"
#include "../TangentPlaneFitter.hh"

PYBIND11_MODULE(panelization, m)
{
    py::module::import("mesh_energy");
    py::module detail = m.def_submodule("detail");

    using PHEMat = PanelizationHingeEnergy<double>::MaterialProperties;
    py::class_<PHEMat, MaterialBase>(detail, "PanelizationMaterial")
        .def_readwrite("stiffness", &PHEMat::stiffness)
        .def_readwrite("delta",     &PHEMat::delta)
        ;

    bindMeshEnergy<HingeMeshEnergy<PanelizationHingeEnergy<double>>>("Panelization", m, detail);

    using TPFMat = TangentPlaneFittingMaterial<double>;
    py::class_<TPFMat, MaterialBase>(detail, "TangentPlaneFittingMaterial")
        .def_property("stiffness", [](const TPFMat &m) { return m.psi.stiffness; }, [](TPFMat &m, double s) { m.psi.stiffness = s; })
        .def_property("FB_tgt",    [](const TPFMat &m) { return m.psi.FB_tgt; }, [](TPFMat &m, const Eigen::Matrix<double, 3, 2> &FB_tgt) { m.psi.FB_tgt = FB_tgt; })
        ;

    using TPFED = TangentPlaneFittingEnergyDensity<double>;
    py::enum_<TPFED::Variant>(m, "TangentPlaneFittingVariant")
        .value("FitMetricAndRotation", TPFED::Variant::FitMetricAndRotation)
        .value("FitArea",              TPFED::Variant::FitArea)
        .value("FitNormalOnly",        TPFED::Variant::FitNormalOnly)
        .export_values()
        ;

    bindMeshEnergy<TangentPlaneFitter>("TangentPlaneFitter", m, detail)
        .def("setVariant",   &TangentPlaneFitter::setVariant)
        .def("setStiffness", &TangentPlaneFitter::setStiffness)
        ;
    m.def("TangentPlaneFitter", [](std::shared_ptr<typename TangentPlaneFitter::Mesh> mesh, std::shared_ptr<typename TangentPlaneFitter::Vars> vars, double stiffness, TPFED::Variant variant) {
        auto me = std::make_shared<TangentPlaneFitter>(mesh, vars, stiffness, variant);
        return me;
    }, py::arg("mesh"), py::arg("vars"), py::arg("stiffness"), py::arg("variant") = TPFED::Variant::FitMetricAndRotation);
}
