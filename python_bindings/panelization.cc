#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include <MeshFEM/../../python_bindings/BindMembraneMaterial.hh>
#include "../PanelizationHingeEnergy.hh"
#include "../TangentPlaneFitter.hh"
#include "../SurfaceAreaFitter.hh"

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

    using TPFED = TangentPlaneFittingEnergyDensity<double>;
    py::enum_<TPFED::Variant>(m, "TangentPlaneFittingVariant")
        .value("FitMetricAndRotation", TPFED::Variant::FitMetricAndRotation)
        .value("FitArea",              TPFED::Variant::FitArea)
        .value("FitNormalOnly",        TPFED::Variant::FitNormalOnly)
        .export_values()
        ;

    bindMembraneMaterial<TPFED>(m, detail)
        ;

    py::class_<TPFED>(detail, "TangentPlaneFittingEnergyDensity")
        .def_readwrite("stiffness", &TPFED::stiffness)
        .def_readwrite("FB_tgt",    &TPFED::FB_tgt)
        .def_readwrite("variant",   &TPFED::variant)
        ;

    bindMeshEnergy<TangentPlaneFitter>("TangentPlaneFitter", m, detail)
        .def("setVariant",   &TangentPlaneFitter::setVariant)
        .def("setStiffness", &TangentPlaneFitter::setStiffness)
        ;
    m.def("TangentPlaneFitter", [](std::shared_ptr<typename TangentPlaneFitter::Mesh> mesh, std::shared_ptr<typename TangentPlaneFitter::Vars> vars, double stiffness, TPFED::Variant variant) {
        auto me = std::make_shared<TangentPlaneFitter>(mesh, vars, stiffness, variant);
        return me;
    }, py::arg("mesh"), py::arg("vars"), py::arg("stiffness"), py::arg("variant") = TPFED::Variant::FitMetricAndRotation);

    bindMeshEnergy<SurfaceAreaFitter>("SurfaceAreaFitter", m, detail, /* bindConstructors= */ false)
        .def_readwrite("A_tgt",     &SurfaceAreaFitter::A_tgt)
        .def("surfaceArea",         &SurfaceAreaFitter::surfaceArea)
        .def("surfaceAreaGradient", &SurfaceAreaFitter::surfaceAreaGradient)
        ;

    m.def("SurfaceAreaFitter", [](std::shared_ptr<typename SurfaceAreaFitter::Mesh> mesh, std::shared_ptr<typename SurfaceAreaFitter::Vars> vars, double A_tgt) {
        auto saf = std::make_shared<SurfaceAreaFitter>(mesh, vars);
        if (A_tgt == -1) saf->A_tgt = saf->surfaceArea();
        else             saf->A_tgt = A_tgt;
        return saf;
    }, py::arg("mesh"), py::arg("vars"), py::arg("A_tgt") = -1);
}
