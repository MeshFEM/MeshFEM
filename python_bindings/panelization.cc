#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include "../PanelizationHingeEnergy.hh"

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
}
