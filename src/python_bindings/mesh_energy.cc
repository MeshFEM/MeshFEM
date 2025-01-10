////////////////////////////////////////////////////////////////////////////////
// mesh_energy.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Bind some of the included MeshEnergy instantiations.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  11/05/2023 23:01:37
*///////////////////////////////////////////////////////////////////////////////
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <memory>

#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/Elements/HingeElement.hh>
#include <MeshFEM/Elements/MembraneElement.hh>
#include <MeshFEM/Elements/ParametrizationElement.hh>
#include <MeshFEM/Elements/SolidElement.hh>
#include <MeshFEM/Elements/DiscreteShellHingeEnergy.hh>
#include <MeshFEM/Stencils.hh>

#include "MeshBindings.hh"
#include "BindMembraneMaterial.hh"
#include "MeshEnergyBinder.hh"

#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/MetricFitting.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <MeshFEM/EnergyDensities/CollapsePreventionEnergy.hh>

// Bind the "NodalVars" factory method on each mesh type.
struct NodalVarsBinder {
    template<class FEMMesh_>
    void bind(py::module &m, py::module &detail) {
        m.def("NodalVars", [](const FEMMesh_ &mesh, size_t dim) -> std::shared_ptr<NewtonVarsBase> {
            if (dim == 1) return std::make_unique<NodalVars<1>>(mesh);
            if (dim == 2) return std::make_unique<NodalVars<2>>(mesh);
            if (dim == 3) return std::make_unique<NodalVars<3>>(mesh);
            throw std::runtime_error("Unsupported dimension");
        }, py::arg("mesh"), py::arg("dim") = FEMMesh_::EmbeddingDimension);
    }
};

template<class NVars>
void bind_nvars(py::module &detail) {
    py::class_<NVars, NewtonVarsBase, std::shared_ptr<NVars>>(detail, NameMangler<NVars>::name().c_str());
}

PYBIND11_MODULE(mesh_energy, m)
{
    m.doc() = "Bindings for the generic mesh energy infrastructure";
    py::module detail = m.def_submodule("detail");

    py::module::import("MeshFEM");
    py::module::import("py_newton_optimizer");
    py::module::import("energy");

    // Bind per-vertex scalar and vector-valued variables.
    bind_nvars<NodalVars<1>>(detail);
    bind_nvars<NodalVars<2>>(detail);
    bind_nvars<NodalVars<3>>(detail);

    generateMeshSpecificBindings(m, detail, NodalVarsBinder());

    py::class_<MaterialBase> pyMB(detail, "MaterialBase");

    py::class_<MeshEnergyBase, NewtonObjectiveTermBase, std::shared_ptr<MeshEnergyBase>>(m, "MeshEnergyBase")
        .def("materialForElement", &MeshEnergyBase::materialForElement, py::arg("ei"), py::return_value_policy::reference_internal)
        .def("numElements", &MeshEnergyBase::numElements)
        ;

    using DSHEMat = DiscreteShellHingeEnergy<double>::MaterialProperties;
    py::class_<DSHEMat, MaterialBase>(detail, "DiscreteShellMaterial")
        .def("setYoungPoisson", &DSHEMat::setYoungPoisson, py::arg("E"), py::arg("nu"))
        .def_readwrite("stiffness", &DSHEMat::stiffness)
        ;

    using NHE = NeoHookeanEnergy<double, 2>;
    bindMembraneMaterial<NHE>(m, detail);

    bindMeshEnergy<MembraneMeshEnergy<NHE>>("NeoHookeanMembrane", m, detail);
    bindMeshEnergy<DiscreteShellHingeMeshEnergy<double>>("DiscreteShellBending", m, detail);

    using NHE    =                       NeoHookeanEnergy<double, 2>;
    using NHE_HP = AutoHessianProjection<NeoHookeanEnergy<double, 2>>;
    bindMeshEnergy<ParametrizationMeshEnergy<NHE   >, NHE   >("Parametrization", m, detail);
    bindMeshEnergy<ParametrizationMeshEnergy<NHE_HP>, NHE_HP>("Parametrization", m, detail);

    // Bind solid element mesh energies
    using NHE3D = NeoHookeanEnergy<double, 3>;
    bindMeshEnergy<SolidMeshEnergy<1, NHE>, NHE>("Solid", m, detail);
    bindMeshEnergy<SolidMeshEnergy<2, NHE>, NHE>("Solid", m, detail);
    bindMeshEnergy<SolidMeshEnergy<1, NHE3D>, NHE3D>("Solid", m, detail);
    bindMeshEnergy<SolidMeshEnergy<2, NHE3D>, NHE3D>("Solid", m, detail);
}
