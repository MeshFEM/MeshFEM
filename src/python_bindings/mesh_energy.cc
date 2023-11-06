////////////////////////////////////////////////////////////////////////////////
// mesh_energy.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Bind some of the included MeshEnergy instantiations.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  11/05/2023 23:01:37
*///////////////////////////////////////////////////////////////////////////////
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/BindingInstantiations.hh>
#include <memory>

#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/Elements/HingeElement.hh>
#include <MeshFEM/Elements/MembraneElement.hh>
#include <MeshFEM/Elements/DiscreteShellHingeEnergy.hh>
#include <MeshFEM/Stencils.hh>
#include "BindMembraneMaterial.hh"
#include "MeshEnergyBinder.hh"

#include <MeshFEM/EnergyDensities/MetricFitting.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <MeshFEM/EnergyDensities/CollapsePreventionEnergy.hh>

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

PYBIND11_MODULE(mesh_energy, m)
{
    m.doc() = "Bindings for the generic mesh energy infrastructure";
    py::module detail = m.def_submodule("detail");

    py::module::import("MeshFEM");
    py::module::import("py_newton_optimizer");
    py::module::import("energy");

    generateMeshSpecificBindings(m, detail, NodalVarsBinder());

    py::class_<MaterialBase> pyMB(detail, "MaterialBase");

    py::class_<MeshEnergyBase, NewtonObjectiveTerm, std::shared_ptr<MeshEnergyBase>>(m, "MeshEnergyBase")
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
}
