#include <pybind11/pybind11.h>
#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include <MeshFEM/../../python_bindings/BindMembraneMaterial.hh>
#include <MeshFEM/../../python_bindings/EnergyBinding.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/SymmetricDirichlet.hh>
#include <IsotropicAutodiff/IsotropicAutodiffEDensity.hh>
#include "ParametrizationVariantBinding.hh"

namespace py = pybind11;
using namespace MeshFEM;

template<class E>
void bindLocalVariants(py::module &m, py::module &detail) {
    bindParametrizationMeshEnergyProjectToRestHessian<E>(m, detail);
    bindParametrizationMeshEnergyAKVF<E>(m, detail);
}

PYBIND11_MODULE(parametrization_energies, m) {
    py::module::import("mesh_energy");
    auto detail = m.def_submodule("detail");
    m.doc() = "Private isotropic and rest-Hessian/AKVF mesh energies";
    bindLocalVariants<NeoHookeanEnergy<double, 2>>(m, detail);
    bindLocalVariants<AutoHessianProjection<NeoHookeanEnergy<double, 2>>>(m, detail);
    bindLocalVariants<SymmetricDirichlet<double, 2>>(m, detail);
    bindLocalVariants<SymmetricDirichletDerivativeFree<double, 2>>(m, detail);

    using CNHE = APADIsotropicEDensity_SBased<CommonNeoHookeanFromInvariants>::membrane_type<double>;
    // The generic F-based binder also instantiates square-matrix PK2Stress.
    // Bind the membrane density's supported rectangular-F operations explicitly.
    py::class_<CNHE>(detail, "CommonNeoHookeanMembraneDensity")
        .def("setDeformationGradient", [](CNHE &e, const CNHE::Matrix &F) { e.setDeformationGradient(F); })
        .def("getDeformationGradient", &CNHE::getDeformationGradient)
        .def("energy", &CNHE::energy)
        .def("denergy", [](const CNHE &e) { return e.denergy(); })
        .def("d2energy", [](const CNHE &e) { return evaluate_d2energy_dF2(e); })
        .def_readwrite("projectionEnabled", &CNHE::projectionEnabled);
    m.def("CommonNeoHookeanMembraneDensity", [](double young, double poisson) {
        const double lambda = young * poisson / (1.0 - poisson * poisson);
        const double mu = young / (2.0 * (1.0 + poisson));
        return CNHE(lambda, mu);
    }, py::arg("young"), py::arg("poisson"));
    bindMembraneMaterial<CNHE>(m, detail);
    bindMeshEnergy<MembraneMeshEnergy<CNHE>>("CommonNeoHookeanMembrane", m, detail);
}
