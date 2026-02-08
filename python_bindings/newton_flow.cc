#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include <MeshFEM/EnergyDensities/SymmetricDirichlet.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>

#include "../NewtonFlow.hh"

template<size_t Dim, size_t FEMDeg, template<typename, size_t> class Psi_>
auto bindNewtonFlow(const std::string &name, py::module &m, py::module &detail) {
    using NFME = NewtonFlowMeshEnergy<Dim, FEMDeg, Psi_>;
    return bindMeshEnergy<NFME>(name, m, detail)
        .def("computeTaylorCoefficients", &NFME::computeTaylorCoefficients, py::arg("hessianFactorization"), py::arg("degree") = 6, py::arg("projectHessian") = false)
        .def("computeTaylorCoefficientsArclen", &NFME::computeTaylorCoefficientsArclen, py::arg("hessianFactorization"), py::arg("degree") = 6, py::arg("projectHessian") = false)
        ;

}

PYBIND11_MODULE(newton_flow, m)
{
    py::module::import("mesh_energy");
    py::module detail = m.def_submodule("detail");

    bindNewtonFlow<2, 1, SymmetricDirichlet>("symmetric_dirichlet", m, detail);
    // bindNewtonFlow<2, 1, LinearElasticEnergy>("linear_elastic", m, detail);
}
