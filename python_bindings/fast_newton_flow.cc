#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include <MeshFEM/EnergyDensities/SymmetricDirichlet.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>

#include "../FastNewtonFlow.hh"

template<size_t Dim, size_t FEMDeg>
auto bindFastNewtonFlow(const std::string &name, py::module &m, py::module &detail) {
    using NFME = FastNewtonFlowMeshEnergy<Dim, FEMDeg>;
    return bindMeshEnergy<NFME>(name, m, detail)
        .def("computeTaylorCoefficients", &NFME::computeTaylorCoefficients, py::arg("hessianFactorization"), py::arg("degree") = 6, py::arg("arclen") = false, py::arg("projectHessian") = false)
        ;
}

PYBIND11_MODULE(fast_newton_flow, m)
{
    py::module::import("mesh_energy");
    py::module::import("py_newton_optimizer");
    py::module::import("rotation_strain_extrapolation");
    py::module detail = m.def_submodule("detail");

    bindFastNewtonFlow<2, 1>("symmetric_dirichlet", m, detail);
}
