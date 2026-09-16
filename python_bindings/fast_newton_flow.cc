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

        .def("initCoefficients", &NFME::initCoefficients, py::arg("d"), py::arg("arclen") = false, py::arg("projectHessian") = false)
        .def("getCoefficient", []( NFME &me, int d) -> py::array {
            const auto &xd = me.getCoefficient(d);
            return py::array(xd.size(), xd.data());
        }, py::arg("d"), "Get the degree-d Taylor coefficient as a numpy array (note: this is a view into the internal storage of the energy, not a copy)")
        .def("upgradeToDegree", &NFME::upgradeToDegree, py::arg("hessianFactorization"), py::arg("targetDegree"))
        .def("computeTaylorCoefficients", &NFME::computeTaylorCoefficients, py::arg("hessianFactorization"), py::arg("x1"), py::arg("degree") = 6, py::arg("arclen") = false, py::arg("projectHessian") = false)
        .def_readwrite("pk1ChunkSize", &NFME::pk1ChunkSize, "PK1/arclength graph chunk size; 0 selects degree-dependent automatic tuning")
        .def_readwrite("projectionChunkSize", &NFME::projectionChunkSize, "Projection graph chunk size; 0 selects automatic tuning for the projected-element count")
        .def_readonly("neg_delta_g", &NFME::neg_delta_g)
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
