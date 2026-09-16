#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include <MeshFEM/EnergyDensities/SymmetricDirichlet.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>

#include "../NewtonFlow.hh"

using namespace MeshFEM;

template<size_t Dim, size_t FEMDeg, template<typename, size_t> class Psi_>
auto bindNewtonFlow(const std::string &name, py::module &m, py::module &detail) {
    using NFME = NewtonFlowMeshEnergy<Dim, FEMDeg, Psi_>;
    return bindMeshEnergy<NFME>(name, m, detail)
        .def("computeTaylorCoefficients", &NFME::computeTaylorCoefficients, py::arg("hessianFactorization"), py::arg("degree") = 6, py::arg("projectHessian") = false)
        .def("computeTaylorCoefficientsArclen", &NFME::computeTaylorCoefficientsArclen, py::arg("hessianFactorization"), py::arg("degree") = 6, py::arg("projectHessian") = false)
        .def("elementDeformationGradient", [](const NFME &me, size_t ei) {
            auto x = me.extractLocalVars(ei);
            EvalPt<Dim> q;
            q.fill(1.0 / (Dim + 1)); // sample at element center
            return me.elements[ei].deformationGradient(x, q);
        }, py::arg("ei"), "Get the (average) deformation gradient over element ei.")

        .def("elementHessianMinimumEigenvalues", &NFME::elementHessianMinimumEigenvalues)
        .def("elementGradientNorms", [](const NFME &me) { return me.elementGradientNorms(); })

        .def_property("projectionSmoothingEpsilon", &NFME::getProjectionSmoothingEpsilon, &NFME::setProjectionSmoothingEpsilon)
        .def_property("eigenvalueClampTarget", &NFME::getEigenvalueClampTarget, &NFME::setEigenvalueClampTarget)
        .def_property("eigenvalueProjectionModulation", &NFME::getEigenvalueProjectionModulation, &NFME::setEigenvalueProjectionModulation)
        .def_readwrite("remove_rigid_translation", &NFME::remove_rigid_translation)
        .def_readwrite("remove_rigid_rotation", &NFME::remove_rigid_rotation)
        .def_readonly("neg_delta_g", &NFME::neg_delta_g)
        ;
}

PYBIND11_MODULE(newton_flow, m)
{
    py::module::import("mesh_energy");
    py::module::import("py_newton_optimizer");
    py::module::import("rotation_strain_extrapolation");
    py::module detail = m.def_submodule("detail");

    bindNewtonFlow<2, 1, SymmetricDirichlet>("symmetric_dirichlet", m, detail);
    // bindNewtonFlow<2, 1, LinearElasticEnergy>("linear_elastic", m, detail);
}
