#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include "../ContinuationParametrization.hh"

PYBIND11_MODULE(continuation_parametrization, m)
{
    py::module::import("mesh_energy");
    py::module detail = m.def_submodule("detail");

    using Mesh = FEMMesh<2, 1, Vector3D>;
    using Vars = NodalVars<2>;
    using Stencil = ElementStencil</* K = */ 2, /* Deg = */ 1, /* N = */ 2>;

    bindMeshEnergy<ContinuationParamMeshEnergy>("symmetric_dirichlet_param", m, detail)
        .def("setInterpolatedReference", &ContinuationParamMeshEnergy::setInterpolatedReference, py::arg("lambda"), py::arg("x"))
        .def("computeTaylorCoefficients", &ContinuationParamMeshEnergy::computeTaylorCoefficients, py::arg("hessianFactorization"), py::arg("degree") = 6)
        .def("computeTaylorCoefficientsArclen", &ContinuationParamMeshEnergy::computeTaylorCoefficientsArclen, py::arg("hessianFactorization"), py::arg("degree") = 6)
        ;
}
