#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "../TinyADParametrization.hh"

using namespace MeshFEM;

PYBIND11_MODULE(tinyad_parametrization, m)
{
    py::module::import("MeshFEM");
    py::module::import("mesh");

    m.def("symmdsParamTinyAD", &TinyADParametrization::symmdsParamTinyAD, py::arg("mesh"), py::arg("uv"),
            py::arg("max_iters") = 1000,
            py::arg("convergence_eps") = 1e-2,
            py::arg("saveUV") = false,
            py::arg("filepath") = "",
            py::arg("proj_eps") = TinyAD::default_hessian_projection_eps,
          "Symmetric Dirichlet Parametrization using TinyAD");

    m.def("symmdsParamTinyADEvalFGH", &TinyADParametrization::symmdsParamTinyADEvalFGH,
            py::arg("mesh"), py::arg("x"), py::arg("project") = true, py::arg("proj_eps") = TinyAD::default_hessian_projection_eps,
            "Evaluate energy, gradient, and projected Hessian at variables `x` using TinyAD");

    m.def("paramTADMeshFEMHybrid", &paramTADMeshFEMHybrid<TinyADParametrization::Mesh>,
            py::arg("mesh"), py::arg("uv"),
            py::arg("max_iters") = 1000,
            py::arg("convergence_eps") = 1e-2,
            py::arg("proj_eps") = TinyAD::default_hessian_projection_eps,
          "Symmetric Dirichlet Parametrization using MeshFEM's energy/gradient/Hessian with the Newton loop of symmdsParamTinyAD");
}
