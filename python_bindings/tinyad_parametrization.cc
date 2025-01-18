#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "../TinyADParametrization.hh"

PYBIND11_MODULE(tinyad_parametrization, m)
{
    py::module::import("MeshFEM");
    py::module::import("mesh");

    m.def("symmdsParamTinyAD", &TinyADParametrization::symmdsParamTinyAD, py::arg("mesh"), py::arg("uv"),
          "Symmetric Dirichlet Parametrization using TinyAD");
}