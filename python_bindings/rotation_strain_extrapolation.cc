#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include "../RotationStrainExtrapolation.hh"

PYBIND11_MODULE(rotation_strain_extrapolation, m) {
    using Base = rotation_strain_extrapolation::Extrapolator<double>;
    using Linear = rotation_strain_extrapolation::LinearExtrapolator<double>;
    using VXd = rotation_strain_extrapolation::VXd;
    using UVMat = rotation_strain_extrapolation::UVMat;

    py::class_<Base, std::shared_ptr<Base>>(m, "Extrapolator")
        .def("linesearch_begin", &Base::linesearch_begin, py::arg("x0"), py::arg("d"))
        .def("linesearch_eval", &Base::linesearch_eval, py::arg("alpha"))
        .def("eval_uvs", &Base::eval_uvs, py::arg("alphas"))
        .def(
            "__call__", &Base::operator(), py::arg("x0"), py::arg("coeffs"), py::arg("alphas"),
            R"pbdoc(
                Evaluate extrapolation along the ray x(alpha) = x0 + alpha * coeffs[0].

                Notes
                -----
                - `x0` and each entry in `coeffs` are flattened vectors with layout:
                  [u0, v0, u1, v1, ...].
                - Returned UV arrays are shape (N, 2).
            )pbdoc")
        .def_static(
            "flatten_view",
            [](const UVMat &uv) -> VXd { return Base::flatten_view(uv); },
            py::arg("uv"),
            R"pbdoc(
                Flatten UV (N, 2) to [u0, v0, u1, v1, ...].
            )pbdoc")
        .def_static("flatten", &Base::flatten, py::arg("uv"))
        .def_static(
            "unflatten_view",
            [](const VXd &x) -> UVMat { return Base::unflatten_view(x); },
            py::arg("x"),
            R"pbdoc(
                Unflatten [u0, v0, u1, v1, ...] to UV with shape (N, 2).
            )pbdoc")
        .def_static("unflatten", &Base::unflatten, py::arg("x"));

    py::class_<Linear, Base, std::shared_ptr<Linear>>(m, "LinearExtrapolator")
        .def(py::init<>());
}
