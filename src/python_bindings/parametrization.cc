#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/Parametrization.hh>

PYBIND11_MODULE(parametrization, m)
{
    py::module::import("mesh");
    py::module::import("sparse_matrices");

    py::enum_<Parametrization::SCPInnerProduct>(m, "SCPInnerProduct")
        .value("I_B",         Parametrization::SCPInnerProduct::I_B,   "Identity matrix restricted to the boundary variables (corresponds to Euclidean norm of the vector of boundary variables)")
        .value("Mass",        Parametrization::SCPInnerProduct::Mass,  "Mass matrix (corresponds to the L2 norm of the piecewise linear mapping function)")
        .value("BMass",       Parametrization::SCPInnerProduct::BMass, "Boundary mass matrix (corresponds to the L2 norm of the restriction of the piecewise linear mapping function to the boundary)")
        ;

    // Parametrization algorithms
    m.def("harmonic", &Parametrization::harmonic, py::arg("mesh"), py::arg("boundaryPositions"), "Harmonic Parametrization");
    m.def("lscm",     &Parametrization::lscm,     py::arg("mesh"), py::arg("initParam") = Parametrization::UVMap(), "Least-Squares Conformal Parametrization");
    m.def("scp",      &Parametrization::scp,      py::arg("mesh"),
            py::arg("iprod") = Parametrization::SCPInnerProduct::Mass,
            py::arg("eps") = 1e-12,
           "Spectral Conformal Parametrization");

    // Mapping analysis
    m.def("scaleFactor", &Parametrization::scaleFactor, py::arg("mesh"), py::arg("uv"),
          "Get factor by which lengths are stretched when mapping from 2D to 3D");
    m.def("conformalDistortion", &Parametrization::conformalDistortion, py::arg("mesh"), py::arg("uv"),
          "Get the (quasi-)conformal distortion strain measure (sigma_0 - sigma_1) / sigma_1 >= 0");

    // Misc utilities
    m.def("rescale", &Parametrization::rescale, py::arg("mesh"), py::arg("uv"),
          "Globally scale the parametrization to minimise the total squared difference in triangle areas caused by flattening");

    // Matrix assembly
    m.def("assembleLSCMMatrix", &Parametrization::assembleLSCMMatrix, py::arg("mesh"), "Matrix defining the quadratic discrete least-squares conformal energy");
    m.def("assembleBMatrixSCP", &Parametrization::assembleBMatrixSCP, py::arg("mesh"), "\"boundary indicator\" B matrix defined in SCP");
    m.def("assembleMassMatrix", &Parametrization::assembleMassMatrix, py::arg("mesh"), "Mass matrix for the UV problem");
}
