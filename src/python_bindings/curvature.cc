#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <MeshFEM/Curvature.hh>

#include <tuple>

namespace py = pybind11;

template<size_t _Degree>
void bindCurvature(py::module &m, py::module &detail_module) {
    using V3d = Eigen::Matrix<double, 3, 1>;
    using Mesh = FEMMesh<2, _Degree, V3d>;

    using GCS = GaussianCurvatureSensitivity<Mesh>;

    py::class_<GCS>(detail_module, ("GaussianCurvatureSensitivity" + std::to_string(_Degree)).c_str())
        .def(py::init<const Mesh &>(), py::arg("mesh"))
        .def("mesh",              &GCS::mesh, py::return_value_policy::reference)
        .def("voronoiAreas",      &GCS::voronoiAreas)
        .def("mixedVoronoiAreas", &GCS::mixedVoronoiAreas)
        .def("integratedK",       &GCS::integratedK)
        .def("K",                 &GCS::K)
        .def("deltaVoronoiAreas", &GCS::deltaVoronoiAreas, py::arg("deltaP"), py::arg("mixed") = false)
        .def("deltaIntegratedK",  &GCS::deltaIntegratedK,  py::arg("deltaP"))
        .def("deltaK",            &GCS::deltaK,            py::arg("deltaP"))
        ;

    m.def("GaussianCurvatureSensitivity", [](const Mesh &mesh) { return std::make_unique<GCS>(mesh); });
}

PYBIND11_MODULE(curvature, m) {
    m.doc() = "Curvature and Curvature Shape Derivative Calculation";

    py::module::import("mesh");
    py::module detail_module = m.def_submodule("detail");

    bindCurvature<1>(m, detail_module); // bindings for linear    tri meshes in 3d
    bindCurvature<2>(m, detail_module); // bindings for quadratic tri meshes in 3d
}
