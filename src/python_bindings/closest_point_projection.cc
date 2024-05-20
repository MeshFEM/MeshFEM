#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <MeshFEM/Types.hh>
#include <MeshFEM/ClosestPointProjection.hh>
#include <MeshFEM/FEMMesh.hh>

namespace py = pybind11;

template<size_t D>
void bind(py::module &detail) {
    using CP = ClosestPointProjection<VecN_T<Real, D>>;
    py::class_<CP, std::shared_ptr<CP>> pyCP(detail, ("ClosestPointProjection" + std::to_string(D) + "D").c_str());
    pyCP.def("project", &CP::project, py::arg("q"), py::arg("computeJacobian") = true)
        .def("numVertices",       &CP::numVertices)
        .def("numElements",       &CP::numElements)
        .def("numElementCorners", &CP::numElementCorners)
        .def_property_readonly("V", &CP::V)
        .def_property_readonly("F", &CP::F)
        ;

    using CPR = typename CP::ProjectionResult;
    py::class_<CPR>(pyCP, "Result")
        .def_readonly("p",          &CPR::p)
        .def_readonly("element",    &CPR::element)
        .def_readonly("barycoords", &CPR::barycoords)
        .def_readonly("dp_dq",      &CPR::dp_dq)
        ;
}

template<class Mesh>
void bindConstructor(py::module &m) {
    m.def("ClosestPointProjection", [](const Mesh &mesh) -> py::object {
        return py::cast(new ClosestPointProjection<typename Mesh::EmbeddingSpace>(getV(mesh), getF(mesh)), py::return_value_policy::take_ownership);
    }, py::arg("mesh"));
}

PYBIND11_MODULE(closest_point_projection, m) {
    m.doc() = "Fast closest-point projection based on AABB data structure";
    py::module detail_module = m.def_submodule("detail");

    bind<2>(detail_module);
    bind<3>(detail_module);

    m.def("ClosestPointProjection", [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) -> py::object {
        if (V.cols() == 2)      return py::cast(new ClosestPointProjection<VecN_T<Real, 2>>(V, F), py::return_value_policy::take_ownership);
        else if (V.cols() == 3) return py::cast(new ClosestPointProjection<VecN_T<Real, 3>>(V, F), py::return_value_policy::take_ownership);
        else throw std::runtime_error("Only 2D and 3D data are supported");
    }, py::arg("V"), py::arg("F"));

    py::module::import("mesh");

    // Linear/Quadratic FEM meshes in 2D and 3D
    bindConstructor<FEMMesh<2, 1, Vector2D>>(m);
    bindConstructor<FEMMesh<2, 2, Vector2D>>(m);
    bindConstructor<FEMMesh<2, 1, Vector3D>>(m);
    bindConstructor<FEMMesh<2, 2, Vector3D>>(m);
}
