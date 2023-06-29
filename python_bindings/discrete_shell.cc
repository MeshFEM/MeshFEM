#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "../DiscreteShell.hh"
#include "../3rdparty//MeshFEM/src/python_bindings/MeshEntities.hh"

PYBIND11_MODULE(discrete_shell, m)
{
    using EO = ElasticObject<double>;
    using DS = DiscreteShell;

    using Mesh = DS::Mesh;

    py::class_<DS, EO, std::shared_ptr<DS>>(m, "DiscreteShell")
          .def_property_readonly_static("dimension", [](py::object /* self */) { return 3; })
          .def(py::init<const std::shared_ptr<Mesh> &, double, double>(),
                  py::arg("mesh"), py::arg("youngModulus") = 200, py::arg("poissonRatio") = 0.3)
          .def("mesh", [](const DS &ds) { return ds.mesh(); })
          .def("numHinges", &DS::numHinges)
          .def("visualizationGeometry", [](const DS &ds, double normalCreaseAngle) {
                Mesh visMesh(getF(ds.mesh()), ds.deformedPositions().topRows(ds.numVertices()));
                return getVisualizationGeometry(visMesh, normalCreaseAngle);
             }, py::arg("normalCreaseAngle") = M_PI)
          .def("visualizationField", [](const DS &ds, const Eigen::VectorXd &f) { return getVisualizationField(ds.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("visualizationField", [](const DS &ds, const Eigen::MatrixXd &f) { return getVisualizationField(ds.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def_readwrite("bendingStiffness", &DS::bendingStiffness)
          .def_readwrite("h",                &DS::h)
          ;

    py::module::import("elastic_object");
}
