#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <MeshFEM/filters/extract_component_polygons.hh>
#include "BindingInstantiations.hh"

namespace py = pybind11;

struct ExtractComponentPolygonsBinder {
    template<class Mesh>
    static std::enable_if_t<Mesh::K == 2>
    bind(py::module &module, py::module &/* detail_module */) {
        module.def("extract_component_polygons", &extract_component_polygons<Mesh>, py::arg("mesh"), py::arg("indicator"));
    }

    template<class Mesh>
    static std::enable_if_t<Mesh::K == 3>
    bind(py::module &, py::module &) { /* NOP */ }
};

PYBIND11_MODULE(filters, m) {
    m.doc() = "Miscellaneous filters/operations that can be performed on meshes.";

    py::module detail_module = m.def_submodule("detail");

    py::class_<IdxPolygon>(detail_module, "IdxPolygon")
        .def_readonly("exterior", &IdxPolygon::exterior, "Indices of mesh vertices making up the polygon's exterior boundary")
        .def_readonly("holes",    &IdxPolygon::holes,     "List of indices of mesh vertices making up each hole boundary (if any)")
        ;

    generateMeshSpecificBindings(m, detail_module, ExtractComponentPolygonsBinder());
}
