#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <MeshFEM/filters/extract_component_polygons.hh>
#include <MeshFEM/filters/reflect.hh>
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

    // Row major for compatibility with numpy
    using VType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using FType = Eigen::Matrix<int,    Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    m.def("reflect", [](const VType &V, const VType &F, const std::string &components) {
            std::vector<MeshIO::IOVertex > rvertices;
            std::vector<MeshIO::IOElement> relements;
            ComponentMask mask(components);
            size_t dim;
            if (V.cols() == 2) dim = 2;
            else if (V.cols() == 3) {
                dim = (V.col(2).maxCoeff() - V.col(2).minCoeff() > 1e-6) ? 3 : 2;
            }
            else throw std::runtime_error("Mesh must be 2 or 3 dimensional");
            reflect(dim, getMeshIOVertices(V), getMeshIOElements(F), rvertices, relements, mask);
            return std::make_pair(getV(rvertices), getF(relements));
        }, py::arg("V"), py::arg("F"), py::arg("components") = "xyz");

    generateMeshSpecificBindings(m, detail_module, ExtractComponentPolygonsBinder());
}
