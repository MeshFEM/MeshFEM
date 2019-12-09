#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/Triangulate.h>
#include <MeshFEM/Utilities/MeshConversion.hh>

#include <tuple>

namespace py = pybind11;

PYBIND11_MODULE(triangulation, m) {
    m.doc() = "Triangulation of line segments";

    m.def("triangulate", [](const std::vector<Eigen::Vector2d> &pts,
                            std::vector<std::pair<size_t, size_t>> &edges,
                            double triArea,
                            const std::string &flags) {
            std::vector<MeshIO::IOVertex > vertices;
            std::vector<MeshIO::IOElement> elements;
            std::vector<int> pointMarkers;
            triangulatePSLC(pts, edges, std::vector<Point2D>(),
                            vertices, elements, triArea, flags, &pointMarkers);
            return py::make_tuple(getV(vertices), getF(elements), pointMarkers);
        }, py::arg("pts"), py::arg("edges"), py::arg("triArea") = 0.01, py::arg("flags") = "");
}
