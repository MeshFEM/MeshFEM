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
                            const std::vector<std::pair<size_t, size_t>> &edges,
                            const std::vector<Point2D> &holePts,
                            const double triArea,
                            const bool omitQualityFlag,
                            const std::string &flags,
                            bool outputPointMarkers,
                            bool outputEdgeMarkers) {
            std::vector<MeshIO::IOVertex > vertices;
            std::vector<MeshIO::IOElement> elements;
            std::vector<int> pointMarkers;
            std::vector<std::array<int, 2>> edgeMarkers;
            triangulatePSLG(pts, edges, holePts,
                            vertices, elements, triArea, flags,
                            &pointMarkers,
                            &edgeMarkers,
                            omitQualityFlag);
            py::list l;
            l.append(getV(vertices));
            l.append(getF(elements));
            if (outputPointMarkers) {
                Eigen::VectorXi pm = Eigen::Map<Eigen::VectorXi>(pointMarkers.data(), pointMarkers.size());
                l.append(pm);
            }
            if (outputEdgeMarkers) {
                const size_t nem = edgeMarkers.size();
                Eigen::MatrixX2i em(nem, 2);
                for (size_t i = 0; i < nem; ++i) {
                    em(i, 0) = edgeMarkers[i][0];
                    em(i, 1) = edgeMarkers[i][1];
                }
                l.append(em);
            }
            return py::tuple(l);
        }, py::arg("pts"), py::arg("edges"), py::arg("holePts") = std::vector<Point2D>(), py::arg("triArea") = 0.01, py::arg("omitQualityFlag") = false, py::arg("flags") = "", py::arg("outputPointMarkers") = true, py::arg("outputEdgeMarkers") = false);
}
