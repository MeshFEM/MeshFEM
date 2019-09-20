#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <Eigen/Dense>
#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Meshing.hh>

#include <MeshFEM/Utilities/NameMangling.hh>
#include "MeshFactory.hh"

template<size_t _K, size_t _Degree, class _EmbeddingSpace>
void
bindMesh(py::module& module)
{
    using Mesh = FEMMesh<_K, _Degree, _EmbeddingSpace>;
    using Real = typename _EmbeddingSpace::Scalar;
    constexpr size_t EmbeddingDimension = _EmbeddingSpace::RowsAtCompileTime;
    using MXNd = Eigen::Matrix<Real, Eigen::Dynamic, EmbeddingDimension>;

    py::class_<Mesh> mesh_bindings(module, getMeshName<Mesh>().c_str());
    mesh_bindings
      .def(py::init([](const std::string& path) {
               std::vector<MeshIO::IOVertex> vertices;
               std::vector<MeshIO::IOElement> elements;
               MeshIO::load(path, vertices, elements);
               return Mesh(elements, vertices);
           }),
           py::arg("path"))
      .def("vertices",
           [&](const Mesh& m) {
               MXNd V(m.numVertices(), _K);
               for (const auto& v : m.vertices())
                   V.row(v.index()) = v.node()->p;
               return V;
           })
      .def("setVertices", [](Mesh &m, MXNd &V) {
              const size_t nv = m.numVertices();
              if (size_t(V.rows()) != nv) throw std::runtime_error("Incorrect vertex count");
              m.setNodePositions(V);
           })
      .def("elements",
           [&](const Mesh& m) {
               std::vector<std::array<size_t, _K + 1>> elements;
               elements.reserve(m.numElements());
               std::array<size_t, _K + 1> current_element;
               for (const auto& e : m.elements()) {
                   for (const auto& v : e.vertices())
                       current_element[v.localIndex()] = v.index();
                   elements.push_back(current_element);
               }
               return elements;
           })
      .def("boundaryElements",
           [&](const Mesh& m) {
               std::vector<std::array<size_t, _K>> elements;
               elements.reserve(m.numBoundaryElements());
               std::array<size_t, _K> current_element;
               for (const auto& be : m.boundaryElements()) {
                   for (const auto& bv : be.vertices())
                       current_element[bv.localIndex()] = bv.volumeVertex().index();
                   elements.push_back(current_element);
               }
               return elements;
           })
      .def("boundaryVertices", [](const Mesh &m) {
                    std::vector<size_t> result;
                    for (const auto &bv : m.boundaryVertices())
                        result.push_back(bv.volumeVertex().index());
                    return result;
           })
      .def("numVertices", &Mesh::numVertices)
      .def("numElements", &Mesh::numElements)
      .def("numNodes",    &Mesh::numNodes)
      .def("save", [&](const Mesh& m, const std::string& path) { return MeshIO::save(path, m); })
      .def("is_tet_mesh", [&](const Mesh& m) { return _K == 3; })
      .def_property_readonly("bbox_volume", [](const Mesh& m) { return m.boundingBox().volume(); }, "bounding box volume")
      .def_property_readonly(     "volume", [](const Mesh& m) { return m.volume(); }, "mesh volume")
      .def_property_readonly_static("degree", [](py::object) { return _Degree; })
      .def_property_readonly_static("simplexDimension", [](py::object) { return _K; })
      .def_property_readonly_static("embeddingDimension", [](py::object) { return EmbeddingDimension; })
      ;
}

template<size_t _Dimension>
void
bindPeriodicCondition(py::module& module)
{
    using PC = PeriodicCondition<_Dimension>;
    using LinearMesh    = FEMMesh<_Dimension, 1, Eigen::Matrix<double, _Dimension, 1>>;
    using QuadraticMesh = FEMMesh<_Dimension, 2, Eigen::Matrix<double, _Dimension, 1>>;

    module.def("PeriodicCondition", [](const LinearMesh    &m, double eps, bool ignore_mismatch, const std::vector<size_t> &ignore_dims) { return std::make_shared<PC>(m, eps, ignore_mismatch, ignore_dims); }, py::arg("mesh"), py::arg("eps") = 1e-7, py::arg("ignore_mismatch") = false, py::arg("ignore_dims") = std::vector<size_t>());
    module.def("PeriodicCondition", [](const QuadraticMesh &m, double eps, bool ignore_mismatch, const std::vector<size_t> &ignore_dims) { return std::make_shared<PC>(m, eps, ignore_mismatch, ignore_dims); }, py::arg("mesh"), py::arg("eps") = 1e-7, py::arg("ignore_mismatch") = false, py::arg("ignore_dims") = std::vector<size_t>());

    module.def("PeriodicCondition", [](const LinearMesh    &m, const std::string &path) { return std::make_shared<PC>(m, path); }, py::arg("mesh"), py::arg("periodic_condition_file"));
    module.def("PeriodicCondition", [](const QuadraticMesh &m, const std::string &path) { return std::make_shared<PC>(m, path); }, py::arg("mesh"), py::arg("periodic_condition_file"));

    // We use a shared_ptr holder to support using PeriodicCondition instances
    // as optionally "None" arguments
    py::class_<PeriodicCondition<_Dimension>, std::shared_ptr<PeriodicCondition<_Dimension>>>(
      module, ("PeriodicCondition" + std::to_string(_Dimension) + "D").c_str())
      .def("periodicDoFsForNodes", &PeriodicCondition<_Dimension>::periodicDoFsForNodes);
}

template<typename _Real>
void addMeshBindings(py::module &m) {
    using V3d = Eigen::Matrix<_Real, 3, 1>;
    using V2d = Eigen::Matrix<_Real, 2, 1>;

    bindMesh<3, 1, V3d>(m); // linear    tet mesh in 3d
    bindMesh<3, 2, V3d>(m); // quadratic tet mesh in 3d

    bindMesh<2, 1, V2d>(m); // linear    tri mesh in 2d
    bindMesh<2, 2, V2d>(m); // quadratic tri mesh in 2d
    bindMesh<2, 1, V3d>(m); // linear    tri mesh in 3d
    bindMesh<2, 2, V3d>(m); // quadratic tri mesh in 3d
}

PYBIND11_MODULE(mesh, m)
{
    addMeshBindings<double>(m);
#if MESHFEM_BIND_LONG_DOUBLE
    addMeshBindings<long double>(m);
#endif

    bindPeriodicCondition<2>(m);
    bindPeriodicCondition<3>(m);

    // Mesh "Factory" function for dynamically creating an instance of the appropriate FEMMesh instantiation.
    m.def("Mesh", [](const std::string &path, size_t degree, size_t embeddingDimension) {
            std::vector<MeshIO::IOVertex > vertices;
            std::vector<MeshIO::IOElement> elements;
            auto type = MeshIO::load(path, vertices, elements, MeshIO::FMT_GUESS, MeshIO::MESH_GUESS);

            // Infer simplex dimension from mesh type.
            size_t K;
            if      (type == MeshIO::MESH_TET) K = 3;
            else if (type == MeshIO::MESH_TRI) K = 2;
            else    throw std::runtime_error("Mesh must be pure triangle or tet.");

            // Default to 2D embedding for triangle meshes, 3D embedding for tet meshes if unspecified
            if (embeddingDimension == 0)
                embeddingDimension = K;
            py::object result = MeshFactory<double>(elements, vertices, K, degree, embeddingDimension);
            return result;
        }, py::arg("path"), py::arg("degree"), py::arg("embeddingDimension") = 0);

    using PSetTriangulation = PolygonSetTriangulation<
        double, Eigen::Vector2d, std::pair<size_t, size_t>>;

    py::class_<PSetTriangulation>(m, "PolygonSetTriangulation")
        .def(py::init<
                const std::vector<Eigen::Vector2d>&,
                const std::vector<std::vector<std::pair<size_t, size_t>>>&,
                const std::vector<Eigen::Vector2d>&,
                double, double>())
        .def("getLinearMesh", [](const PSetTriangulation& triangulation)
                {
                    return std::make_unique<FEMMesh<2, 1, Eigen::Vector2d>>(
                            triangulation.getElements(),
                            triangulation.getVertices());
                })
        .def("getQuadraticMesh", [](const PSetTriangulation& triangulation)
                {
                    return std::make_unique<FEMMesh<2, 2, Eigen::Vector2d>>(
                            triangulation.getElements(),
                            triangulation.getVertices());
                });
}
