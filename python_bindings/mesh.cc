#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <Eigen/Dense>
#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Meshing.hh>
#include <MeshFEM/MSHFieldWriter.hh>

#include <MeshFEM/Utilities/NameMangling.hh>
#include "MeshFactory.hh"

#include "MSHFieldWriter_bindings.hh"

template<size_t _K, class Mesh>
void addDimensionSpecificBindings(py::class_<Mesh> &mesh_bindings);

// Gets the *volume* vertex indices making up a volume or boundary element.
template<class _EHandle, size_t... I>
std::array<size_t, sizeof...(I)> getElementCorners(const _EHandle &e, Future::index_sequence<I...>) {
    static constexpr size_t nv = _EHandle::numVertices();
    static_assert(sizeof...(I) == nv, "Incorrect index sequence length.");
    return std::array<size_t, _EHandle::numVertices()>{{size_t(e.vertex(I).volumeVertex().index())...}};
}

template<class _Mesh, template<class> class _HType>
std::vector<std::array<size_t, _HType<_Mesh>::numVertices()>> getElementCorners(const HandleRange<_Mesh, _HType> &range) {
    std::vector<std::array<size_t, _HType<_Mesh>::numVertices()>> elements;
    elements.reserve(range.size());
    for (const auto& e : range)
        elements.emplace_back(getElementCorners(e, Future::make_index_sequence<e.numVertices()>()));
    return elements;
}

template<size_t _K, size_t _Degree, class _EmbeddingSpace>
struct MeshBindingsBase {
    using Mesh = FEMMesh<_K, _Degree, _EmbeddingSpace>;
    using Real = typename _EmbeddingSpace::Scalar;
    static constexpr size_t EmbeddingDimension = _EmbeddingSpace::RowsAtCompileTime;
    using MXNd = Eigen::Matrix<Real, Eigen::Dynamic, EmbeddingDimension>;

    static py::class_<Mesh> bind(py::module& module) {
        return py::class_<Mesh>(module, getMeshName<Mesh>().c_str())
          .def(py::init([](const std::string& path) {
                   std::vector<MeshIO::IOVertex> vertices;
                   std::vector<MeshIO::IOElement> elements;
                   MeshIO::load(path, vertices, elements);
                   return Mesh(elements, vertices);
               }),
               py::arg("path"))
          .def("vertices",
               [](const Mesh& m) {
                   MXNd V(m.numVertices(), EmbeddingDimension);
                   for (const auto& v : m.vertices())
                       V.row(v.index()) = v.node()->p;
                   return V;
               })
          .def("setVertices", [](Mesh &m, MXNd &V) {
                  const size_t nv = m.numVertices();
                  if (size_t(V.rows()) != nv) throw std::runtime_error("Incorrect vertex count");
                  m.setNodePositions(V);
               })
          .def("elements",         [](const Mesh &m) { return getElementCorners(m.elements()); })
          .def("boundaryElements", [](const Mesh &m) { return getElementCorners(m.boundaryElements()); })
          .def("boundaryVertices", [](const Mesh &m) {
                    std::vector<size_t> result;
                    for (const auto &bv : m.boundaryVertices())
                        result.push_back(bv.volumeVertex().index());
                    return result;
               })
          .def("numVertices", &Mesh::numVertices)
          .def("numElements", &Mesh::numElements)
          .def("numNodes",    &Mesh::numNodes)
          .def("save", [&](const Mesh &m, const std::string& path) { return MeshIO::save(path, m); })
          .def("field_writer", [&](const Mesh &m, const std::string &path) { return Future::make_unique<MSHFieldWriter>(path, m); }, py::arg("path"))
          .def("is_tet_mesh", [&](const Mesh &m) { return _K == 3; })
          .def_property_readonly("bbox_volume", [](const Mesh& m) { return m.boundingBox().volume(); }, "bounding box volume")
          .def_property_readonly(     "volume", [](const Mesh& m) { return m.volume(); }, "mesh volume")
          .def_property_readonly_static("degree", [](py::object) { return _Degree; })
          .def_property_readonly_static("simplexDimension", [](py::object) { return _K; })
          .def_property_readonly_static("embeddingDimension", [](py::object) { return EmbeddingDimension; })
          ;
    }
};

template<size_t _Degree, class _EmbeddingSpace>
struct TriMeshSpecificBindings : public MeshBindingsBase<2, _Degree, _EmbeddingSpace> {
    using Base = MeshBindingsBase<2, _Degree, _EmbeddingSpace>;
    using Mesh = typename Base::Mesh;
    static py::class_<Mesh> bind(py::module& module) {
        auto mesh_bindings = Base::bind(module);
        mesh_bindings
            .def("numTris",     &Mesh::numTris)
            .def("triangles", [](const Mesh &m) { return getElementCorners(m.elements()); })
            .def("trisAdjTri", [](const Mesh &m, size_t ti) {
                    std::vector<int> result;
                    if (ti >= m.numTris()) throw std::runtime_error("Triangle index out of bounds");
                    for (const auto &tri_j : m.tri(ti).neighbors()) {
                        if (!tri_j) continue;
                        result.push_back(tri_j.index());
                    }
                    return result;
                })
            .def("vtsAdjVtx", [](const Mesh &m, size_t vi) {
                    std::vector<int> result;
                    if (vi >= m.numVertices()) throw std::runtime_error("Vertex index out of bounds");
                    for (const auto &he : m.vertex(vi).incidentHalfEdges())
                        result.push_back(he.tail().index());
                    return result;
                })
            .def("valences", [](const Mesh &m) {
                    std::vector<int> result(m.numVertices());
                    for (const auto &tri : m.elements()) {
                        for (const auto &v : tri.vertices())
                            ++result[v.index()];
                    }
                    return result;
                })
        ;
        return mesh_bindings;
    }
};

template<size_t _Degree, class _EmbeddingSpace>
struct TetMeshSpecificBindings : public MeshBindingsBase<3, _Degree, _EmbeddingSpace> {
    using Base = MeshBindingsBase<3, _Degree, _EmbeddingSpace>;
    using Mesh = typename Base::Mesh;
    static py::class_<Mesh> bind(py::module& module) {
        auto mesh_bindings = Base::bind(module);
        mesh_bindings
            .def("numTets",     &Mesh::numTets)
            .def("tets", [](const Mesh &m) { return getElementCorners(m.elements()); });
        return mesh_bindings;
    }
};

template<size_t _K, size_t _Degree, class _EmbeddingSpace>
struct MeshBindings;

template<size_t _Degree, class _EmbeddingSpace>
struct MeshBindings<2, _Degree, _EmbeddingSpace> : public TriMeshSpecificBindings<_Degree, _EmbeddingSpace> { };

template<size_t _Degree, class _EmbeddingSpace>
struct MeshBindings<3, _Degree, _EmbeddingSpace> : public TetMeshSpecificBindings<_Degree, _EmbeddingSpace> { };

// Triangle meshes in 3D also provide normals.
template<size_t _Degree, class _Real>
struct MeshBindings<2, _Degree, Eigen::Matrix<_Real, 3, 1>> : public TriMeshSpecificBindings<_Degree, Eigen::Matrix<_Real, 3, 1>> {
    using Base = MeshBindingsBase<2, _Degree, Eigen::Matrix<_Real, 3, 1>>;
    using Mesh = typename Base::Mesh;
    using V3d = Eigen::Matrix<_Real, 3, 1>;
    static py::class_<Mesh> bind(py::module& module) {
        auto mesh_bindings = Base::bind(module);
        mesh_bindings
            .def("vertexNormals", [](const Mesh &m) {
                    Eigen::Matrix<_Real, Eigen::Dynamic, 3> N(m.numVertices(), 3);
                    for (auto v : m.vertices()) {
                        V3d n(V3d::Zero());
                        for (auto he : v.incidentHalfEdges()) {
                            auto t = he.tri();
                            if (!t) continue;
                            n += t->volume() * t->normal();
                        }
                        N.row(v.index()) = n.normalized();
                    }
                    return N;
                }, "Vertex normals (triangle area weighted average)")
            ;
        return mesh_bindings;
    }
};

template<size_t _Dimension>
void bindPeriodicCondition(py::module& module)
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

    MeshBindings<3, 1, V3d>::bind(m); // linear    tet mesh in 3d
    MeshBindings<3, 2, V3d>::bind(m); // quadratic tet mesh in 3d

    MeshBindings<2, 1, V2d>::bind(m); // linear    tri mesh in 2d
    MeshBindings<2, 2, V2d>::bind(m); // quadratic tri mesh in 2d
    MeshBindings<2, 1, V3d>::bind(m); // linear    tri mesh in 3d
    MeshBindings<2, 2, V3d>::bind(m); // quadratic tri mesh in 3d
}

PYBIND11_MODULE(mesh, m)
{
    m.doc() = "MeshFEM finite element mesh data structure bindings";

    bindMSHFieldWriter(m);

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

            // Default to 2D embedding for triangle meshes, 3D embedding for tet meshes if unspecified,
            // but upgrade to 3D if any z components are nonzero.
            if (embeddingDimension == 0) {
                embeddingDimension = K;
                for (const auto &v : vertices)
                    if (std::abs(v[2]) > 1e-10) embeddingDimension = 3;
            }
            return MeshFactory<double>(elements, vertices, K, degree, embeddingDimension);
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
