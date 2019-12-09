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
#include <MeshFEM/Utilities/MeshConversion.hh>
#include "MeshFactory.hh"

#include "MSHFieldWriter_bindings.hh"

////////////////////////////////////////////////////////////////////////////////
// Helper functions for extracting mesh entities
////////////////////////////////////////////////////////////////////////////////
// Gets the *volume* vertex indices making up a volume or boundary element.
template<class _EHandle, size_t... I>
Eigen::Matrix<int, sizeof...(I), 1> getElementCorners(const _EHandle &e, bool volumeIndices, Future::index_sequence<I...>) {
    constexpr size_t nv = _EHandle::numVertices();
    static_assert(sizeof...(I) == nv, "Incorrect index sequence length.");
    if (volumeIndices) return Eigen::Matrix<int, nv, 1>{e.vertex(I).volumeVertex().index()...};
    else               return Eigen::Matrix<int, nv, 1>{e.vertex(I).index()...};
}

template<class _HandleRange>
Eigen::Matrix<int, Eigen::Dynamic, _HandleRange::HType::numVertices()> getElementCorners(const _HandleRange &range, bool volumeIndices = true) {
    constexpr size_t nvPerElem = _HandleRange::HType::numVertices();
    Eigen::Matrix<int, Eigen::Dynamic, nvPerElem> elements(range.size(), nvPerElem);
    for (const auto& e : range)
        elements.row(e.index()) = getElementCorners(e, volumeIndices, Future::make_index_sequence<nvPerElem>());
    return elements;
}

template<class _Mesh, template<class> class _HType>
Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension>
getVertices(const HandleRange<_Mesh, _HType> &vrange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension> V(vrange.size(), size_t(_Mesh::EmbeddingDimension)); // size_t cast to prevent undefined symbol due to ODR-use
    for (const auto& v : vrange)
        V.row(v.index()) = v.volumeVertex().node()->p;
    return V;
}

template<class _Mesh, template<class> class _HType>
Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension>
getNodes(const HandleRange<_Mesh, _HType> &nrange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension> V(nrange.size(), size_t(_Mesh::EmbeddingDimension)); // size_t cast to prevent undefined symbol due to ODR-use
    for (const auto& n : nrange)
        V.row(n.index()) = n.volumeNode()->p;
    return V;
}

template<class _Mesh, template<class> class _HType>
typename std::enable_if<_Mesh::EmbeddingDimension == 3,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const HandleRange<_Mesh, _HType> &vrange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3> N(vrange.size(), 3);
    using V3d = Eigen::Matrix<typename _Mesh::Real, 3, 1>;
    for (auto v : vrange) {
        V3d n(V3d::Zero());
        for (auto he : v.incidentHalfEdges()) {
            auto t = he.tri();
            if (!t) continue;
            n += t->volume() * t->normal();
        }
        N.row(v.index()) = n.normalized();
    }
    return N;
}

// Vertex normals for meshes embedded in 2D are defined to be 3D vectors in the
// +z direction (this is needed for visualization).
template<class _Mesh, template<class> class _HType>
typename std::enable_if<_Mesh::EmbeddingDimension == 2,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const HandleRange<_Mesh, _HType> &vrange) {
    size_t nv = vrange.size();
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3> N(nv, 3);
    N.block(0, 0, nv, 2).setZero();
    N.block(0, 2, nv, 1).setOnes();
    return N;
}

// Normals for tri meshes
template<class _Mesh>
typename std::enable_if<_Mesh::K == 2, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const _Mesh &m) { return getAreaWeightedNormals(m.vertices()); }

// Surface normals for tet meshes
template<class _Mesh>
typename std::enable_if<_Mesh::K == 3, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const _Mesh &m) { return getAreaWeightedNormals(m.boundaryVertices()); }

template<class Mesh>
using MeshBindingsType = py::class_<Mesh, std::shared_ptr<Mesh>>;

// Geometry in the form expected by our triangle mesh viewer.
// Always a triangle mesh in 3D; this is either the boundary of a tet mesh or
// the original triangle mesh padded to when needed
using VisualizationGeometry = std::tuple<Eigen::Matrix<float,    Eigen::Dynamic, 3>,  // Pts
                                         Eigen::Matrix<uint32_t, Eigen::Dynamic, 3>,  // Tris
                                         Eigen::Matrix<float,    Eigen::Dynamic, 3>>; // Normals

template<class Mesh> typename std::enable_if<Mesh::K == 2, Eigen::Matrix<int, Eigen::Dynamic, 3>>::type getVisualizationTriangles(const Mesh &m) { return getElementCorners(m.elements()); }
template<class Mesh> typename std::enable_if<Mesh::K == 3, Eigen::Matrix<int, Eigen::Dynamic, 3>>::type getVisualizationTriangles(const Mesh &m) { return getElementCorners(m.boundaryElements(), false); }

template<class Mesh>
Eigen::Matrix<typename Mesh::Real, Eigen::Dynamic, 3> getVisualizationVertices(const Mesh &m) {
    Eigen::Matrix<typename Mesh::Real, Eigen::Dynamic, Eigen::Dynamic> dynamicResult;
    if (Mesh::K == 3) dynamicResult = getVertices(m.boundaryVertices());
    else              dynamicResult = getVertices(m.vertices());
    Eigen::Matrix<typename Mesh::Real, Eigen::Dynamic, 3> result(dynamicResult.rows(), 3);
    result. leftCols(    dynamicResult.cols()) = dynamicResult;
    result.rightCols(3 - dynamicResult.cols()).setZero();
    return result;
}

template<class Mesh>
VisualizationGeometry getVisualizationGeometry(const Mesh &m) {
    return VisualizationGeometry{getVisualizationVertices (m).template cast<float>(),
                                 getVisualizationTriangles(m).template cast<uint32_t>(),
                                 getAreaWeightedNormals   (m).template cast<float>()};
}

// Convert the field data to per-visualization-tri or per-visualization-vtx
// (NOP for triangle meshes, extract boundary data for tet meshes).
template<class Mesh, class FieldType>
Eigen::Matrix<typename FieldType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
getVisualizationField(const Mesh &m, const FieldType &field) {
    Eigen::Matrix<typename FieldType::Scalar, Eigen::Dynamic, Eigen::Dynamic> result;
    if (Mesh::K == 2) {
        size_t numValues = field.rows();
        // per-node fields are visualized as per-vertex fields
        if (numValues == m.numNodes())
            numValues = m.numVertices();
        size_t numComponents = field.cols();
        if (numComponents == 2)
            numComponents = 3; // pad 2D vectors to 3D
        result.resize(numValues, numComponents);
        result.leftCols(field.cols()) = field.topRows(numValues);
        int colsToPad = numComponents - field.cols();
        if (colsToPad > 0) result.rightCols(colsToPad).setZero();
        return result;
    }
    if (Mesh::K == 3) {
        if (size_t(field.rows()) == m.numVertices() || (size_t(field.rows()) == m.numNodes())) {
            result.resize(m.numBoundaryVertices(), field.cols());
            for (const auto &bv : m.boundaryVertices())
                result.row(bv.index()) = field.row(bv.volumeVertex().index());
        }
        else if (size_t(field.rows()) == m.numElements()) {
            result.resize(m.numBoundaryElements(), field.cols());
            for (const auto &be : m.boundaryElements()) {
                if (size_t(be.opposite().simplex().index()) >= size_t(field.rows())) throw std::runtime_error("out of bounds field");
                if (size_t(be.                     index()) >= size_t(result.rows())) throw std::runtime_error("out of bounds result");
                result.row(be.index()) = field.row(be.opposite().simplex().index());
            }
        }
        else throw std::runtime_error("Unexpected field size " + std::to_string(field.rows()));
        return result;
    }
    throw std::runtime_error("Unimplemented");
}

template<size_t _K, size_t _Degree, class _EmbeddingSpace>
struct MeshBindingsBase {
    using Mesh = FEMMesh<_K, _Degree, _EmbeddingSpace>;
    using Real = typename _EmbeddingSpace::Scalar;
    static constexpr size_t EmbeddingDimension = Mesh::EmbeddingDimension;
    using MXNd   = Eigen::Matrix<Real, Eigen::Dynamic, EmbeddingDimension>;
    using MX3d   = Eigen::Matrix<Real, Eigen::Dynamic,                  3>;
    using MXKp1i = Eigen::Matrix< int, Eigen::Dynamic, _K + 1>;

    static MeshBindingsType<Mesh> bind(py::module& module) {
        MeshBindingsType<Mesh> mb(module, getMeshName<Mesh>().c_str());
        // WARNING: Mesh's holder type is a shared_ptr; returning a unique_ptr will lead to a dangling pointer in the current version of Pybind11
        mb.def(py::init([](       const std::string &path) { return std::shared_ptr<Mesh>(Mesh::load(path)); }), py::arg("path"))
          .def(py::init([](const MXNd &V, const MXKp1i &F) { return std::make_shared<Mesh>(F, V);  }), py::arg("V"), py::arg("F"));
        if (EmbeddingDimension != 3) {
            // Also add a truncating constructor for 3D vertex arrays (if the mesh isn't embedded in 3D)
           mb.def(py::init([](const MX3d &V, const MXKp1i &F) { return std::make_shared<Mesh>(F, V);  }), py::arg("V"), py::arg("F"));
        }
        mb.def("vertices", [](const Mesh& m) { return getVertices(m.vertices()); })
          .def("nodes",    [](const Mesh& m) { return    getNodes(m.nodes()); })
          .def("setVertices", [](Mesh &m, MXNd &V) {
                  const size_t nv = V.rows();
                  if ((nv != m.numVertices()) && (nv != m.numNodes())) throw std::runtime_error("Incorrect vertex count");
                  m.setNodePositions(V);
               })
          .def("elements",         [](const Mesh &m) { return getElementCorners(m.elements()); })
          .def("boundaryElements", [](const Mesh &m) { return getElementCorners(m.boundaryElements()); })
          .def("boundaryVertices", [](const Mesh &m) {
                    Eigen::VectorXi result(m.numBoundaryVertices());
                    for (const auto &bv : m.boundaryVertices())
                        result(bv.index()) = bv.volumeVertex().index();
                    return result;
               })
          .def("elementsAdjacentBoundary", [](const Mesh &m) {
                  Eigen::VectorXi result(m.numBoundaryElements());
                  for (const auto &be : m.boundaryElements())
                      result[be.index()] = be.opposite().simplex().index();
                  return result;
              })

          .def("visualizationTriangles", &getVisualizationTriangles<Mesh>)
          .def("visualizationVertices",  &getVisualizationVertices <Mesh>)
          .def("visualizationGeometry",  &getVisualizationGeometry <Mesh>)
          .def("visualizationField", [](const Mesh &m, const Eigen::VectorXd &f) { return getVisualizationField(m, f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("visualizationField", [](const Mesh &m, const MXNd            &f) { return getVisualizationField(m, f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("vertexNormals", &getAreaWeightedNormals<Mesh>, (_K == 2) ? "Vertex normals (triangle area weighted)"
                                                                         : "Boundary vertex normals (triangle area weighted)")

          .def("numVertices", &Mesh::numVertices)
          .def("numElements", &Mesh::numElements)
          .def("numNodes",    &Mesh::numNodes)
          .def("save", [&](const Mesh &m, const std::string& path) { return MeshIO::save(path, m); })
          .def("field_writer", [&](const Mesh &m, const std::string &path) { return Future::make_unique<MSHFieldWriter>(path, m); }, py::arg("path"))
          .def("is_tet_mesh", [&](const Mesh &/* m */) { return _K == 3; })
          .def_property_readonly("bbox_volume", [](const Mesh& m) { return m.boundingBox().volume(); }, "bounding box volume")
          .def_property_readonly(     "volume", [](const Mesh& m) { return m.volume(); }, "mesh volume")
          .def_property_readonly_static("degree", [](py::object) { return _Degree; })
          .def_property_readonly_static("simplexDimension", [](py::object) { return _K; })
          .def_property_readonly_static("embeddingDimension", [](py::object) { return EmbeddingDimension; })
          ;
      return mb;
    }
};

template<size_t _Degree, class _EmbeddingSpace>
struct TriMeshSpecificBindings : public MeshBindingsBase<2, _Degree, _EmbeddingSpace> {
    using Base = MeshBindingsBase<2, _Degree, _EmbeddingSpace>;
    using Mesh = typename Base::Mesh;
    static MeshBindingsType<Mesh> bind(py::module& module) {
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
    using BoundaryMesh = FEMMesh<2, _Degree, _EmbeddingSpace>;
    static MeshBindingsType<Mesh> bind(py::module& module) {
        auto mesh_bindings = Base::bind(module);
        mesh_bindings
            .def("numTets",     &Mesh::numTets)
            .def("tets", [](const Mesh &m) { return getElementCorners(m.elements()); })
            .def("boundaryMesh", [](const Mesh &m) {
                        return std::make_shared<BoundaryMesh>(getElementCorners(m.boundaryElements(), false), getVertices(m.boundaryVertices()));
                }, "Get a triangle mesh of the boundary (copy)")
        ;
        return mesh_bindings;
    }
};

template<size_t _K, size_t _Degree, class _EmbeddingSpace>
struct MeshBindings;

template<size_t _Degree, class _EmbeddingSpace>
struct MeshBindings<2, _Degree, _EmbeddingSpace> : public TriMeshSpecificBindings<_Degree, _EmbeddingSpace> { };

template<size_t _Degree, class _EmbeddingSpace>
struct MeshBindings<3, _Degree, _EmbeddingSpace> : public TetMeshSpecificBindings<_Degree, _EmbeddingSpace> { };

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
        }, py::arg("path"), py::arg("degree") = 1, py::arg("embeddingDimension") = 0);
    m.def("Mesh", [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, size_t degree, size_t embeddingDimension) {
            size_t K = F.cols() - 1;
            if ((K < 2) || (K > 3)) throw std::runtime_error("Mesh must be triangle or tet.");

            if (embeddingDimension == 0)
                embeddingDimension = V.cols();

            std::vector<MeshIO::IOVertex > vertices;
            std::vector<MeshIO::IOElement> elements;
            std::tie(vertices, elements) = getMeshIO(V, F);

            return MeshFactory<double>(elements, vertices, K, degree, embeddingDimension);
        }, py::arg("V"), py::arg("F"), py::arg("degree") = 1, py::arg("embeddingDimension") = 0);

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
                    return std::make_shared<FEMMesh<2, 1, Eigen::Vector2d>>(
                            triangulation.getElements(),
                            triangulation.getVertices());
                })
        .def("getQuadraticMesh", [](const PSetTriangulation& triangulation)
                {
                    return std::make_shared<FEMMesh<2, 2, Eigen::Vector2d>>(
                            triangulation.getElements(),
                            triangulation.getVertices());
                });
}
