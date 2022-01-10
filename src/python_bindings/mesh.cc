#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include "BindingInstantiations.hh"

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
#include "MSHFieldParser_bindings.hh"

#include "MeshEntities.hh"

template<class Mesh>
struct MeshBindingsBase {
    using Real = typename Mesh::Real;
    static constexpr size_t K = Mesh::K;
    static constexpr size_t EmbeddingDimension = Mesh::EmbeddingDimension;
    using MXNd   = Eigen::Matrix<Real, Eigen::Dynamic, EmbeddingDimension>;
    using MX3d   = Eigen::Matrix<Real, Eigen::Dynamic,     3>;
    using MXKp1i = Eigen::Matrix< int, Eigen::Dynamic, K + 1>;

    MeshBindingsBase() { }
    ~MeshBindingsBase() { }

    static MeshBindingsType<Mesh> bind(py::module &/* module */, py::module &detail_module) {
        MeshBindingsType<Mesh> mb(detail_module, NameMangler<Mesh>::name().c_str());
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
          .def("elements",            [](const Mesh &m) { return getElementCorners(m.elements()); })
          .def("boundaryElements",    [](const Mesh &m) { return getElementCorners(m.boundaryElements()); })
          .def("elementNodes",        [](const Mesh &m) { return getElementNodes(m.elements()); })
          .def("boundaryElementNodes",[](const Mesh &m) { return getElementNodes(m.boundaryElements()); })
          .def("boundaryVertices", [](const Mesh &m) {
                    Eigen::VectorXi result(m.numBoundaryVertices());
                    for (const auto bv : m.boundaryVertices())
                        result(bv.index()) = bv.volumeVertex().index();
                    return result;
               })
          .def("boundaryNodes", [](const Mesh &m) {
                    Eigen::VectorXi result(m.numBoundaryNodes());
                    for (const auto bn : m.boundaryNodes())
                        result(bn.index()) = bn.volumeNode().index();
                    return result;
               })
          .def("elementsAdjacentBoundary", [](const Mesh &m) {
                  Eigen::VectorXi result(m.numBoundaryElements());
                  for (const auto be : m.boundaryElements())
                      result[be.index()] = be.opposite().simplex().index();
                  return result;
              })
          .def_static("integratedShapeFunctions",         []() { return integratedShapeFunctions<Mesh::Deg, Mesh::    K>(); })
          .def_static("integratedBoundaryShapeFunctions", []() { return integratedShapeFunctions<Mesh::Deg, Mesh::K - 1>(); })

          .def("visualizationTriangles", &getVisualizationTriangles<Mesh>)
          .def("visualizationVertices",  &getVisualizationVertices <Mesh>)
          .def("visualizationGeometry",  [](const Mesh &m, double normalCreaseAngle) { return getVisualizationGeometry(m, normalCreaseAngle); }, py::arg("normalCreaseAngle") = M_PI)
          .def("visualizationField", [](const Mesh &m, const Eigen::VectorXd &f) { return getVisualizationField(m, f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("visualizationField", [](const Mesh &m, const MXNd            &f) { return getVisualizationField(m, f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("vertexNormals", &getAreaWeightedNormals<Mesh>, (K == 2) ? "Vertex normals (triangle area weighted)"
                                                                        : "Boundary vertex normals (triangle area weighted)")
          .def("normals", &getNormals<Mesh>, (K == 2) ? "Triangle normals"
                                                      : "Boundary triangle normals")
          .def("elementVolumes", [](const Mesh &m) {
                      Eigen::VectorXd result(m.numElements());
                      for (auto e : m.elements()) result[e.index()] = e->volume();
                      return result;
                  })
          .def("boundaryElementVolumes", [](const Mesh &m) {
                      Eigen::VectorXd result(m.numBoundaryElements());
                      for (auto be : m.boundaryElements()) result[be.index()] = be->volume();
                      return result;
                  })
          .def("edgeLengths", [](const Mesh &m) {
                std::map<UnorderedPair, Real> edgeLengths;
                for (const auto he : m.halfEdges()) {
                    edgeLengths.emplace(UnorderedPair(he.tail().index(), he.tip().index()),
                                        (he.tip().node()->p -he.tail().node()->p).norm());
                }
                Eigen::VectorXd result(edgeLengths.size());
                int i = 0;
                for (const auto &e : edgeLengths)
                    result[i++] = e.second;
                return result;
          }, "Get the length of each mesh edge in arbitrary order")
          .def("barycenters", [](const Mesh &m) {
                Eigen::Matrix<typename Mesh::Real, Eigen::Dynamic, EmbeddingDimension> result(m.numElements(), size_t(EmbeddingDimension));
                for (const auto e : m.elements()) {
                    auto b = result.row(e.index());
                    b.setZero();
                    for (const auto v : e.vertices()) {
                        b += v.node()->p;
                    }
                    b /= e.numVertices();
                }
                return result;
              }, "Get the barycenters of each element")

          .def("numVertices", &Mesh::numVertices)
          .def("numElements", &Mesh::numElements)
          .def("numNodes",    &Mesh::numNodes)
          .def("save", [&](const Mesh &m, const std::string& path) { return MeshIO::save(path, m); })
          .def("field_writer", [](const Mesh &m, const std::string &path) { return Future::make_unique<MSHFieldWriter>(path, m); }, py::arg("path"))
          .def("is_tet_mesh",  [](const Mesh &) { return K == 3; })
          .def_property_readonly(       "bbox", [](const Mesh& m) { const auto bb = m.boundingBox(); return std::make_pair(bb.minCorner, bb.maxCorner); })
          .def_property_readonly("bbox_volume", [](const Mesh& m) { return m.boundingBox().volume(); }, "bounding box volume")
          .def_property_readonly(     "volume", [](const Mesh& m) { return m.volume(); }, "mesh volume")
          .def_property_readonly_static("degree",             [](py::object) { return Mesh::Deg; })
          .def_property_readonly_static("simplexDimension",   [](py::object) { return Mesh::K; })
          .def_property_readonly_static("embeddingDimension", [](py::object) { return EmbeddingDimension; })

          .def("copy", [](const Mesh &m) { return std::make_shared<Mesh>(m); })

          .def(py::pickle([](const Mesh &m) { return py::make_tuple(getV(m), getF(m)); },
                          [](const py::tuple &t) {
                              if (t.size() != 2)  throw std::runtime_error("Invalid pickled state!");
                              auto V = t[0].cast<VType<Mesh>>();
                              auto F = t[1].cast<FType<Mesh>>();
                              return std::make_shared<Mesh>(F, V);
                          }));
          ;
      return mb;
    }
};

template<class Mesh>
struct MeshBindings;

// Triangle-mesh-specific bindings.
template<size_t _Deg, class _EmbeddingSpace>
struct MeshBindings<FEMMesh<2, _Deg, _EmbeddingSpace>> : public MeshBindingsBase<FEMMesh<2, _Deg, _EmbeddingSpace>> {
    using Mesh = FEMMesh<2, _Deg, _EmbeddingSpace>;
    using TMesh = typename Mesh::BaseMesh; // TriMesh data structure underlying FEMMesh
    using CHEHandle = typename TMesh::template HEHandle<const TMesh>;
    using Base = MeshBindingsBase<Mesh>;
    using Real = typename Mesh::Real;
    using V3d  = Eigen::Matrix<Real, 3, 1>;
    static MeshBindingsType<Mesh> bind(py::module &module, py::module &detail_module) {
        auto mesh_bindings = Base::bind(module, detail_module);
        mesh_bindings
            .def("numTris",     &Mesh::numTris)
            .def("triangles",  [](const Mesh &m) { return getElementCorners(m.elements()); })
            .def("trisAdjTri", [](const Mesh &m, size_t ti) {
                    std::vector<int> result;
                    if (ti >= m.numTris()) throw std::runtime_error("Triangle index out of bounds");
                    for (const auto tri_j : m.tri(ti).neighbors()) {
                        if (!tri_j) continue;
                        result.push_back(tri_j.index());
                    }
                    return result;
                })
            .def("vtsAdjVtx", [](const Mesh &m, size_t vi) {
                    std::vector<int> result;
                    if (vi >= m.numVertices()) throw std::runtime_error("Vertex index out of bounds");
                    for (const auto he : m.vertex(vi).incidentHalfEdges())
                        result.push_back(he.tail().index());
                    return result;
                })
            .def("valences", [](const Mesh &m) {
                    std::vector<int> result(m.numVertices());
                    for (const auto tri : m.elements()) {
                        for (const auto v : tri.vertices())
                            ++result[v.index()];
                    }
                    return result;
                })
            .def("angleDeficits", [](const Mesh &m) {
                        Eigen::VectorXd result(m.numVertices());
                        for (const auto v : m.vertices()) {
                            Real deficit = 2 * M_PI;
                            if (v.isBoundary()) { result[v.index()] = 0.0; continue; }
                            for (const auto he : v.incidentHalfEdges()) {
                                if (!he.tri()) continue;
                                V3d  p = padTo3D(he.tip().node()->p);
                                V3d e1 = padTo3D(he.next().tip().node()->p) - p,
                                    e2 = padTo3D(he.tail().node()->p) - p;
                                deficit -= atan2(e1.cross(e2).norm(), e1.dot(e2));
                            }
                            result[v.index()] = deficit;
                        }
                        return result;
                    })
            .def("boundaryLoops", [](const Mesh &m) {
                    std::vector<bool> visited(m.numBoundaryVertices());
                    std::vector<std::vector<size_t>> result;
                    for (const auto be : m.boundaryEdges()) {
                        if (visited[be.tip().index()]) continue;
                        auto curr = be;
                        result.emplace_back();
                        auto &loop = result.back();
                        while (!visited[curr.tip().index()]) {
                            size_t bvi = curr.tip().index();
                            visited[bvi] = true;
                            curr = curr.next();
                            loop.push_back(bvi);
                        }
                    }
                    return result;
                }, "Get the lists of *boundary vertex indices* making up each boundary loop")
            .def("visitEdges", [](const Mesh &m, const std::function<void(std::pair<size_t, size_t>, size_t)> &pyvisitor) {
                    m.visitEdges([&pyvisitor](const CHEHandle &he, size_t edgeIdx) {
                            pyvisitor(std::make_pair(he.tail().index(), he.tip().index()), edgeIdx);
                    });
                }, py::arg("visitor"))
        ;
        return mesh_bindings;
    }
};

// Tetrahedral-mesh-specific bindings.
template<size_t _Deg, class _EmbeddingSpace>
struct MeshBindings<FEMMesh<3, _Deg, _EmbeddingSpace>> : public MeshBindingsBase<FEMMesh<3, _Deg, _EmbeddingSpace>> {
    using Mesh = FEMMesh<3, _Deg, _EmbeddingSpace>;
    using Base = MeshBindingsBase<Mesh>;
    using BoundaryMesh = FEMMesh<2, Mesh::Deg, typename Mesh::EmbeddingSpace>;
    static MeshBindingsType<Mesh> bind(py::module &module, py::module &detail_module) {
        auto mesh_bindings = Base::bind(module, detail_module);
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

// Wrapper to conform to the BindingInstantiations Binder interface.
struct MeshBinder {
    template<class Mesh>
    static void bind(py::module &module, py::module &detail_module) {
        MeshBindings<Mesh>::bind(module, detail_module);
    }
};

PYBIND11_MODULE(mesh, m)
{
    m.doc() = "MeshFEM finite element mesh data structure bindings";
    py::module detail_module = m.def_submodule("detail");

    bindMSHFieldWriter(m);
    bindMSHFieldParser(m);

    generateMeshSpecificBindings<MeshBinder>(m, detail_module, MeshBinder());

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

    using PST = PolygonSetTriangulation<
        double, Eigen::Vector2d, std::pair<size_t, size_t>>;

    py::class_<PST>(m, "PolygonSetTriangulation")
        .def(py::init<const std::vector<Eigen::Vector2d>&,
                      const std::vector<std::vector<std::pair<size_t, size_t>>>&,
                      const std::vector<Eigen::Vector2d>&,
                      double, double>(), py::arg("points"), py::arg("polygons"), py::arg("holes"), py::arg("target_area"), py::arg("min_hinge_radius") = 0.0)
        .def_property_readonly("V", [](const PST &pst) { return getV(pst.getVertices()); })
        .def_property_readonly("F", [](const PST &pst) { return getF(pst.getElements()); })
        .def_readonly("updatedInputPoints",   &PST::updatedInputPoints)
        .def_readonly("updatedInputPolygons", &PST::updatedInputPolygons)
        .def("getMesh", [](const PST &pst, size_t deg) { return MeshFactory<double>(pst.getElements(), pst.getVertices(), /* K = */ 2, deg, /* N = */ 2); }, py::arg("deg"))
        ;

    ////////////////////////////////////////////////////////////////////////////
    // Free-standing utility functions
    ////////////////////////////////////////////////////////////////////////////
    m.def("save", [&](const std::string& path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
            std::vector<MeshIO::IOVertex > vertices;
            std::vector<MeshIO::IOElement> elements;
            std::tie(vertices, elements) = getMeshIO(V, F);

            MeshIO::save(path, vertices, elements); }, py::arg("path"), py::arg("V"), py::arg("F"))
        ;

    m.def("load_raw", [&](const std::string& path) {
        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> elements;
        MeshIO::load(path, vertices, elements, MeshIO::FMT_GUESS, MeshIO::MESH_GUESS);

        std::pair<Eigen::MatrixXd, Eigen::MatrixXi> result;
        result.first = getV(vertices);
        result.second = getF(elements);
        return result;
    });
}
