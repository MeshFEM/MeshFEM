#ifndef COLLISIONMESH_HH
#define COLLISIONMESH_HH
#include <MeshFEM/Types.hh>
#include <MeshFEM/Geometry.hh>
#include <map>

struct CollisionMesh {
    template<class Mesh_>
    static CollisionMesh constructForMesh(const Mesh_ &m, size_t N = Mesh_::EmbeddingSpace::RowsAtCompileTime, bool forceLinear = false) {
        static constexpr size_t Deg = Mesh_::Deg;
        static constexpr size_t K   = Mesh_::K;
        if (N < K) throw std::runtime_error("Embedding dimension (" + std::to_string(N) + ") must not be less than the simplex dimension (" + std::to_string(K) + ")");

        CollisionMesh result;
        auto &faces = result.faces;
        auto &edges = result.edges;
        const bool use_midedge_nodes = (Deg == 2) && !forceLinear;

        if constexpr (K == 3) {
            // For tet meshes, first extract the boundary triangles
            // and then use those to determine edges.
            if (!use_midedge_nodes) {
                faces.resize(m.numBoundaryElements(), 3);
                for (auto be : m.boundaryElements())
                    for (auto bv : be.vertices())
                        faces(be.index(), bv.localIndex()) = bv.index();
            }
            else {
                faces.resize(4 * m.numBoundaryElements(), 3);
                for (auto be : m.boundaryElements()) {
                    //     0
                    //    / \ 
                    //   3---5
                    //  / \ / \ 
                    // 1---4---2
                    faces.row(4 * be.index() + 0) << be.node(0).index(), be.node(3).index(), be.node(5).index();
                    faces.row(4 * be.index() + 1) << be.node(3).index(), be.node(1).index(), be.node(4).index();
                    faces.row(4 * be.index() + 2) << be.node(3).index(), be.node(4).index(), be.node(5).index();
                    faces.row(4 * be.index() + 3) << be.node(4).index(), be.node(2).index(), be.node(5).index();
                }
            }

            std::map<UnorderedPair, size_t> emap;
            for (int fi = 0; fi < faces.rows(); ++fi) {
                for (size_t c = 0; c < 3; ++c) {
                    UnorderedPair key(faces(fi, c), faces(fi, (c + 1) % 3));
                    if (emap.count(key) == 0) emap.emplace(key, emap.size());
                }
            }
            edges.resize(emap.size(), 2);
            for (auto &k : emap)
                edges.row(k.second) << k.first[0], k.first[1];
        }
        else {
            // For triangle meshes, we have only edges to extract.
            static_assert(K == 2, "Only 2D and 3D simplicial meshes are supported");
            if (!use_midedge_nodes) {
                edges.resize(m.numBoundaryElements(), 2);
                for (auto be : m.boundaryElements())
                    edges.row(be.index()) << be.node(0).index(), be.node(1).index();
            }
            else {
                edges.resize(2 * m.numBoundaryElements(), 2);
                for (auto be : m.boundaryElements()) {
                    edges.row(2 * be.index() + 0) << be.node(0).index(), be.node(2).index();
                    edges.row(2 * be.index() + 1) << be.node(2).index(), be.node(1).index();
                }
            }
        }

        auto &nfcv = result.nodeForCollisionMeshVertex;
        nfcv.resize(use_midedge_nodes ? m.numBoundaryNodes() : m.numBoundaryVertexNodes());
        for (int i = 0; i < nfcv.size(); ++i)
            nfcv[i] = m.boundaryNode(i).volumeNode().index();
        result.N = N;
        result.bbox = m.boundingBox();
        result.fullModelBlockVars = m.numNodes();
        return result;
    }

    // Index tables representing the boundary mesh used for collision (e.g.,
    // for IPC). Note that these tables hold indices of *collision mesh
    // vertices*, which are different from nodes/vertices of the underlying
    // elastic object. Currently we only implement support for elastic
    // objects whose collision mesh vertices coincide with nodes of a FEM
    // mesh (which can be determined by `nodeForCollisionMeshVertex`). This
    // notably excludes elastic rods.
    Eigen::MatrixXi edges, faces;
    Eigen::VectorXi nodeForCollisionMeshVertex;

    using VMaxd = VecMaxN_T<Real, 3>;
    BBox<VMaxd> bbox;

    size_t fullModelBlockVars = 0; // number of block variables (nodes) in the simulation mesh
    size_t N = 0;
    size_t numCollisionVertices() const { return nodeForCollisionMeshVertex.size(); }

    // Extract a per-vertex vector field over this collision mesh from the
    // full simulation DoF vector `vars.`
    Eigen::MatrixXd extractVectorField(const Eigen::VectorXd &vars) const {
        const size_t ncv = numCollisionVertices();
        Eigen::MatrixXd result(ncv, N);
        extractVectorFieldToDst(vars, result);
        return result;
    }

    // Extract a per-vertex vector field over this collision mesh from the
    // full simulation DoF vector `vars.`
    template<class Derived>
    void extractVectorFieldToDst(const Eigen::VectorXd &vars, Eigen::MatrixBase<Derived> &dst) const {
        const size_t ncv = numCollisionVertices();
        if ((size_t(dst.rows()) != ncv) || (size_t(dst.cols()) != N)) throw std::runtime_error("CollisionMesh.extractVectorField: dst must already be the correct size");
        parallel_for_range(ncv, [&](size_t i) {
            dst.row(i) = vars.segment(N * nodeForCollisionMeshVertex[i], N);
        }, /* grain_size */ 0, /* parallelism_threshold */ 1000);
    }
};

#endif /* end of include guard: COLLISIONMESH_HH */
