////////////////////////////////////////////////////////////////////////////////
// FEMMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides a mesh with basic support for linear and quadratic FEM
//      discretizations.
//      For linear FEM, nodes are located only on the vertices, and for
//      quadratic FEM, nodes are located on both vertices and edge midpoints.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/16/2014 16:22:51
////////////////////////////////////////////////////////////////////////////////
#ifndef FEMMESH_HH
#define FEMMESH_HH

#include <map>
#include <cassert>
#include <type_traits>
#include <memory>

#include <MeshFEM/Geometry.hh>
#include <MeshFEM/EmbeddedElement.hh>

#include <MeshFEM/SimplicialMesh.hh>
#include <MeshFEM/BoundaryMesh.hh>
#include <MeshFEM/Handles/FEMMeshHandles.hh>

#include <MeshFEM/Utilities/VertexArrayAdaptor.hh>

////////////////////////////////////////////////////////////////////////////////
// Forward Declarations
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace> struct DefaultFEMData;
template<size_t _K, size_t _Deg, class EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData = DefaultFEMData>
class FEMMesh;

#include <MeshFEM/Utilities/NameMangling.hh>

// The EmbeddedElement interface depends on which simplex type we were
// embedding--we use this class to wrap it.
template<size_t _K>
struct Embedder;
template<> struct Embedder<2> {
    template<size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
    static void embed(FEMMesh<2, _Deg, EmbeddingSpace, _FEMData> &mesh) {
        for (auto e : mesh.elements())
            e->embed(e.node(0)->p, e.node(1)->p, e.node(2)->p);
        for (auto be : mesh.boundaryElements())
            be->embed(be.node(0).volumeNode()->p, be.node(1).volumeNode()->p);
    }
};
template<> struct Embedder<3> {
    template<size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
    static void embed(FEMMesh<3, _Deg, EmbeddingSpace, _FEMData> &mesh) {
        for (auto e : mesh.elements())
            e->embed(e.node(0)->p, e.node(1)->p, e.node(2)->p, e.node(3)->p);
        for (auto be : mesh.boundaryElements())
            be->embed(be.node(0).volumeNode()->p, be.node(1).volumeNode()->p, be.node(2).volumeNode()->p);
    }
};

// Store positions on all nodes (this will allow support for nonlinear
// elasticity in the future). Typically, the edge node positions will be the
// average of the edge endpoint node positions.
template<class EmbeddingSpace>
struct NodeData {
    EmbeddingSpace p;
};

// Wrapper for all the data types to be included in the FEMMesh.
template<size_t _K, size_t _Deg, class EmbeddingSpace>
struct DefaultFEMData {
    typedef TMEmptyData                                           Vertex;
    typedef NodeData<EmbeddingSpace>                              Node;
    typedef LinearlyEmbeddedElement<_K, _Deg, EmbeddingSpace>     Element;
    typedef TMEmptyData                                           BoundaryVertex;
    typedef TMEmptyData                                           BoundaryNode;
    typedef LinearlyEmbeddedElement<_K - 1, _Deg, EmbeddingSpace> BoundaryElement;
};

template<size_t _K, size_t _Deg, class _EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData>
class FEMMesh : public Concepts::ElementMesh,
    public SimplicialMesh<_K,
        // Store mesh-tied entities ({boundary,volume} {vertex,element} data) in the
        // underlying mesh data structure. The node data is managed by this data
        // structure.
        typename _FEMData<_K, _Deg, _EmbeddingSpace>::Vertex,
        typename _FEMData<_K, _Deg, _EmbeddingSpace>::Element,
        typename _FEMData<_K, _Deg, _EmbeddingSpace>::BoundaryVertex,
        typename _FEMData<_K, _Deg, _EmbeddingSpace>::BoundaryElement>
{
public:
    static constexpr size_t K   = _K;
    static constexpr size_t Deg = _Deg;
    static constexpr size_t EmbeddingDimension = _EmbeddingSpace::RowsAtCompileTime;
    using EmbeddingSpace = _EmbeddingSpace;
    using Real = typename EmbeddingSpace::Scalar;

    // Unpack entity data types.
    using FEMData = _FEMData<_K, _Deg, EmbeddingSpace>;
    using VertexData          = typename FEMData::Vertex;
    using NodeData            = typename FEMData::Node;
    using ElementData         = typename FEMData::Element;
    using BoundaryVertexData  = typename FEMData::BoundaryVertex;
    using BoundaryNodeData    = typename FEMData::BoundaryNode;
    using BoundaryElementData = typename FEMData::BoundaryElement;

    // Determine base mesh type
    using BaseMesh = SimplicialMesh<_K, VertexData, ElementData, BoundaryVertexData, BoundaryElementData>;

    size_t numElementNodes() const { return 0; }
    size_t numVertexNodes()  const { return BaseMesh::numVertices(); }
    size_t numEdgeNodes()    const { return m_edgeForEdgeNode.size(); }
    size_t numNodes()        const { return numVertexNodes() +  numEdgeNodes() + numElementNodes(); }
    size_t numElements()     const { return BaseMesh::numSimplices(); }

    template<typename Elements, typename Vertices>
    FEMMesh(const Elements &elems, const Vertices &vertices);
    static std::unique_ptr<FEMMesh> load(const std::string &path);

    // ~FEMMesh() { std::cout << "FEMMesh (" << getMeshName<FEMMesh>() << ") destructor called" << std::endl; }

    // Entity handles (declared in Handles/FEMMeshHandles.hh).
    template<class _Mesh> using  VHandle = typename HandleTraits<FEMMesh>::template  VHandle<_Mesh>; // Vertex
    template<class _Mesh> using  NHandle = typename HandleTraits<FEMMesh>::template  NHandle<_Mesh>; // Node
    template<class _Mesh> using  EHandle = typename HandleTraits<FEMMesh>::template  EHandle<_Mesh>; // Element
    template<class _Mesh> using BVHandle = typename HandleTraits<FEMMesh>::template BVHandle<_Mesh>; // Boundary vertex
    template<class _Mesh> using BNHandle = typename HandleTraits<FEMMesh>::template BNHandle<_Mesh>; // Boundary node
    template<class _Mesh> using BEHandle = typename HandleTraits<FEMMesh>::template BEHandle<_Mesh>; // Boundary element

    // We also want to allow traversal of this derived mesh starting from hafledges,
    // so we need to override the halfEdge(i)/halfEdges() methods.
    template<class _Mesh> using HEHandle = typename BaseMesh::template HEHandle<_Mesh>; // Halfedge (tri or tet)

    // Number of strictly interior nodes (excluding nodes on the boundary).
    size_t numInternalNodes() const { return numNodes() - numBoundaryNodes(); }

    size_t numBoundaryElementNodes() const { return 0; }
    size_t numBoundaryVertexNodes()  const { return BaseMesh::numBoundaryVertices(); }
    size_t numBoundaryEdgeNodes()    const { return m_volEdgeForBdryEdge.size();  }
    size_t numBoundaryNodes()        const { return numBoundaryVertexNodes() + numBoundaryEdgeNodes() + numBoundaryElementNodes(); }
    size_t numBoundaryElements()     const { return BaseMesh::numBoundarySimplices(); }

    ////////////////////////////////////////////////////////////////////////////
    // Entity access
    ////////////////////////////////////////////////////////////////////////////
     VHandle<FEMMesh>          vertex(size_t i) { return  VHandle<FEMMesh>(i, *this); }
     NHandle<FEMMesh>            node(size_t i) { return  NHandle<FEMMesh>(i, *this); }
     EHandle<FEMMesh>         element(size_t i) { return  EHandle<FEMMesh>(i, *this); }
    HEHandle<FEMMesh>        halfEdge(size_t i) { return HEHandle<FEMMesh>(i, *this); }
    BVHandle<FEMMesh>  boundaryVertex(size_t i) { return BVHandle<FEMMesh>(i, *this); }
    BNHandle<FEMMesh>    boundaryNode(size_t i) { return BNHandle<FEMMesh>(i, *this); }
    BEHandle<FEMMesh> boundaryElement(size_t i) { return BEHandle<FEMMesh>(i, *this); }

     VHandle<const FEMMesh>          vertex(size_t i) const { return  VHandle<const FEMMesh>(i, *this); }
     NHandle<const FEMMesh>            node(size_t i) const { return  NHandle<const FEMMesh>(i, *this); }
     EHandle<const FEMMesh>         element(size_t i) const { return  EHandle<const FEMMesh>(i, *this); }
    HEHandle<const FEMMesh>        halfEdge(size_t i) const { return HEHandle<const FEMMesh>(i, *this); }
    BVHandle<const FEMMesh>  boundaryVertex(size_t i) const { return BVHandle<const FEMMesh>(i, *this); }
    BNHandle<const FEMMesh>    boundaryNode(size_t i) const { return BNHandle<const FEMMesh>(i, *this); }
    BEHandle<const FEMMesh> boundaryElement(size_t i) const { return BEHandle<const FEMMesh>(i, *this); }

    ////////////////////////////////////////////////////////////////////////////
    // Entity ranges (for range-based for).
    // (We must overload the ones provided by the base mesh or else we'll get
    //  the base mesh's handles instead of our derived ones.)
    ////////////////////////////////////////////////////////////////////////////
    // Specialization for nested class templates isn't allowed, so we can't
    // implement a true traits design pattern...
    template<template<class> class _Handle> using  HR = HandleRange<      FEMMesh, _Handle>;
    template<template<class> class _Handle> using CHR = HandleRange<const FEMMesh, _Handle>;

public:
    HR< VHandle>         vertices() { return HR< VHandle>(*this); }
    HR< NHandle>            nodes() { return HR< NHandle>(*this); }
    HR< EHandle>         elements() { return HR< EHandle>(*this); }
    HR<HEHandle>        halfEdges() { return HR<HEHandle>(*this); }
    HR<BVHandle> boundaryVertices() { return HR<BVHandle>(*this); }
    HR<BNHandle>    boundaryNodes() { return HR<BNHandle>(*this); }
    HR<BEHandle> boundaryElements() { return HR<BEHandle>(*this); }

    CHR< VHandle>         vertices() const { return CHR< VHandle>(*this); }
    CHR< NHandle>            nodes() const { return CHR< NHandle>(*this); }
    CHR< EHandle>         elements() const { return CHR< EHandle>(*this); }
    CHR<HEHandle>        halfEdges() const { return CHR<HEHandle>(*this); }
    CHR<BVHandle> boundaryVertices() const { return CHR<BVHandle>(*this); }
    CHR<BNHandle>    boundaryNodes() const { return CHR<BNHandle>(*this); }
    CHR<BEHandle> boundaryElements() const { return CHR<BEHandle>(*this); }

    // Explicit const handle ranges (for const iteration over nonconst mesh)
    CHR< VHandle>         constVertices() const { return CHR< VHandle>(*this); }
    CHR< NHandle>            constNodes() const { return CHR< NHandle>(*this); }
    CHR< EHandle>         constElements() const { return CHR< EHandle>(*this); }
    CHR<HEHandle>        constHalfEdges() const { return CHR<HEHandle>(*this); }
    CHR<BVHandle> constBoundaryVertices() const { return CHR<BVHandle>(*this); }
    CHR<BNHandle>    constBoundaryNodes() const { return CHR<BNHandle>(*this); }
    CHR<BEHandle> constBoundaryElements() const { return CHR<BEHandle>(*this); }

    // (re-)embed the mesh elements.
    // Mesh vertex nodes are read from the passed vertex position array and edge
    // nodes are positioned at the edge midpoint.
    template<typename Vertices>
    void setNodePositions(const Vertices &vertices) {
        for (auto n : nodes()) {
            assert(n.isVertexNode() || n.isEdgeNode());
            if (n.isVertexNode())
                n->p = truncateFrom3D<EmbeddingSpace>(VertexArrayAdaptor<Vertices>::get(vertices, n.vertex().index()));
        }
        for (auto n : nodes()) {
            if (n.isEdgeNode()) {
                const UnorderedPair &edge = m_edgeForEdgeNode.at(n.edgeNodeIndex());
                n->p = 0.5 * (vertex(edge[0]).node()->p + vertex(edge[1]).node()->p);
            }
        }

        m_embedElements();
        m_computeBBox();
    }

    const UnorderedPair& edgeForEdgeNode(size_t edgeNodeIndex) const {
        assert(edgeNodeIndex >= 0 && edgeNodeIndex < m_edgeForEdgeNode.size());
        return m_edgeForEdgeNode.at(edgeNodeIndex);
    }

    // Also support reading from Luigi/Nico's vertex format
    void setNodePositions(const std::vector<std::array<double,
            EmbeddingSpace::RowsAtCompileTime>> &vertices) {
        std::vector<V3MatchingScalarType<EmbeddingSpace>> convertedVertices(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            convertedVertices[i][0] = vertices[i][0];
            convertedVertices[i][1] = vertices[i][1];
            convertedVertices[i][2] = (EmbeddingSpace::RowsAtCompileTime == 3)
                                        ? vertices[i][2] : 0.0;
        }
        setNodePositions(convertedVertices);
    }

    const BBox<EmbeddingSpace> &boundingBox() const {
        return m_bbox;
    }

    Real volume() const {
        Real vol = 0.0;
        for (size_t i = 0; i < numElements(); ++i)
            vol += element(i)->volume();
        return vol;
    }

    EmbeddingSpace centerOfMass() const {
        EmbeddingSpace result(EmbeddingSpace::Zero());
        for (const auto &e : elements()) {
            // Center of mass of each element is just its barycenter...
            EmbeddingSpace contrib(EmbeddingSpace::Zero());
            for (const auto &v : e.vertices())
                contrib += v.node()->p;
            result += contrib * (e->volume() / e.numVertices());
        }
        return result / volume();
    }

    EmbeddingSpace elementBarycenter(size_t ei) const {
        EmbeddingSpace b(EmbeddingSpace::Zero());
        auto e = element(ei);
        assert(e);
        for (size_t i = 0; i < e.numVertices(); ++i) {
            // Nodes 0...numVertices - 1 are located on the vertices
            b += e.vertex(i).node()->p;
        }
        b /= e.numVertices();
        return b;
    }

    EmbeddingSpace boundaryElementBarycenter(size_t ei) const {
        EmbeddingSpace b(EmbeddingSpace::Zero());
        auto e = boundaryElement(ei);
        assert(e);
        for (size_t i = 0; i < e.numVertices(); ++i) {
            // Nodes 0...numVertices - 1 are located on the vertices
            b += e.vertex(i).node()->p;
        }
        b /= e.numVertices();
        return b;
    }


    BoundaryMesh<      FEMMesh> boundary()       { return BoundaryMesh<      FEMMesh>(*this); }
    BoundaryMesh<const FEMMesh> boundary() const { return BoundaryMesh<const FEMMesh>(*this); }

private:
    // Table of **non-vertex** node indices for each element. We needn't store
    // vertex node indices because our mesh data structure knows them.
    // The true node index is numVertexNodes() + m_N[i]
    std::vector<int>  m_N;
    // Table of **non-vertex** boundary node indices for each boundary element.
    // We needn't store boundary vertex node indices because our mesh data
    // structure knows them.
    // The true node index is numBoundaryVertexNodes() + m_BN[i]
    std::vector<int> m_BN;

    std::vector<UnorderedPair> m_edgeForEdgeNode;

    // Look up the boundary/volume edge coinciding with a volume/boundary edge
    // Every boundary edge has a corresponding volume edge but not the other way
    // around--m_bdryEdgeForVolEdge is -1 for edges without a boundary edge
    std::vector<int> m_bdryEdgeForVolEdge;
    std::vector<int> m_volEdgeForBdryEdge;

    // Node data storage
    DataStorage<typename FEMData::Node>         m_nodeData;
    DataStorage<typename FEMData::BoundaryNode> m_boundaryNodeData;

    // Mesh bounding box, updated every time the node positions change with
    // setNodePositions()
    BBox<EmbeddingSpace> m_bbox;

    // Handles need access to private traversal operations below
    template<class Mesh> friend class _FEMMeshHandles::VHandle;
    template<class Mesh> friend class _FEMMeshHandles::NHandle;
    template<class Mesh> friend class _FEMMeshHandles::EHandle;
    template<class Mesh> friend class _FEMMeshHandles::BVHandle;
    template<class Mesh> friend class _FEMMeshHandles::BNHandle;
    template<class Mesh> friend class _FEMMeshHandles::BEHandle;

    // Nodes 0..#Vertices-1 are located on the corresponding vertex.
    // The remaining nodes do not have vertices.
    int m_vertexForNode(int n) const {
        if (size_t(n) < BaseMesh::numVertices()) return n;
        else return -1;
    }
    int m_nodeForVertex(int v) const {
        if (size_t(v) < BaseMesh::numVertices()) return v;
        else return -1;
    }

    // Boundary Nodes 0..#BdryVertices-1 are located on the corresponding
    // boundary vertex. The remaining nodes do not have vertices.
    int m_bdryVtxForBdryNode(int bn) const {
        if (size_t(bn) < BaseMesh::numBoundaryVertices()) return bn;
        else return -1;
    }
    int m_nodeForBoundaryVertex(int bv) const {
        if (size_t(bv) < BaseMesh::numBoundaryVertices()) return bv;
        else return -1;
    }

    // Node index of each volume elements' nodes
    // Nodes 0..Simplex::numVertices(_K)-1 indices coincide with vertex index
    // Nodes Simplex::numVertices(_K)..Simplex::numNodes()-1 indices are in m_N
    int m_nodeOfElement(size_t n, size_t e) const {
        assert((e < numElements()) && (n < Simplex::numNodes(_K, _Deg)));
        int nidx;
        if (n < Simplex::numVertices(_K))
            nidx = BaseMesh::simplex(e).vertex(n).index();
        else {
            n -= Simplex::numVertices(_K);
            assert(n < Simplex::numEdges(_K));
            nidx = numVertexNodes() + m_N[Simplex::numEdges(_K) * e + n];
        }
        assert(size_t(nidx) < numNodes());
        return nidx;
    }
    int m_nodeOfBdryElement(size_t bn, size_t be) const {
        assert((be < numBoundaryElements()) && bn < Simplex::numNodes(_K - 1, _Deg));
        int bnidx;
        if (bn < Simplex::numVertices(_K - 1))
            bnidx = BaseMesh::boundarySimplex(be).vertex(bn).index();
        else {
            bn -= Simplex::numVertices(_K - 1);
            assert(bn < Simplex::numEdges(_K - 1));
            bnidx = numBoundaryVertexNodes() + m_BN[Simplex::numEdges(_K - 1) * be + bn];
        }
        assert(size_t(bnidx) < numBoundaryNodes());
        return bnidx;
    }

    // The edge node index associated with a node
    int m_edgeNodeIndex(size_t n) const {
        if (n < BaseMesh::numVertices())
            return -1;
        n -= BaseMesh::numVertices();
        assert(n < numEdgeNodes());
        return n;
    }
    int m_bdryEdgeNodeIndex(size_t bn) const {
        if (bn < BaseMesh::numBoundaryVertices())
            return -1;
        bn -= BaseMesh::numBoundaryVertices();
        assert(bn < numBoundaryEdgeNodes());
        return bn;
    }

    // returns -1 for interior nodes
    int m_bdryNodeForVolNode(int n) const {
        if (n == -1) return -1;
        int v = m_vertexForNode(n);
        // Vertex node indices coincide with vertex indices
        if (v >= 0) return BaseMesh::m_bdryVertexIdx(v); // -1 if internal
        // Edge nodes indices correspond to numVertices + edge index
        size_t volEn = m_edgeNodeIndex(n);
        assert(volEn < m_bdryEdgeForVolEdge.size());
        int beidx = m_bdryEdgeForVolEdge[volEn];
        if (beidx < 0) return beidx; // interior edge node
        return beidx + numBoundaryVertexNodes();
    }

    int m_volNodeForBdryNode(int n) const {
        if (n < 0) return -1;
        // Vertex node indices coincide with vertex indices
        if (size_t(n) < BaseMesh::numBoundaryVertices()) return BaseMesh::m_vertexForBdryVertex(n);
        size_t beidx = m_bdryEdgeNodeIndex(n);
        assert(beidx < m_volEdgeForBdryEdge.size());
        return m_volEdgeForBdryEdge[beidx] + numVertexNodes();
    }

    // (re-)embed the elements in EmbeddingSpace (when vertex positions change)
    void m_embedElements() {
        Embedder<_K>::embed(*this);
    }

    // (re-)compute the bounding box (when vertex positions change)
    void  m_computeBBox() {
        if (BaseMesh::numVertices() == 0) {
            m_bbox = BBox<EmbeddingSpace>();
            return;
        }
        m_bbox = BBox<EmbeddingSpace>(node(0)->p, node(0)->p);
        for (size_t i = 1; i < numNodes(); ++i)
            m_bbox.unionPoint(node(i)->p);
    }
};

#include <MeshFEM/FEMMesh.inl>

#endif /* end of include guard: FEMMESH_HH */
