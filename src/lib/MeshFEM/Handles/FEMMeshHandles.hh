#include <type_traits>
#include <MeshFEM/Handles/Handle.hh>
// #include <MeshFEM/BoundaryMesh.hh>
#include <MeshFEM/MeshDataTraits.hh>

namespace _FEMMeshHandles {

// We need to expliclty reference this enclosing scope to hack around an old
// clang bug involving injected class names; make it less verbose
namespace _hndl = ::_FEMMeshHandles;

template<class _Mesh> using  _VData = typename MeshDataTraits<_Mesh>::VertexData;
template<class _Mesh> using  _NData = typename MeshDataTraits<_Mesh>::NodeData;
template<class _Mesh> using  _EData = typename MeshDataTraits<_Mesh>::ElementData;
template<class _Mesh> using _BVData = typename MeshDataTraits<_Mesh>::BoundaryVertexData;
template<class _Mesh> using _BNData = typename MeshDataTraits<_Mesh>::BoundaryNodeData;
template<class _Mesh> using _BEData = typename MeshDataTraits<_Mesh>::BoundaryElementData;

template<class _Mesh>
struct BaseMesh;

template<size_t _K, size_t _Deg, class _EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData>
struct BaseMesh<FEMMesh<_K, _Deg, _EmbeddingSpace, _FEMData>> {
    using DMesh = FEMMesh<_K, _Deg, _EmbeddingSpace, _FEMData>;
    using type = SimplicialMesh<_K, _VData<DMesh>, _EData<DMesh>, _BVData<DMesh>, _BEData<DMesh>>;
    static constexpr size_t   K = _K;
    static constexpr size_t Deg = _Deg;
};

template<class _Mesh>
struct BaseMesh : public BaseMesh<typename std::decay<_Mesh>::type> { };

template<class _Mesh> using BaseMesh_t = typename BaseMesh<_Mesh>::type;

////////////////////////////////////////////////////////////////////////////////
// Vertex Handles: just add node() to access the node at this vertex
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class VHandle : public BaseMesh_t<_Mesh>::template VHandle<_Mesh> {
protected:
    using Base = typename BaseMesh_t<_Mesh>::template VHandle<_Mesh>;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    using NH = typename _Mesh::template NHandle<_Mesh>;
public:
    NH node() const { return NH(m_mesh.m_nodeForVertex(m_idx), m_mesh); }
};

////////////////////////////////////////////////////////////////////////////////
// Node Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class NHandle : public Handle<_Mesh, NHandle, _NData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::NHandle, _NData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    using  VH = typename _Mesh::template  VHandle<_Mesh>;
    using BNH = typename _Mesh::template BNHandle<_Mesh>;
    using HEH = typename _Mesh::template HEHandle<_Mesh>;
public:
    bool valid() const { return (m_idx >= 0) && (size_t(m_idx) < m_mesh.numNodes()); }

    int edgeNodeIndex() const { return m_mesh.m_edgeNodeIndex(m_idx); }
    bool isEdgeNode()   const { return edgeNodeIndex() >= 0; }
    bool isVertexNode() const { return m_mesh.m_vertexForNode(m_idx) >= 0; }

    // If this is a vertex node, get an incident halfedge; if it's an edge node,
    // get the containing halfedge.
    HEH halfEdge() const { return !isEdgeNode() ? vertex().halfEdge() : HEH(m_mesh.m_halfEdgeForEdgeNode[edgeNodeIndex()], m_mesh); }

    // Get the vertex this node is sitting on (if any)
    VH vertex()     const { return VH(m_mesh.m_vertexForNode(m_idx), m_mesh); }

    // Get the boundary node collocated with this volume node
    // Returns invalid if internal
    BNH boundaryNode() const { return BNH(m_mesh.m_bdryNodeForVolNode(m_idx), m_mesh); }

    // Call `visitor(ei)` for each incident element `ei`.
    template<class F>
    void visitIncidentElements(F &&visitor) const {
        if (isEdgeNode()) { halfEdge().visitIncidentElements(visitor); }
        else              {   vertex().visitIncidentElements(visitor); }
    }

    // Identity operation--avoids explicitly handling some special use cases.
    const NHandle &volumeNode() const { return *this; }
          NHandle &volumeNode()       { return *this; }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_nodeData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// Element Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class EHandle : public BaseMesh_t<_Mesh>::template SHandle<_Mesh> {
protected:
    using Base = typename BaseMesh_t<_Mesh>::template SHandle<_Mesh>;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    using NH = typename _Mesh::template NHandle<_Mesh>;
    using VH = typename _Mesh::template VHandle<_Mesh>;
public:
    static constexpr size_t numNodes() { return Simplex::numNodes(BaseMesh<_Mesh>::K, BaseMesh<_Mesh>::Deg); }
    NH node(size_t i) const { return NH(m_mesh.m_nodeOfElement(i, m_idx), m_mesh); }

    // Support range-based for over nodes
    struct NRangeTraits { using SEHType = NH; using EHType = EHandle; static constexpr size_t count = numNodes(); static constexpr auto get = &EHType::node; };
    SubEntityHandleRange<NRangeTraits> nodes() const { return SubEntityHandleRange<NRangeTraits>(*this); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Vertex Handles: enhance with node() to access the boundary node at
// this boundary vertex
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class BVHandle : public BaseMesh_t<_Mesh>::template BVHandle<_Mesh> {
protected:
    using Base = typename BaseMesh_t<_Mesh>::template BVHandle<_Mesh>;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    using BNH = typename _Mesh::template BNHandle<_Mesh>;
public:
    BNH node() const { return BNH(m_mesh.m_nodeForBoundaryVertex(m_idx), m_mesh); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Node Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class BNHandle : public Handle<_Mesh, BNHandle, _BNData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::BNHandle, _BNData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    using  NH = typename _Mesh::template  NHandle<_Mesh>;
    using BVH = typename _Mesh::template BVHandle<_Mesh>;
public:
    bool valid() const { return (m_idx >= 0) && (size_t(m_idx) < m_mesh.numNodes()); }

    int edgeNodeIndex() const { return m_mesh.m_bdryEdgeNodeIndex(m_idx); }
    bool   isEdgeNode() const { return edgeNodeIndex() >= 0; }
    bool isVertexNode() const { return m_mesh.m_bdryVtxForBdryNode(m_idx) >= 0; }

    BVH       vertex() const { return BVH(m_mesh.m_bdryVtxForBdryNode(m_idx), m_mesh); }
    // Get the volume node collocated with this boundary node.
    NH    volumeNode() const { return NH(m_mesh.m_volNodeForBdryNode(m_idx), m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_boundaryNodeData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Element Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class BEHandle : public BaseMesh_t<_Mesh>::template BSHandle<_Mesh> {
protected:
    using Base = typename BaseMesh_t<_Mesh>::template BSHandle<_Mesh>;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    using BNH = typename _Mesh::template BNHandle<_Mesh>;
public:
    static constexpr size_t numNodes() { return Simplex::numNodes(BaseMesh<_Mesh>::K - 1, BaseMesh<_Mesh>::Deg); }
    BNH node(size_t i) const { return BNH(m_mesh.m_nodeOfBdryElement(i, m_idx), m_mesh); }

    // Support range-based for over boundary nodes
    struct NRangeTraits { using SEHType = BNH; using EHType = BEHandle; static constexpr size_t count = numNodes(); static constexpr auto get = &EHType::node; };
    SubEntityHandleRange<NRangeTraits> nodes() const { return SubEntityHandleRange<NRangeTraits>(*this); }
};

////////////////////////////////////////////////////////////////////////////////
// Half Edge Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class HEHandle : public BaseMesh_t<_Mesh>::template HEHandle<_Mesh> {
protected:
    using Base = typename BaseMesh_t<_Mesh>::template HEHandle<_Mesh>;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    friend Base;
};

} // _FEMMeshHandles

template<size_t _K, size_t _Deg, class EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData>
struct HandleTraits<FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>> {
    template<class _Mesh> using  VHandle = _FEMMeshHandles:: VHandle<_Mesh>; // Vertex
    template<class _Mesh> using  NHandle = _FEMMeshHandles:: NHandle<_Mesh>; // Node
    template<class _Mesh> using  EHandle = _FEMMeshHandles:: EHandle<_Mesh>; // Element
    template<class _Mesh> using HEHandle = _FEMMeshHandles::HEHandle<_Mesh>; // Half edge
    template<class _Mesh> using BVHandle = _FEMMeshHandles::BVHandle<_Mesh>; // Boundary vertex
    template<class _Mesh> using BNHandle = _FEMMeshHandles::BNHandle<_Mesh>; // Boundary node
    template<class _Mesh> using BEHandle = _FEMMeshHandles::BEHandle<_Mesh>; // Boundary element
};

template<class _Mesh> struct HandleRangeTraits<_FEMMeshHandles:: VHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numVertices()        ; } }; // Vertex
template<class _Mesh> struct HandleRangeTraits<_FEMMeshHandles:: NHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numNodes()           ; } }; // Node
template<class _Mesh> struct HandleRangeTraits<_FEMMeshHandles:: EHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numElements()        ; } }; // Element
template<class _Mesh> struct HandleRangeTraits<_FEMMeshHandles::HEHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numHalfEdges()       ; } }; // Half edges
template<class _Mesh> struct HandleRangeTraits<_FEMMeshHandles::BVHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numBoundaryVertices(); } }; // Boundary vertex
template<class _Mesh> struct HandleRangeTraits<_FEMMeshHandles::BNHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numBoundaryNodes()   ; } }; // Boundary node
template<class _Mesh> struct HandleRangeTraits<_FEMMeshHandles::BEHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numBoundaryElements(); } }; // Boundary element
