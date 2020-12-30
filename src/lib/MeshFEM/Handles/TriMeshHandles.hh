#pragma once

#include <MeshFEM/MeshDataTraits.hh>
#include "Handle.hh"
#include "Circulator.hh"

namespace _TriMeshHandles {

// We need to expliclty reference this enclosing scope to hack around an old
// clang bug involving injected class names; make it less verbose
namespace _hndl = ::_TriMeshHandles;

template<class _Mesh> using  _VData = typename MeshDataTraits<_Mesh>::VertexData;
template<class _Mesh> using _HEData = typename MeshDataTraits<_Mesh>::HalfEdgeData;
template<class _Mesh> using  _TData = typename MeshDataTraits<_Mesh>::TriData;
template<class _Mesh> using _BVData = typename MeshDataTraits<_Mesh>::BoundaryVertexData;
template<class _Mesh> using _BEData = typename MeshDataTraits<_Mesh>::BoundaryEdgeData;

////////////////////////////////////////////////////////////////////////////////
// Vertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class VHandle : public Handle<_Mesh, VHandle, _VData<_Mesh>> {
protected:
    // For older clang, VHandle incorrectly refers to the injected class name (not the
    // VHandle template) even when used in a template template parameter context.
    // To support these compilers, we must explicitly refer to the template
    // using the enclosing namespace.
    // "VHandle::template VHandle" solution from the following stackoverflow
    // link no longer works because it now refers to the VHandle constructor as
    // required by the C++ standard.
    // http://stackoverflow.com/questions/17287778/how-do-i-use-the-class-name-inside-the-class-itself-as-a-template-argument
    using _H = Handle<_Mesh, _hndl::VHandle, _VData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using  VH = typename _Mesh::template  VHandle<_Mesh>;
    using BVH = typename _Mesh::template BVHandle<_Mesh>;
    using HEH = typename _Mesh::template HEHandle<_Mesh>;
public:
    bool valid() const { return size_t(m_idx) < m_mesh.numVertices(); }
    bool isBoundary() const { return m_mesh.m_bdryVertexIdx(m_idx) >= 0; }
    BVH boundaryVertex() const { return BVH(   m_mesh.m_bdryVertexIdx(m_idx), m_mesh); }
    // Half-edge incident on this vertex; guaranteed to be opposite the boundary if v is on the boundary.
    HEH       halfEdge() const { return HEH(m_mesh.m_halfEdgeOfVertex(m_idx), m_mesh); }
    // Range circulating counter-clockwise around this vertex.
    CirculatorRange<HEH> incidentHalfEdges() const { return CirculatorRange<HEH>(halfEdge()); }

    // Call `visitor(ei)` for each incident element `ei`.
    template<class F>
    void visitIncidentElements(F &&visitor) {
        for (const auto &he : incidentHalfEdges()) {
            auto t = he.tri();
            if (t) visitor(t.index());
        }
    }

    // Identity operation for unified writing of surface and volume meshes
    // (since point data is typically stored only on the volume vertex)
    VH    volumeVertex() const { return VH(m_idx, m_mesh); }

    typename _H::value_ptr dataPtr() const { return m_mesh.m_vertexData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// HalfEdge Handles
////////////////////////////////////////////////////////////////////////////////
// Circulating around a boundary vertex leads to a complication: we will hit a
// boundary edge at some point. Unfortunately, because C++ is statically typed,
// there's no way for cw() and ccw() to return a BEHandle when
// this happens.
// To address this situation, we allow HEHandles to contain
// negative indices that encode their corresponding boundary halfedge. In this
// state, valid() is false and halfedge data can't be accessed, but all
// traversal operators can still be called (and act like the corresponding
// boundary edge traversal operators). However, ++ and -- operators cannot be
// used; they are only safe on valid handles.
//
// In usage, this means one can still simply use repeated calls to cw() and
// ccw() to circulate around a vertex, but at a single step of the cirulation
// valid() will be false and halfedge data cannot be accessed.
template<class _Mesh>
class HEHandle : public Handle<_Mesh, HEHandle, _HEData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::HEHandle, _HEData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using HEH = typename _Mesh::template HEHandle<_Mesh>;
    using BEH = typename _Mesh::template BEHandle<_Mesh>;
    using  TH = typename _Mesh::template  THandle<_Mesh>;
    using  VH = typename _Mesh::template  VHandle<_Mesh>;
public:
    bool valid()      const { return size_t(m_idx) < m_mesh.numHalfEdges(); }
    bool isBoundary() const { return m_mesh.m_bdryEdgeIdx(m_idx) >= 0; }

    // 1) For half-edges on the boundary, get the "opposite" boundary edge.
    // 2) For half-edges actually encoding a boundary edge (negative
    //    m_idx--should only happen during circulation around boundary vertices)
    //    get a handle on that boundary edge.
    BEH boundaryEdge() const { return BEH(m_mesh.m_bdryEdgeIdx(m_idx), m_mesh); }
    // Dimension-independent terminology:
    BEH boundaryEntity() const { return boundaryEdge(); }

     TH     tri() const { return TH(m_mesh.m_triOfHE(m_idx), m_mesh); }
     TH simplex() const { return tri(); }
     TH element() const { return tri(); }
    HEH next() const {
        if (m_idx < 0) return boundaryEdge().next().m_volumeCast();
        return HEH(m_mesh.template m_HE<_Mesh::Direction::NEXT>(m_idx), m_mesh);
    }

    HEH     prev() const { if (m_idx < 0) return boundaryEdge().prev().m_volumeCast(); return HEH(m_mesh.template m_HE<_Mesh::Direction::PREV>(m_idx), m_mesh); }
    HEH opposite() const { if (m_idx < 0) return boundaryEdge().opposite();            return HEH(m_mesh.template m_HE<_Mesh::Direction::OPP >(m_idx), m_mesh); }

    VH tip()  const { if (m_idx < 0) return boundaryEdge().tip().volumeVertex();  return VH(m_mesh.template m_vertexOfHE<_Mesh::HEVertex::TIP >(m_idx), m_mesh); }
    VH tail() const { if (m_idx < 0) return boundaryEdge().tail().volumeVertex(); return VH(m_mesh.template m_vertexOfHE<_Mesh::HEVertex::TAIL>(m_idx), m_mesh); }

    HEH primary() const {
        if (m_idx < 0) return opposite(); // encoded boundary edge: single volume halfedge is primary, invalid: -1
        if (!isBoundary()) return HEH(std::min(m_idx, opposite().index()), m_mesh); // internally, smaller index is primary
        return HEH(m_idx, m_mesh); // we're the single volume halfedge, so we're primary!
    }

    bool isPrimary() const {
        if (m_idx < 0) return false;           // boundary halfedges aren't primary
        if (opposite().m_idx < 0) return true; // the single interior halfedge on the boundary is primary
        return m_idx < opposite().m_idx;       // in the interior, the smaller indexed halfedge is primary
    }

    // Call `visitor(ei)` for each incident tri `ei`.
    template<class F>
    void visitIncidentElements(F &&visitor) const {
        if (tri())            visitor(tri().index());
        if (opposite().tri()) visitor(opposite().tri().index());
    }

    // Note: these are only correct because of the careful boundary-case
    // handling above.
    HEH ccw() const { return opposite().prev(); }
    HEH  cw() const { return next().opposite(); }

    typename _H::value_ptr dataPtr() const { return m_mesh.m_halfEdgeData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// Triangle Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class THandle : public Handle<_Mesh, THandle, _TData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::THandle, _TData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using  TH = typename _Mesh::template  THandle<_Mesh>;
    using  VH = typename _Mesh::template  VHandle<_Mesh>;
    using HEH = typename _Mesh::template HEHandle<_Mesh>;
public:
    bool      valid() const { return size_t(m_idx) < m_mesh.numTris(); }
    bool isBoundary() const { return halfEdge(0).isBoundary()
                                  || halfEdge(1).isBoundary()
                                  || halfEdge(2).isBoundary(); }

    // Note: true neighbor count can be less than 3; must check if neighbor(i)
    // is valid.
    static constexpr size_t numNeighbors() { return 3; }
    static constexpr size_t numVertices()  { return 3; }

     VH   vertex(size_t i) const { return  VH(m_mesh.m_vertexOfTri(i, m_idx), m_mesh); }
     TH neighbor(size_t i) const { return  TH(m_mesh.m_triAdjTri(i, m_idx), m_mesh); }
    HEH halfEdge(size_t i) const { return HEH(m_mesh.m_halfEdgeOfTri(i, m_idx), m_mesh); }

    // Dimension-independent terminology:
    //  interface of a tet is a half-face
    //  interface of a tri is a half-edge
    HEH interface(size_t i) const { return halfEdge(i); }

    // Support range-based for over vertices, neighboring triangles, and halfedges (interfaces)
    struct  VRangeTraits { using SEHType =  VH; using EHType = THandle; static constexpr size_t count = numVertices (); static constexpr auto get = &EHType::vertex  ; };
    struct NTRangeTraits { using SEHType =  TH; using EHType = THandle; static constexpr size_t count = numNeighbors(); static constexpr auto get = &EHType::neighbor; };
    struct HERangeTraits { using SEHType = HEH; using EHType = THandle; static constexpr size_t count = numNeighbors(); static constexpr auto get = &EHType::halfEdge; };
    SubEntityHandleRange< VRangeTraits>   vertices() const { return SubEntityHandleRange< VRangeTraits>(*this); }
    SubEntityHandleRange<NTRangeTraits>  neighbors() const { return SubEntityHandleRange<NTRangeTraits>(*this); }
    SubEntityHandleRange<HERangeTraits>  halfEdges() const { return SubEntityHandleRange<HERangeTraits>(*this); }
    SubEntityHandleRange<HERangeTraits> interfaces() const { return SubEntityHandleRange<HERangeTraits>(*this); }

    typename _H::value_ptr dataPtr() const { return m_mesh.m_triData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Vertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class BVHandle : public Handle<_Mesh, BVHandle, _BVData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::BVHandle, _BVData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using BVH = typename _Mesh::template BVHandle<_Mesh>;
    using  VH = typename _Mesh::template  VHandle<_Mesh>;
    using BEH = typename _Mesh::template BEHandle<_Mesh>;
    using  TH = typename _Mesh::template  THandle<_Mesh>;
public:
    bool valid() const { return size_t(m_idx) < m_mesh.numBoundaryVertices(); }

     VH volumeVertex() const { return  VH(m_mesh.m_vertexForBdryVertex(m_idx), m_mesh); }
    // Outgoing and incoming boundary edges. Note: getting the incoming edge is
    // slightly more expensive since it can't be retrieved directly from the
    // lookup tables--circulation is required.
    BEH outEdge() const { return BEH(m_mesh.m_bdryELeavingBdryVertex(m_idx), m_mesh); }
    BEH    edge() const { return outEdge().prev(); }

    typename _H::value_ptr dataPtr() const { return m_mesh.m_boundaryVertexData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Edge Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class BEHandle : public Handle<_Mesh, BEHandle, _BEData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::BEHandle, _BEData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using BEH = typename _Mesh::template BEHandle<_Mesh>;
    using HEH = typename _Mesh::template HEHandle<_Mesh>;
    using BVH = typename _Mesh::template BVHandle<_Mesh>;
public:
    bool valid() const { return size_t(m_idx) < m_mesh.numBoundaryEdges(); }

    HEH volumeHalfEdge() const { return HEH(m_mesh.m_HEForBdryEdge(m_idx), m_mesh); }
    HEH       opposite() const { return volumeHalfEdge(); }
    BVH            tip() const { return BVH( m_mesh.m_bdryEdgeTip(m_idx), m_mesh); }
    BVH           tail() const { return BVH(m_mesh.m_bdryEdgeTail(m_idx), m_mesh); }
    BEH           next() const { return BEH(m_mesh.m_nextBdryEdge(m_idx), m_mesh); }
    // Get the previous boundary edge in the clockwise boundary traversal
    // Unfortunately, this data isn't directly accessible from our index tables.
    // Instead, we must circulate clockwise around the tail vertex until we hit
    // the boundary again. For example, to get from current boundary edge, c, to
    // previous boundary edge, p:
    //        ---c--->
    //      T---------+
    //    ^/^\<---1--/
    //   //  \\     /
    //  p/    2\   /
    // //      \\ /
    // +---------+
    // we circulate clockwise around T starting with opposite volume halfedge 1,
    // visiting volume halfedge 2 before finally reaching boundary edge p.
    // Note: HEHandle::cw doesn't call prev, so this isn't an infinite
    // recusion. (Moreover, cw/ccw never call BEHandle's methods when
    // invoked on true volume half-edges).
    BEH prev() const {
        HEH h_it = opposite();
        do { h_it = h_it.cw(); } while (!h_it.isBoundary());
        return h_it.boundaryEdge();
    }

    // Dimension-independent access
    static constexpr size_t  numVertices() { return 2; }
    static constexpr size_t numNeighbors() { return 2; }
    BVH   vertex(size_t i) const { assert(i < 2); return (i == 0) ? tail() : tip(); }
    BEH neighbor(size_t i) const { assert(i < 2); return (i == 0) ? prev() : next(); }

    // Support range-based for over vertices, neighboring edges
    struct  VRangeTraits { using SEHType = BVH; using EHType = BEHandle; static constexpr size_t count = numVertices (); static constexpr auto get = &EHType::vertex  ; };
    struct NERangeTraits { using SEHType = BEH; using EHType = BEHandle; static constexpr size_t count = numNeighbors(); static constexpr auto get = &EHType::neighbor; };
    SubEntityHandleRange< VRangeTraits>  vertices() const { return SubEntityHandleRange< VRangeTraits>(*this); }
    SubEntityHandleRange<NERangeTraits> neighbors() const { return SubEntityHandleRange<NERangeTraits>(*this); }

    typename _H::value_ptr dataPtr() const { return m_mesh.m_boundaryEdgeData.getPtr(m_idx); }

private:
    HEH m_volumeCast() const { return HEH(m_mesh.m_bdryEBdryIdxToVolIdx(m_idx), m_mesh); }
    friend class HEHandle<_Mesh>;
};

} // namespace _TriMeshHandles

template<class _VertexData, class _HalfEdgeData, class _TriData, class _BoundaryVertexData, class _BoundaryEdgeData>
struct HandleTraits<TriMesh<_VertexData, _HalfEdgeData, _TriData, _BoundaryVertexData, _BoundaryEdgeData>> {
    template<class _Mesh> using  VHandle = _TriMeshHandles:: VHandle<_Mesh>; // Vertex
    template<class _Mesh> using HEHandle = _TriMeshHandles::HEHandle<_Mesh>; // Half-edge
    template<class _Mesh> using  THandle = _TriMeshHandles:: THandle<_Mesh>; // Triangle
    template<class _Mesh> using BVHandle = _TriMeshHandles::BVHandle<_Mesh>; // Boundary vertex
    template<class _Mesh> using BEHandle = _TriMeshHandles::BEHandle<_Mesh>; // Boundary edge
};

// Range traits for all tri handle types: get the corresponding entity counts.
template<class _Mesh> struct HandleRangeTraits<_TriMeshHandles:: VHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numVertices()        ; } };
template<class _Mesh> struct HandleRangeTraits<_TriMeshHandles::HEHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numHalfEdges()       ; } };
template<class _Mesh> struct HandleRangeTraits<_TriMeshHandles:: THandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numTris()            ; } };
template<class _Mesh> struct HandleRangeTraits<_TriMeshHandles::BVHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numBoundaryVertices(); } };
template<class _Mesh> struct HandleRangeTraits<_TriMeshHandles::BEHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numBoundaryEdges()   ; } };
