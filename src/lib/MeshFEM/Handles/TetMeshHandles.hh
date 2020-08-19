#include <MeshFEM/MeshDataTraits.hh>
#include "Handle.hh"
#include "Circulator.hh"
#include <queue>

namespace _TetMeshHandles {

// We need to expliclty reference this enclosing scope to hack around an old
// clang bug involving injected class names; make it less verbose
namespace _hndl = ::_TetMeshHandles;

template<class _Mesh> using   _VData = typename MeshDataTraits<_Mesh>::VertexData;
template<class _Mesh> using  _HFData = typename MeshDataTraits<_Mesh>::HalfFaceData;
template<class _Mesh> using  _HEData = typename MeshDataTraits<_Mesh>::HalfEdgeData;
template<class _Mesh> using   _TData = typename MeshDataTraits<_Mesh>::TetData;
template<class _Mesh> using  _BVData = typename MeshDataTraits<_Mesh>::BoundaryVertexData;
template<class _Mesh> using _BHEData = typename MeshDataTraits<_Mesh>::BoundaryHalfEdgeData;
template<class _Mesh> using  _BFData = typename MeshDataTraits<_Mesh>::BoundaryFaceData;

////////////////////////////////////////////////////////////////////////////////
// Vertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class VHandle : public Handle<_Mesh, VHandle, _VData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::VHandle, _VData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using  VH = typename _Mesh::template  VHandle<_Mesh>;
    using BVH = typename _Mesh::template BVHandle<_Mesh>;
    using HFH = typename _Mesh::template HFHandle<_Mesh>;
    using HEH = typename _Mesh::template HEHandle<_Mesh>;
public:
    bool valid() const { return (m_idx >= 0) && (size_t(m_idx) < m_mesh.numVertices()); }

    bool    isBoundary() const { return m_mesh.m_bdryVertexIdx(m_idx) >= 0; }
    BVH boundaryVertex() const { return BVH(m_mesh.m_bdryVertexIdx(m_idx), m_mesh); }
    // Arbitrary half face containing this vertex.
    HFH       halfFace() const { return HFH(m_mesh.m_halfFaceOfVertex(m_idx), m_mesh); }
    // Half edge incident this vertex within halfFace().
    HEH       halfEdge() const {
        const auto &hf = halfFace();
        HEH he = hf.halfEdge(0); if (he.tip().index() == m_idx) return he;
            he = hf.halfEdge(1); if (he.tip().index() == m_idx) return he;
            he = hf.halfEdge(2); if (he.tip().index() == m_idx) return he;
        assert(false);
        return HEH(-1, m_mesh);
    }

    // Call `visitor(ei)` for each incident element `ei`.
    // We need a BFS algorithm to visit all incident elements since there is
    // no simple strategy to visit each exactly once like for triangle meshes.
    template<class F>
    void visitIncidentElements(F &&visitor) {
        auto startHE = halfEdge(); // start at an arbitrary incident half-edge
        visitor(startHE.tet().index());
        std::vector<size_t> visited{size_t(startHE.tet().index())};
        auto isVisited = [&](size_t ti) { return std::find(visited.begin(), visited.end(), ti) != visited.end(); };

        std::queue<size_t> bfsQueue;
        bfsQueue.push(startHE.index());
        while (!bfsQueue.empty()) {
            size_t u = bfsQueue.front();
            bfsQueue.pop();

            // Circulate around the tet corner, adding half-edges in unvisited
            // adjacent tets.
            for (const auto &he : m_mesh.halfEdge(u).tipCirculator()) {
                if (!he.isBoundary()) {
                    const auto &r = he.radial().prev();
                    size_t ti = r.tet().index();
                    if (isVisited(ti)) continue;
                    visited.push_back(ti);
                    visitor(ti);
                    bfsQueue.push(r.index());
                }
            }
        }
    }

    // Identity operation for unified writing of surface and volume meshes
    // (since point data is typically stored only on the volume vertex)
    VH volumeVertex() const { return VH(m_idx, m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_vertexData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// HalfFace Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class HFHandle : public Handle<_Mesh, HFHandle, _HFData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::HFHandle, _HFData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using  VH = typename _Mesh::template  VHandle<_Mesh>;
    using HFH = typename _Mesh::template HFHandle<_Mesh>;
    using  TH = typename _Mesh::template  THandle<_Mesh>;
    using BFH = typename _Mesh::template BFHandle<_Mesh>;
    using  HE = typename _Mesh::template HEHandle<_Mesh>;
public:
    bool         valid() const { return m_idx >= 0 && m_idx < m_mesh.numHalfFaces(); }
    bool    isBoundary() const { return m_mesh.m_oppFaceIdx(m_idx) < 0; }
    BFH   boundaryFace() const { return BFH(m_mesh.m_bdryFaceOfVolumeFace(m_idx), m_mesh); }
    HFH       opposite() const { return HFH(m_mesh.m_oppFaceIdx(m_idx), m_mesh); }
    // Dimension-independent terminology:
    BFH boundaryEntity() const { return boundaryFace(); }

    static constexpr size_t numVertices() { return 3; }
    VH vertex(size_t i) const { return VH(m_mesh.m_vertexOfHalfFace(i, m_idx), m_mesh); }
    TH            tet() const { return TH(m_mesh.m_tetOfHF(m_idx), m_mesh); }
    TH        simplex() const { return tet(); }
    TH        element() const { return tet(); }

    HE halfEdge(size_t lhe) const { return HE(m_mesh.m_heOfHF(m_idx, lhe), m_mesh); }

    // Support range-based for over vertices
    struct  VRangeTraits { using SEHType = VH; using EHType = HFHandle; static constexpr size_t count = numVertices(); static constexpr SEHType (EHType::*get)(size_t) const = &EHType::vertex; };
    SubEntityHandleRange<VRangeTraits> vertices() const { return SubEntityHandleRange<VRangeTraits>(*this); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_halfFaceData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// HalfEdge Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class HEHandle : public Handle<_Mesh, HEHandle, _HEData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::HEHandle, _HEData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using   VH = typename _Mesh::template   VHandle<_Mesh>;
    using  HEH = typename _Mesh::template  HEHandle<_Mesh>;
    using  HFH = typename _Mesh::template  HFHandle<_Mesh>;
    using   TH = typename _Mesh::template   THandle<_Mesh>;
    using BHEH = typename _Mesh::template BHEHandle<_Mesh>;
public:
    bool      valid() const { return (m_idx >= 0) && (size_t(m_idx) < m_mesh.numHalfEdges()); }
    // Is this (radially opposite) a boundary half-edge?
    bool isBoundary() const { return halfFace().isBoundary(); }
    // Boundary half-edge corresponding to this half-edge (only valid when isBoundary() is true).
    BHEH boundaryHalfEdge() const { if (m_idx < -1) return BHEH(-2 - m_idx, m_mesh); // This is actually an encoded boundary half-edge
                                    else            return BHEH(m_mesh.m_bdryHEOfHE(m_idx), m_mesh); }

    HFH halfFace() const { return HFH(m_mesh.m_faceOfHE(m_idx), m_mesh); }
    TH       tet() const { return (m_idx < -1) ? TH(-1, m_mesh) : TH(m_mesh. m_tetOfHE(m_idx), m_mesh); }
    TH   simplex() const { return tet(); }
    TH   element() const { return tet(); }

    // circulate in halfFace()
    HEH     next() const { return HEH(m_mesh.  m_nextHE(m_idx), m_mesh); }
    HEH     prev() const { return HEH(m_mesh.  m_prevHE(m_idx), m_mesh); }

    // Opposite within tet()
    HEH     mate() const { return (m_idx < -1) ? HEH(-1, m_mesh) : HEH(m_mesh.  m_mateHE(m_idx), m_mesh); }
    // Opposite within face halfFace()->opposite()
    HEH   radial() const { return (m_idx < -1) ? boundaryHalfEdge().volumeHalfEdge() : HEH(m_mesh.m_radialHE(m_idx), m_mesh); }

    // Circulate within tet() counter-clockwise around tip()'s (outward-pointing) normal
    // Note: half-edges circulate *clockwise* when viewed from outside (half-faces are inward-oriented).
    HEH ccw() const { return next().mate(); }
    HEH  cw() const { return mate().prev(); }

    // Range circulating counter-clockwise within this tet around the tip corner
    CirculatorRange<HEH> tipCirculator() const { return CirculatorRange<HEH>(HEH(m_idx, m_mesh)); } // Must use HEH() instead of *this for correct derived type.

    // Dimension-independent terminology:
    BHEH boundaryEntity() const { return boundaryHalfEdge(); }

    // Call `visitor(ei)` for each incident tet `ei`.
    template<class F>
    void visitIncidentElements(F &&visitor) const {
        auto he = *this;
        do {
            visitor(he.tet().index());
            he = he.mate().radial();
        } while (he && (he != *this)); // Stop if we reach the boundary or return to where we started.

        if (!he) {
            // If we hit the boundary, also traverse in the opposite direction, in case we started from a non-boundary half-edge
            he = *this;
            while (!he.isBoundary()) {
                he = he.radial().mate();
                visitor(he.tet().index());
            }
        }
    }

    static constexpr size_t numVertices() { return 2; }
    VH  tip() const { return (m_idx < -1) ? boundaryHalfEdge(). tip().volumeVertex() : VH(m_mesh. m_tipOfHE(m_idx), m_mesh); }
    VH tail() const { return (m_idx < -1) ? boundaryHalfEdge().tail().volumeVertex() : VH(m_mesh.m_tailOfHE(m_idx), m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_halfEdgeData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// Tet Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class THandle : public Handle<_Mesh, THandle, _TData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::THandle, _TData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using  TH = typename _Mesh::template  THandle<_Mesh>;
    using  VH = typename _Mesh::template  VHandle<_Mesh>;
    using HFH = typename _Mesh::template HFHandle<_Mesh>;
    using HEH = typename _Mesh::template HEHandle<_Mesh>;
public:
    bool valid() const { return (m_idx >= 0) && (size_t(m_idx) < m_mesh.numTets()); }
    bool isBoundary() const {
        return halfFace(0).isBoundary() || halfFace(1).isBoundary() ||
               halfFace(2).isBoundary() || halfFace(3).isBoundary();
    }

    // Note: true neighbor count can be less than 4; must check if neighbor(i)
    // is valid.
    static constexpr size_t numNeighbors() { return 4; }
    static constexpr size_t numVertices()  { return 4; }

     VH   vertex(size_t i) const { return  VH(m_mesh.m_vertexOfTet(i, m_idx), m_mesh); }
     TH neighbor(size_t i) const { return  TH(m_mesh.m_tetAdjTet(i, m_idx), m_mesh); }
    HFH halfFace(size_t i) const { return HFH(m_mesh.m_faceOfTet(i, m_idx), m_mesh); }
    HEH halfEdge(size_t i) const { return (valid() && (i < 12)) ? HEH(12 * m_idx + i, m_mesh) : HEH(-1, m_mesh); }

    // Support range-based for over vertices, neighboring tets, and half-faces (interfaces)
    struct  VRangeTraits { using SEHType =  VH; using EHType = THandle; static constexpr size_t count = numVertices() ; static constexpr auto get = &EHType::vertex  ; };
    struct NTRangeTraits { using SEHType =  TH; using EHType = THandle; static constexpr size_t count = numNeighbors(); static constexpr auto get = &EHType::neighbor; };
    struct HFRangeTraits { using SEHType = HFH; using EHType = THandle; static constexpr size_t count = numNeighbors(); static constexpr auto get = &EHType::halfFace; };
    struct HERangeTraits { using SEHType = HEH; using EHType = THandle; static constexpr size_t count =             12; static constexpr auto get = &EHType::halfEdge; };
    SubEntityHandleRange< VRangeTraits>   vertices() const { return SubEntityHandleRange< VRangeTraits>(*this); }
    SubEntityHandleRange<NTRangeTraits>  neighbors() const { return SubEntityHandleRange<NTRangeTraits>(*this); }
    SubEntityHandleRange<HFRangeTraits>  halfFaces() const { return SubEntityHandleRange<HFRangeTraits>(*this); }
    SubEntityHandleRange<HFRangeTraits> interfaces() const { return SubEntityHandleRange<HFRangeTraits>(*this); }
    SubEntityHandleRange<HFRangeTraits>  halfEdges() const { return SubEntityHandleRange<HFRangeTraits>(*this); }

    // Dimension-independent terminology:
    //  interface of a tet is a half-face
    //  interface of a tri is a half-edge
    HFH interface(size_t i) const { return halfFace(i); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_tetData.getPtr(m_idx); }
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
    using   VH = typename _Mesh::template   VHandle<_Mesh>;
    using  BFH = typename _Mesh::template  BFHandle<_Mesh>;
    using BHEH = typename _Mesh::template BHEHandle<_Mesh>;
public:
    bool valid() const { return (m_idx >= 0) && (size_t(m_idx) < m_mesh.numBoundaryVertices()); }
    // The boundary of a tet mesh has no boundary.
    bool isBoundary() const { return false; }

    // Get handle for tet mesh vertex corresponding to this boundary vertex.
     VH volumeVertex() const { return VH(m_mesh.m_vertexForBdryVertex(m_idx), m_mesh); }
    // Get some incident boundary face. This works because the incident half-face
    // for a vertex on the boundary is guaranteed to be on the boundary.
     BFH        face() const { assert(valid()); BFH bf = volumeVertex().halfFace().boundaryFace(); assert(bf); return bf; }
    BHEH    halfEdge() const { return BHEH(m_mesh.m_bdryHEOfBdryVertex(m_idx), m_mesh); }

    CirculatorRange<BHEH> incidentHalfEdges() const { return CirculatorRange<BHEH>(halfEdge()); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_boundaryVertexData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary HalfEdge Handles
// Index is of the form 3 * boundary_face_idx + corner
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class BHEHandle : public Handle<_Mesh, BHEHandle, _BHEData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::BHEHandle, _BHEData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    using  HEH = typename _Mesh::template  HEHandle<_Mesh>;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using  BVH = typename _Mesh::template  BVHandle<_Mesh>;
    using  BFH = typename _Mesh::template  BFHandle<_Mesh>;
    using BHEH = typename _Mesh::template BHEHandle<_Mesh>;
public:
    bool valid() const { return (m_idx >= 0) && (size_t(m_idx) < m_mesh.numBoundaryHalfEdges()); }
    // The boundary of a tet mesh has no border.
    bool isBoundary()     const { return false; }
    bool isBoundaryEdge() const { return false; }

    BHEH opposite() const { return BHEH(m_mesh.m_oppositeBdryHE(m_idx), m_mesh); }
    BHEH  primary() const {   int opp = m_mesh.m_oppositeBdryHE(m_idx); return BHEH((opp < m_idx) ? opp : m_idx, m_mesh); }
    BHEH     next() const { return BHEH(m_mesh.template m_bdryHE<_Mesh::Direction::NEXT>(m_idx), m_mesh); }
    BHEH     prev() const { return BHEH(m_mesh.template m_bdryHE<_Mesh::Direction::PREV>(m_idx), m_mesh); }
     BVH      tip() const { return  BVH(m_mesh.template m_bdryVertexOfBdryHE<_Mesh::HEVertex::TIP>(m_idx), m_mesh); }
     BVH     tail() const { return  BVH(m_mesh.template m_bdryVertexOfBdryHE<_Mesh::HEVertex::TAIL>(m_idx), m_mesh); }

    // Circulation around tip
    BHEH  ccw() const { return opposite().prev(); }
    BHEH   cw() const { return next().opposite(); }

     BFH face() const { return BFH(m_mesh.m_bdryFaceOfBdryHE(m_idx), m_mesh); }
     BFH  tri() const { return face(); } // for compatibility with TriMesh's HEHandle

    HEH volumeHalfEdge() const { return HEH(m_mesh.m_volHEOfBdryHE(m_idx).index(), m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_boundaryHalfEdgeData.getPtr(m_idx); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Face Handles
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
class BFHandle : public Handle<_Mesh, BFHandle, _BFData<_Mesh>> {
protected:
    using _H = Handle<_Mesh, _hndl::BFHandle, _BFData<_Mesh>>;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    using  BVH = typename _Mesh::template  BVHandle<_Mesh>;
    using  HFH = typename _Mesh::template  HFHandle<_Mesh>;
    using  BFH = typename _Mesh::template  BFHandle<_Mesh>;
    using BHEH = typename _Mesh::template BHEHandle<_Mesh>;
public:
    bool valid() const { return (m_idx >= 0) && (size_t(m_idx) < m_mesh.numBoundaryFaces()); }

    static constexpr size_t numNeighbors() { return 3; }
    static constexpr size_t numVertices()  { return 3; }

     HFH   volumeHalfFace() const { return  HFH(m_mesh.     m_faceForBdryFace(   m_idx), m_mesh); }
     BVH   vertex(size_t i) const { return  BVH(m_mesh.m_bdryVertexOfBdryFace(i, m_idx), m_mesh); }
     BFH neighbor(size_t i) const { return  BFH(m_mesh. m_bdryFaceAdjBdryFace(i, m_idx), m_mesh); }
    BHEH halfEdge(size_t i) const { return BHEH(m_mesh.    m_bdryHEOfBdryFace(i, m_idx), m_mesh); }

     HFH opposite() const { return volumeHalfFace(); }

    // Support range-based for over boundary vertices, neighboring boundary triangles, and half-edges
    struct   VRangeTraits { using SEHType =  BVH; using EHType = BFHandle; static constexpr size_t count = numVertices() ;  static constexpr auto get = &EHType::vertex  ; };
    struct  NTRangeTraits { using SEHType =  BFH; using EHType = BFHandle; static constexpr size_t count = numNeighbors();  static constexpr auto get = &EHType::neighbor; };
    struct  HERangeTraits { using SEHType = BHEH; using EHType = BFHandle; static constexpr size_t count = numNeighbors();  static constexpr auto get = &EHType::halfEdge; };
    SubEntityHandleRange< VRangeTraits>  vertices() const { return SubEntityHandleRange< VRangeTraits>(*this); }
    SubEntityHandleRange<NTRangeTraits> neighbors() const { return SubEntityHandleRange<NTRangeTraits>(*this); }
    SubEntityHandleRange<HERangeTraits> halfEdges() const { return SubEntityHandleRange<HERangeTraits>(*this); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return m_mesh.m_boundaryFaceData.getPtr(m_idx); }
};

}

template<class _VertexData, class _HalfFaceData, class _HalfEdgeData, class _TetData, class _BoundaryVertexData, class _BoundaryHalfEdgeData, class _BoundaryFaceData>
struct HandleTraits<TetMesh<_VertexData, _HalfFaceData, _HalfEdgeData, _TetData, _BoundaryVertexData, _BoundaryHalfEdgeData, _BoundaryFaceData>> {
    template<class _Mesh> using   VHandle = _TetMeshHandles::  VHandle<_Mesh>; // Vertex
    template<class _Mesh> using  HFHandle = _TetMeshHandles:: HFHandle<_Mesh>; // Half-face
    template<class _Mesh> using  HEHandle = _TetMeshHandles:: HEHandle<_Mesh>; // Half-edge
    template<class _Mesh> using   THandle = _TetMeshHandles::  THandle<_Mesh>; // Tetrahedron
    template<class _Mesh> using  BVHandle = _TetMeshHandles:: BVHandle<_Mesh>; // Boundary vertex
    template<class _Mesh> using BHEHandle = _TetMeshHandles::BHEHandle<_Mesh>; // Boundary half-edge
    template<class _Mesh> using  BFHandle = _TetMeshHandles:: BFHandle<_Mesh>; // Boundary face
};

// Range traits for all tet handle types: get the corresponding entity counts.
template<class _Mesh> struct HandleRangeTraits<_TetMeshHandles::  VHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numVertices()         ; } };
template<class _Mesh> struct HandleRangeTraits<_TetMeshHandles:: HFHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numHalfFaces()        ; } };
template<class _Mesh> struct HandleRangeTraits<_TetMeshHandles:: HEHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numHalfEdges()        ; } };
template<class _Mesh> struct HandleRangeTraits<_TetMeshHandles::  THandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numTets()             ; } };
template<class _Mesh> struct HandleRangeTraits<_TetMeshHandles:: BVHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numBoundaryVertices() ; } };
template<class _Mesh> struct HandleRangeTraits<_TetMeshHandles::BHEHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numBoundaryHalfEdges(); } };
template<class _Mesh> struct HandleRangeTraits<_TetMeshHandles:: BFHandle<_Mesh>> { static size_t entityCount(const _Mesh &m) { return m.numBoundaryFaces()    ; } };
