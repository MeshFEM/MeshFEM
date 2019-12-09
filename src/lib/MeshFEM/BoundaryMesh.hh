////////////////////////////////////////////////////////////////////////////////
// BoundaryMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Wrapper to provide access to the boundary of a tri/tet mesh as if it
//      were a mesh datastructure itself.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  09/09/2016 20:06:11
////////////////////////////////////////////////////////////////////////////////
#ifndef BOUNDARYMESH_HH
#define BOUNDARYMESH_HH
#include <type_traits>
#include <MeshFEM/Handles/Handle.hh>
#include <MeshFEM/TemplateHacks.hh>
#include <MeshFEM/MeshDataTraits.hh>

template<class _Mesh, size_t VolK = _Mesh::K, bool IsFemMesh = MeshDataTraits<_Mesh>::isFEMMesh>
struct BoundaryMesh;

template<class _Mesh>
struct GetVolMesh { using type = _Mesh; };
template<class _VolMesh> struct GetVolMesh<BoundaryMesh   <_VolMesh>> { using type = _VolMesh; };

// Metafunction to strip off the boundary mesh wrapper (but still allow cv
// qualifications). This is needed for the handle types, which must be
// templated on their volume mesh type (not this boundary mesh type).
// If there is no boundary mesh wrapper, just return the type.
template<class _BdryMesh>
using CVVolMesh_t = CopyCV_t<_BdryMesh, typename GetVolMesh<typename std::remove_cv<_BdryMesh>::type>::type>;

// Specialization for the boundary of tetrahedral (3D) meshes.
template<class _Mesh>
struct BoundaryMesh<_Mesh, 3, false> {
    using VolMesh = _Mesh;

    static constexpr size_t K = 2;
    BoundaryMesh(_Mesh &mesh) : m_mesh(mesh) { }

    size_t  numVertices() const { return m_mesh.numBoundaryVertices(); }
    size_t     numFaces() const { return m_mesh.numBoundaryFaces(); }
    size_t numHalfEdges() const { return m_mesh.numBoundaryHalfEdges(); }

    template <class _M> using  VHandle = typename _Mesh::template  BVHandle<CVVolMesh_t<_M>>;
    template <class _M> using HEHandle = typename _Mesh::template BHEHandle<CVVolMesh_t<_M>>;
    template <class _M> using  FHandle = typename _Mesh::template  BFHandle<CVVolMesh_t<_M>>;

    // Note: when _Mesh is const, even the non-const accessors actually get const handles.
     VHandle<_Mesh>   vertex(size_t i) { return m_mesh.boundaryVertex(i); }
    HEHandle<_Mesh> halfEdge(size_t i) { return m_mesh.boundaryHalfEdge(i); }
     FHandle<_Mesh>     face(size_t i) { return m_mesh.boundaryFace(i); }

    // These are always const handles: "const const" just becomes "const"
     VHandle<const _Mesh>   vertex(size_t i) const { return m_mesh.boundaryVertex(i); }
    HEHandle<const _Mesh> halfEdge(size_t i) const { return m_mesh.boundaryHalfEdge(i); }
     FHandle<const _Mesh>     face(size_t i) const { return m_mesh.boundaryFace(i); }

    // Get the halfedge pointing from s to e.
    HEHandle<      _Mesh> halfEdge(size_t s, size_t e)       { return halfEdge(halfedgeIndex(s, e)); }
    HEHandle<const _Mesh> halfEdge(size_t s, size_t e) const { return halfEdge(halfedgeIndex(s, e)); }

    // Boundary handle ranges renamed as volume ranges.
    template<class _M> using  VHR = HandleRange<_M, _Mesh::template BVHandle>;
    template<class _M> using HEHR = HandleRange<_M, _Mesh::template BHEHandle>;
    template<class _M> using  FHR = HandleRange<_M, _Mesh::template BFHandle>;

    VHR<      _Mesh>      vertices()       { return m_mesh.     boundaryVertices(); }
    VHR<const _Mesh>      vertices() const { return m_mesh.constBoundaryVertices(); }
    VHR<const _Mesh> constVertices() const { return m_mesh.constBoundaryVertices(); }

    HEHR<      _Mesh>      halfEdges()       { return m_mesh.     boundaryHalfEdges(); }
    HEHR<const _Mesh>      halfEdges() const { return m_mesh.constBoundaryHalfEdges(); }
    HEHR<const _Mesh> constHalfEdges() const { return m_mesh.constBoundaryHalfEdges(); }

    FHR<      _Mesh>      faces()       { return m_mesh.     boundaryFaces(); }
    FHR<const _Mesh>      faces() const { return m_mesh.constBoundaryFaces(); }
    FHR<const _Mesh> constFaces() const { return m_mesh.constBoundaryFaces(); }

    /*! Get the index of the halfedge pointing from s to e */
    int halfedgeIndex(size_t s, size_t e) const {
        assert((s < numVertices()) && (e < numVertices()));

        auto h = vertex(e).halfEdge();
        auto hit = h;
        do {
            if (size_t(hit.tail().index()) == s) {
                return hit.index();
            }
        } while ((hit = hit.cw()) != h);

        return -1;
    }
protected:
    _Mesh &m_mesh;
};

// Specialization for the boundary of triangle (2D) meshes.
template<class _Mesh>
struct BoundaryMesh<_Mesh, 2, false> {
    using VolMesh = _Mesh;
    static constexpr size_t K = 1;
    BoundaryMesh(_Mesh &mesh) : m_mesh(mesh) { }

    size_t numVertices() const { return m_mesh.numBoundaryVertices(); }
    size_t    numEdges() const { return m_mesh.numBoundaryEdges(); }

    template <class _M> using VHandle = typename _Mesh::template BVHandle<CVVolMesh_t<_M>>;
    template <class _M> using EHandle = typename _Mesh::template BEHandle<CVVolMesh_t<_M>>;

    // Note: when _Mesh is const, even the non-const accessors actually get const handles.
    VHandle<_Mesh> vertex(size_t i) { return m_mesh.boundaryVertex(i); }
    EHandle<_Mesh>   edge(size_t i) { return m_mesh.boundaryEdge(i); }

    // These are always const handles: "const const" just becomes "const"
    VHandle<const _Mesh> vertex(size_t i) const { return m_mesh.boundaryVertex(i); }
    EHandle<const _Mesh>   edge(size_t i) const { return m_mesh.boundaryEdge(i); }

    // Boundary handle ranges renamed as volume ranges.
    template<class _M> using  VHR = HandleRange<_M, _Mesh::template BVHandle>;
    template<class _M> using  EHR = HandleRange<_M, _Mesh::template BEHandle>;

    VHR<      _Mesh>      vertices()       { return m_mesh.     boundaryVertices(); }
    VHR<const _Mesh>      vertices() const { return m_mesh.constBoundaryVertices(); }
    VHR<const _Mesh> constVertices() const { return m_mesh.constBoundaryVertices(); }

    EHR<      _Mesh>      edges()       { return m_mesh.     boundaryEdges(); }
    EHR<const _Mesh>      edges() const { return m_mesh.constBoundaryEdges(); }
    EHR<const _Mesh> constEdges() const { return m_mesh.constBoundaryEdges(); }
protected:
    _Mesh &m_mesh;
};

// For the boundary of FEMMesh, we also need to access to nodes and elments
// Thankfully all dimensions of FEMMesh have a uniform interface, so no
// further specialization is needed.
// We derive from the non-FEMMesh BoundaryMesh class to add node and element
// access.
template<class _FEMMesh, size_t VolK>
struct BoundaryMesh<_FEMMesh, VolK, true> : public BoundaryMesh<_FEMMesh, VolK, false> {
    using Base = BoundaryMesh<_FEMMesh, VolK, false>;
    using Base::Base;
    using VolMesh = _FEMMesh;

    size_t numElementNodes() const { return m_mesh.numBoundaryElementNodes(); }
    size_t    numEdgeNodes() const { return m_mesh.numBoundaryEdgeNodes();    }
    size_t  numVertexNodes() const { return m_mesh.numBoundaryVertexNodes();  }

    size_t        numNodes() const { return m_mesh.numBoundaryNodes(); }
    size_t     numElements() const { return m_mesh.numBoundaryElements(); }

    // Note: in the 2D case, there is a naming collision and boundary
    // edge/element handles are both called BEHandle (with the latter
    // overriding). This should be fine since boundary elements derive from
    // boundary edge)
    template <class _M> using EHandle = typename _FEMMesh::template BEHandle<CVVolMesh_t<_M>>;
    template <class _M> using NHandle = typename _FEMMesh::template BNHandle<CVVolMesh_t<_M>>;

    // Note: when _FEMMesh is const, even the non-const accessors actually get const handles.
    EHandle<_FEMMesh> element(size_t i) { return m_mesh.boundaryElement(i); }
    NHandle<_FEMMesh>    node(size_t i) { return m_mesh.boundaryNode(i); }

    // These are always const handles: "const const" just becomes "const"
    EHandle<const _FEMMesh> element(size_t i) const { return m_mesh.boundaryElement(i); }
    NHandle<const _FEMMesh>    node(size_t i) const { return m_mesh.boundaryNode(i); }

    // Boundary handle ranges renamed as volume ranges.
    template<class _M> using  NHR = HandleRange<_M, _FEMMesh::template BNHandle>;
    template<class _M> using  EHR = HandleRange<_M, _FEMMesh::template BEHandle>;

    NHR<      _FEMMesh>      nodes()       { return m_mesh.boundaryNodes(); }
    NHR<const _FEMMesh>      nodes() const { return m_mesh.constBoundaryNodes(); }
    NHR<const _FEMMesh> constNodes() const { return m_mesh.constBoundaryNodes(); }

    EHR<      _FEMMesh>      elements()       { return m_mesh.boundaryElements(); }
    EHR<const _FEMMesh>      elements() const { return m_mesh.constBoundaryElements(); }
    EHR<const _FEMMesh> constElements() const { return m_mesh.constBoundaryElements(); }
protected:
    using Base::m_mesh;
};

#endif /* end of include guard: BOUNDARYMESH_HH */
