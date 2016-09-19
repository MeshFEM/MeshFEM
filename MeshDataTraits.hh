////////////////////////////////////////////////////////////////////////////////
// MeshDataTraits.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides access to the typenames of data stored on the mesh entites.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  09/11/2016 07:39:40
////////////////////////////////////////////////////////////////////////////////
#ifndef MESHDATATRAITS_HH
#define MESHDATATRAITS_HH

template<class Mesh>
struct MeshDataTraits;

// Const meshes have the same data as non-const
template<class Mesh>
struct MeshDataTraits<const Mesh> : public MeshDataTraits<Mesh> { };

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
template<class VertexData, class HalfEdgeData, class TriData, class BoundaryVertexData, class BoundaryEdgeData>
class TriMesh;
template<class VertexData, class HalfFaceData, class HalfEdgeData, class TetData, class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh;
template<size_t _K, size_t _Deg, class _EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
class FEMMesh;

////////////////////////////////////////////////////////////////////////////////
// Tetrahedral mesh traits
////////////////////////////////////////////////////////////////////////////////
template<class _VertexData, class _HalfFaceData, class _HalfEdgeData, class _TetData, class _BoundaryVertexData, class _BoundaryHalfEdgeData, class _BoundaryFaceData>
struct MeshDataTraits<TetMesh<_VertexData, _HalfFaceData, _HalfEdgeData, _TetData, _BoundaryVertexData, _BoundaryHalfEdgeData, _BoundaryFaceData>>
{
    static constexpr bool isFEMMesh = false;
    size_t K = 3;

    using VertexData           = _VertexData;
    using HalfFaceData         = _HalfFaceData;
    using HalfEdgeData         = _HalfEdgeData;
    using TetData              = _TetData;
    using BoundaryVertexData   = _BoundaryVertexData;
    using BoundaryHalfEdgeData = _BoundaryHalfEdgeData;
    using BoundaryFaceData     = _BoundaryFaceData;
};

////////////////////////////////////////////////////////////////////////////////
// Triangle mesh traits
////////////////////////////////////////////////////////////////////////////////
template<class _VertexData, class _HalfEdgeData, class _TriData, class _BoundaryVertexData, class _BoundaryEdgeData>
struct MeshDataTraits<TriMesh<_VertexData, _HalfEdgeData, _TriData, _BoundaryVertexData, _BoundaryEdgeData>>
{
    static constexpr bool isFEMMesh = false;
    size_t K = 2;

    using VertexData         = _VertexData;
    using HalfEdgeData       = _HalfEdgeData;
    using TriData            = _TriData;
    using BoundaryVertexData = _BoundaryVertexData;
    using BoundaryEdgeData   = _BoundaryEdgeData;
};

////////////////////////////////////////////////////////////////////////////////
// FEMMesh traits
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class _EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData>
struct MeshDataTraits<FEMMesh<_K, _Deg, _EmbeddingSpace, _FEMData>> {
    static constexpr bool isFEMMesh = true;
    size_t K = _K;

    using FD = _FEMData<_K, _Deg, _EmbeddingSpace>;
    using VertexData          = typename FD::Vertex;
    using NodeData            = typename FD::Node;
    using ElementData         = typename FD::Element;
    using BoundaryVertexData  = typename FD::BoundaryVertex;
    using BoundaryNodeData    = typename FD::BoundaryNode;
    using BoundaryElementData = typename FD::BoundaryElement;

    // Allow base mesh (TriMesh or TetMesh) to determine what simplex/boundary
    // simplex data to store...
    // Element data is stored on the triangles/tets
    // Boundary element data is stored on the boundary faces/edges
    using TriData = ElementData;
    using TetData = ElementData;
    using BoundaryFaceData = BoundaryElementData;
    using BoundaryEdgeData = BoundaryElementData;

    // FEMMesh never stores data on half-faces and half-edges (TetMesh) or half-edges (TriMesh)
    using HalfFaceData         = TMEmptyData;
    using HalfEdgeData         = TMEmptyData;
    using BoundaryHalfEdgeData = TMEmptyData;
};

// template<class _FEMMesh>
// struct MeshDataTraits<BoundaryFEMMesh<_FEMMesh>> : public MeshDataTraits<_FEMMesh> { };

#endif /* end of include guard: MESHDATATRAITS_HH */
