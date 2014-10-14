#ifndef FEMMESH_HH
#define FEMMESH_HH

#include "Handle.hh"
////////////////////////////////////////////////////////////////////////////////
// Adaptor for use by the generic FEMMesh template.
////////////////////////////////////////////////////////////////////////////////
template<size_t _K>
class MeshDatastructure { };
template<> class MeshDatastructure<2> {
    // 2D Simplices -> Triangle Mesh
    template <class VData, class TData, class BVData, class BEData>
    using Mesh = TriMesh<VData, TMEmptyData, TData, BVData, BEData>;
};
template<> class MeshDatastructure<3> {
    // 3D Simplices -> Tet Mesh
    template <class VData, class TData, class BVData, class BFData>
    using Mesh = TetMesh< VData, TMEmptyData, TData,
                         BVData, TMEmptyData, BFData>;
};

// Store positions on all nodes (this will allow support for nonlinear
// elasticity in the future). Typically, the edge node positions will be the
// average of the edge endpoint node positions.
template<size_t _K, size_t _Deg, class EmbeddingSpace>
struct NodeData {
    EmbeddingSpace p;
};

template<size_t _K, size_t _Deg, class EmbeddingSpace>
using EmbeddingSpace = TMEmptyData;

template<size_t _K, size_t _Deg, class EmbeddingSpace,
         template class <size_t, size_t, class> _VertexData = EmbeddedEmptyData,
         template class <size_t, size_t, class> _NodeData   = NodeData,
         template class <size_t, size_t, class> _VolumeElementData   = LinearlyEmbeddedElement,
         template class <size_t, size_t, class> _BoundaryVertexData = EmbeddedEmptyData,
         template class <size_t, size_t, class> _BoundaryNodeData = EmbeddedEmptyData,
         template class <size_t, size_t, class> _BoundaryElementData = LinearlyEmbeddedElement>
class FEMMesh : public typename MeshDatastructure<_K>::template Mesh<
    // Store mesh-tied entities ({boundary,volume} {vertex,element} data) in the
    // underlying mesh data structure. The node data is managed by this data
    // structure.
    _VertexData<_K, _Deg, EmbeddingSpace>,
    _VolumeElementData<_K, _Deg, EmbeddingSpace>,
    _BoundaryVertexData<_K - 1, _Deg, EmbeddingSpace>,
    _BoundaryElementData<_K - 1, _Deg, EmbeddingSpace>>
{
    typedef          _VertexData<_K    , _Deg, EmbeddingSpace> VertexData;
    typedef            _NodeData<_K    , _Deg, EmbeddingSpace> NodeData;
    typedef   _VolumeElementData<_K    , _Deg, EmbeddingSpace> ElementData;
    typedef  _BoundaryVertexData<_K - 1, _Deg, EmbeddingSpace> BoundaryNodeData;
    typedef    _BoundaryNodeData<_K - 1, _Deg, EmbeddingSpace> BoundaryNodeData;
    typedef _BoundaryElementData<_K - 1, _Deg, EmbeddingSpace> BoundaryElementData;

    typedef typename MeshDatastructure<_K>::template Mesh<VertexData,
        ElementData, BoundaryElementData, BoundaryElementData>   BaseMesh;

    // Node and Element Entity handles (declared out-of-line in FEMMesh.inl).
    template<template<class, class, class, class> class _HType> class  EHandle;
    template<template<class, class, class, class> class _HType> class BEHandle;
    template<template<class, class, class, class> class _HType> class  NHandle;
    template<template<class, class, class, class> class _HType> class BNHandle;
    typedef  EHandle<Handle>         ElementHandle; typedef  EHandle<ConstHandle>         ConstElementHandle;
    typedef BEHandle<Handle> BoundaryElementHandle; typedef BEHandle<ConstHandle> ConstBoundaryElementHandle;
    typedef  NHandle<Handle>            NodeHandle; typedef  NHandle<ConstHandle>            ConstNodeHandle;
    typedef BNHandle<Handle>    BoundaryNodeHandle; typedef BNHandle<ConstHandle>    ConstBoundaryNodeHandle;

    // Table of **non-[boundary] vertex** [boundary] node indices for each
    // [boundary] element. We needn't store the [boundary] vertex node indices
    // because our mesh data structure knows that.
    std::vector<int>  N;
    std::vector<int> BN;

    std::vector<NodeData>            m_nodeData;
    std::vector<BoundaryNodeData>    m_boundaryNodeData;
    // A pointer to the following is returned when accessing the data of type
    // "TMEmptyData" to avoid allocating the above vectors
    TMEmptyData m_emptyData;

    template<class Mesh, class Subtype, class ConstSubtype, class Data>
    friend class Handle;
    template<class Mesh, class Subtype, class ConstSubtype, class Data>
    friend class ConstHandle;

    // Nodes 0..#Vertices are located on the corresponding vertex.
    // The remaining nodes do not have vertices.
    int m_vertexForNode(int n) const {
        if (n < this->numVertices()) return n;
        else return -1;
    }

    // Boundary Nodes 0..#BdryVertices are located on the corresponding boundary
    // vertex. The remaining nodes do not have vertices.
    int m_boundaryVertexForBoundaryNode(int bn) const {
        if (bn < this->numBoundaryVertices()) return bn;
        else return -1;
    }
};

#endif /* end of include guard: FEMMESH_HH */
