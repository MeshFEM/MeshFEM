#include "Simplex.hh"
////////////////////////////////////////////////////////////////////////////////
// Vertex Handles: just add node() to access the node at this vertex
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
template<class _Mesh, template<class, class, class, class> class _HType>
class FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
VHandle : public BaseMesh::template VHandle<_Mesh, _HType> {
    typedef typename BaseMesh::template VHandle<_Mesh, _HType> Base;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    typedef typename _Mesh::template NHandle<_Mesh, _HType>  NH;
public:
    NH node() const { return NH(m_mesh.m_nodeForVertex(m_idx), m_mesh); }
};

////////////////////////////////////////////////////////////////////////////////
// Node Handles
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
template<class _Mesh, template<class, class, class, class> class _HType>
class FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
NHandle : public _HType<_Mesh, NHandle<_Mesh, Handle>, NHandle<_Mesh, ConstHandle>, NodeData> {
    typedef _HType<_Mesh, NHandle<_Mesh, Handle>, NHandle<_Mesh, ConstHandle>, NodeData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    typedef typename _Mesh::template  VHandle<_Mesh, _HType>  VH;
    typedef typename _Mesh::template BNHandle<_Mesh, _HType> BNH;
public:
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numNodes(); }

    int edgeNodeIndex() const { return m_mesh.m_edgeNodeIndex(m_idx); }
    bool isEdgeNode()   const { return edgeNodeIndex() >= 0; }
    bool isVertexNode() const { return m_mesh.m_vertexForNode(m_idx) >= 0; }

    // Get the vertex this node is sitting on (if any)
    VH vertex() const { return VH(m_mesh.m_vertexForNode(m_idx), m_mesh); }

    // Get the boundary node collocated with this volume node.
    BNH boundaryNode() const { return vertex().boundaryVertex().node(); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return &m_mesh.m_nodeData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Element Handles
// TODO: reimplement traversal operations to stay on the derived mesh (e.g.
// neighbor, vertex, ...)
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
template<class _Mesh, template<class, class, class, class> class _HType>
class FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
EHandle : public BaseMesh::template SHandle<_Mesh, _HType> {
    typedef typename BaseMesh::template SHandle<_Mesh, _HType> Base;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    typedef typename _Mesh::template NHandle<_Mesh, _HType>  NH;
public:
    constexpr size_t numNodes()  const { return Simplex::numNodes(_K, _Deg); }
    NH node(size_t i) const { return NH(m_mesh.m_nodeOfElement(i, m_idx), m_mesh); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Node Handles
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
template<class _Mesh, template<class, class, class, class> class _HType>
class FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
BNHandle : public _HType<_Mesh, BNHandle<_Mesh, Handle>, BNHandle<_Mesh, ConstHandle>, BoundaryNodeData> {
    typedef _HType<_Mesh, BNHandle<_Mesh, Handle>, BNHandle<_Mesh, ConstHandle>, BoundaryNodeData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    typedef typename _Mesh::template  NHandle<_Mesh, _HType>  NH;
    typedef typename _Mesh::template BVHandle<_Mesh, _HType> BVH;
public:
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numNodes(); }

    int edgeNodeIndex() const { return m_mesh.m_bdryEdgeNodeIndex(m_idx); }
    bool   isEdgeNode() const { return edgeNodeIndex() >= 0; }
    bool isVertexNode() const { return m_mesh.m_boundaryVertexForBoundaryNode(m_idx) >= 0; }

    BVH        vertex() const { return BVH(m_mesh.m_boundaryVertexForBoundaryNode(m_idx), m_mesh); }
     NH    volumeNode() const { return vertex().volumeVertex().node(); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return &m_mesh.m_boundaryNodeData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Vertex Handles: just add node() to access the boundary node at this
// boundary vertex
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
template<class _Mesh, template<class, class, class, class> class _HType>
class FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
BVHandle : public BaseMesh::template BVHandle<_Mesh, _HType> {
    typedef typename BaseMesh::template BVHandle<_Mesh, _HType> Base;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    typedef typename _Mesh::template BNHandle<_Mesh, _HType> BNH;
public:
    BNH node() const { return BNH(m_mesh.m_nodeForBoundaryVertex(m_idx), m_mesh); }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Element Handles
// TODO: reimplement traversal operations to stay on the derived mesh (e.g.
// neighbor, vertex, ...)
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
template<class _Mesh, template<class, class, class, class> class _HType>
class FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
BEHandle : public BaseMesh::template BSHandle<_Mesh, _HType> {
    typedef typename BaseMesh::template BSHandle<_Mesh, _HType> Base;
    using Base::m_mesh; using Base::m_idx; using Base::Base;
    typedef typename _Mesh::template BNHandle<_Mesh, _HType> BNH;
public:
    constexpr size_t numNodes()  const { return Simplex::numNodes(_K - 1, _Deg); }
    BNH node(size_t i) const { return BNH(m_mesh.m_nodeOfBdryElement(i, m_idx), m_mesh); }
};

////////////////////////////////////////////////////////////////////////////
/*! Constructor builds up the edge node connectivity after constructing the
//  underlying mesh.
*///////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
template<typename Elements, typename Vertices>
FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
FEMMesh(const Elements &elems, const Vertices &vertices)
    : BaseMesh(elems, vertices.size()) {
    if (_Deg == 2) {
        std::map<UnorderedPair, int> edgeNodes;
        // Construct an edge node for each volume edge.
        // We could optimize this in the future by using BaseMesh's
        // traversal operations.
        size_t edgesPerSimplex = Simplex::numEdges(_K);
        size_t numEdgeNodes = 0;
        m_edgeForEdgeNode.clear();
        m_N.resize(BaseMesh::numSimplices() * edgesPerSimplex);
        for (size_t si = 0; si < BaseMesh::numSimplices(); ++si) {
            auto s = BaseMesh::simplex(si);
            for (size_t ei = 0; ei < edgesPerSimplex; ++ei) {
                UnorderedPair edge(s.vertex(Simplex::edgeStartNode(ei)).index(),
                                   s.vertex(  Simplex::edgeEndNode(ei)).index());
                auto it = edgeNodes.find(edge);
                size_t nodeIdx;
                if (it == edgeNodes.end()) {
                    edgeNodes[edge] = nodeIdx = m_edgeForEdgeNode.size();
                    m_edgeForEdgeNode.push_back(edge);
                }
                else {
                    nodeIdx = it->second;
                    // Note: we can't erase the entry in the tet case
                    // because many elements share the same edge. It would
                    // would be more efficient in the triangle case to
                    // specialize and delete edge entry as soon as possible.
                }
                m_N[si * edgesPerSimplex + ei] = nodeIdx;
            }
        }
        edgeNodes.clear();

        // Construct a boundary edge node for each surface edge.
        // Again, we could optimize this by using BaseMesh's traversal
        // operations.
        // Also, for triangle meshes, there is a unique edge node per
        // element (none is shared), so we don't need any gluing.
        size_t numBoundaryEdgeNodes = 0;
        m_edgeForBdryEdgeNode.clear();
        size_t edgesPerBoundarySimplex = Simplex::numEdges(_K - 1);
        m_BN.resize(BaseMesh::numBoundarySimplices() * edgesPerBoundarySimplex);
        for (size_t si = 0; si < BaseMesh::numBoundarySimplices(); ++si) {
            auto s = BaseMesh::boundarySimplex(si);
            for (size_t ei = 0; ei < edgesPerBoundarySimplex; ++ei) {
                UnorderedPair edge(s.vertex(Simplex::edgeStartNode(ei)).index(),
                                   s.vertex(  Simplex::edgeEndNode(ei)).index());
                auto it = edgeNodes.find(edge);
                size_t nodeIdx;
                if (it == edgeNodes.end()) {
                    edgeNodes[edge] = nodeIdx = m_edgeForBdryEdgeNode.size();
                    m_edgeForBdryEdgeNode.push_back(edge);
                }
                else {
                    nodeIdx = it->second;
                    edgeNodes.erase(it);
                }
                m_BN[si * edgesPerBoundarySimplex + ei] = nodeIdx;
            }
        }
    }

    // Allocate data arrays unless the special TMEmptyData type is passed
    if (typeid(        NodeData) != typeid(TMEmptyData))         m_nodeData.resize(numNodes());
    if (typeid(BoundaryNodeData) != typeid(TMEmptyData)) m_boundaryNodeData.resize(numBoundaryNodes());

    setNodePositions(vertices);
}
