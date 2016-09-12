#include "Simplex.hh"

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
                    // because many elements share the same edge.
                    // Also, we need to use edgeNodes to efficiently link the
                    // boundary/volume edges together below.
                }
                m_N[si * edgesPerSimplex + ei] = nodeIdx;
            }
        }

        std::map<UnorderedPair, int> boundaryEdgeNodes;
        // Construct a boundary edge node for each surface edge.
        // Again, we could optimize this by using BaseMesh's traversal
        // operations.
        // Also, for triangle meshes, there is a unique edge node per
        // element (none is shared), so we don't need any gluing.
        m_edgeForBdryEdgeNode.clear();
        size_t edgesPerBoundarySimplex = Simplex::numEdges(_K - 1);
        m_BN.resize(BaseMesh::numBoundarySimplices() * edgesPerBoundarySimplex);
        for (size_t si = 0; si < BaseMesh::numBoundarySimplices(); ++si) {
            auto s = BaseMesh::boundarySimplex(si);
            for (size_t ei = 0; ei < edgesPerBoundarySimplex; ++ei) {
                UnorderedPair edge(s.vertex(Simplex::edgeStartNode(ei)).index(),
                                   s.vertex(  Simplex::edgeEndNode(ei)).index());
                auto it = boundaryEdgeNodes.find(edge);
                size_t nodeIdx;
                if (it == boundaryEdgeNodes.end()) {
                    boundaryEdgeNodes[edge] = nodeIdx = m_edgeForBdryEdgeNode.size();
                    m_edgeForBdryEdgeNode.push_back(edge);
                }
                else {
                    nodeIdx = it->second;
                    boundaryEdgeNodes.erase(it);
                }
                m_BN[si * edgesPerBoundarySimplex + ei] = nodeIdx;
            }
        }

        // Link the boundary and volume edges
        // (Allows traversal between boundary nodes and collocated volume nodes
        //  on the edges)
        m_bdryEdgeForVolEdge.assign(numEdgeNodes(), -1);
        m_volEdgeForBdryEdge.assign(numBoundaryEdgeNodes(), -1);
        for (size_t bni = 0; bni < numBoundaryEdgeNodes(); ++bni) {
            auto be = m_edgeForBdryEdgeNode[bni];
            UnorderedPair vbe(BaseMesh::m_vertexForBdryVertex(be[0]),
                              BaseMesh::m_vertexForBdryVertex(be[1]));
            m_volEdgeForBdryEdge[bni] = edgeNodes.at(vbe);
            assert(m_bdryEdgeForVolEdge.at(m_volEdgeForBdryEdge[bni]) == -1);
            m_bdryEdgeForVolEdge[m_volEdgeForBdryEdge[bni]] = bni;
        }
    }

    // Allocate data arrays unless the special TMEmptyData type is passed
    m_nodeData        .resize(numNodes());
    m_boundaryNodeData.resize(numBoundaryNodes());

    setNodePositions(vertices);
}
