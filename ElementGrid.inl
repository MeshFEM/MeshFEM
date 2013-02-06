////////////////////////////////////////////////////////////////////////////////
// ElementGrid.inl
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a regular grid of rectangular, axis-aligned elements
//      surrounding some implicitly-defined geometry.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//
//  Created:  02/03/2013 13:41:22
//  Revision History:
//      02/03/2013  Julian Panetta    Initial Revision
////////////////////////////////////////////////////////////////////////////////

// Should be called whenever the quadrature or model changes
template<typename Model>
void ElementGrid2D<Model>::update()
{
    m_bbox = m_model.boundingBox();

    m_elementForCell.assign(m_numCells(), -1);
    std::vector<bool> isFullElement(m_numCells(), false);
    size_t numElements = 0;
    for (size_t r = 0; r < m_Ny; ++r) {
        for (size_t c = 0; c < m_Nx; ++c) {
            size_t cell = m_get1DCellIndex(r, c);
            BBox_t b = m_cellBoundingBox(r, c);
            std::vector<Vector> quadraturePoints =
                                m_quadrature.quadraturePoints(b);
            size_t insideCount = 0;
            for (size_t pi = 0; pi < quadraturePoints.size(); ++pi) {
                if (m_model.isInside(quadraturePoints[pi]))
                    ++insideCount;
            }
            if (insideCount > 0) {
                isFullElement[cell] = (insideCount == quadraturePoints.size());
                m_elementForCell[cell] = numElements++;
            }
        }
    }

    m_cellForElement.resize(numElements);
    m_isFullElement.resize(numElements);

    const size_t numVertices = m_numVertices();
    std::vector<bool> isNode(numVertices);

    // Invert m_elementForCell, mark nodes and full elements
    AdjacencyVec cellVertices;
    for (size_t cell = 0; cell < m_elementForCell.size(); ++cell) {
        int e = m_elementForCell[cell];
        if (e >= 0) {
            assert((size_t) e < numElements);
            m_cellForElement[e] = cell;
            m_isFullElement[e] = isFullElement[cell];
            m_cellVertices(cell, cellVertices);
            // All corners of an element cell are nodes.
            for (size_t i = 0; i < (size_t) cellVertices.rows(); ++i) {
                assert((size_t) cellVertices[i] < numVertices);
                isNode[cellVertices[i]] = true;
            }
        }
    }

    // Compute m_vertexForNode, m_nodeForVertex
    m_nodeForVertex.assign(numVertices, -1);
    size_t numNodes = 0;
    for (size_t v = 0; v < numVertices; ++v) {
        if (isNode[v])
            m_nodeForVertex[v] = numNodes++;
    }
    m_vertexForNode.resize(numNodes);
    for (size_t v = 0; v < numVertices; ++v) {
        int n = m_nodeForVertex[v];
        if (n >= 0) {
            assert((size_t) n < numNodes);
            m_vertexForNode[n] = v;
        }
    }
}
