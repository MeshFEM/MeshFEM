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
void ElementGrid2D<Model>::update(std::vector<Scalar> cellOverlaps)
{
    m_updatePending = true;

    if (!updatesEnabled())
        return;

    m_updatePending = false;

    if (!m_boundingBoxLocked)
        setBoundingBox(m_model.boundingBox());
    
    if (cellOverlaps.size() != numCells()) {
        cellOverlaps.resize(numCells());

        std::vector<Vector> qPoints;
        for (size_t r = 0; r < rows(); ++r) {
            for (size_t c = 0; c < cols(); ++c) {
                BBox_t b = cellBoundingBox(r, c);
                m_quadrature.quadraturePoints(b, qPoints);
                size_t insideCount = 0;
                for (size_t pi = 0; pi < qPoints.size(); ++pi) {
                    if (m_model.isInside(qPoints[pi]))
                        ++insideCount;
                }
                size_t cell = get1DCellIndex(r, c);
                cellOverlaps[cell] = ((Scalar) insideCount) / qPoints.size();
            }
        }
    }

    m_elementForCell.assign(numCells(), -1);
    m_elementOverlap.clear();
    size_t numElements = 0;
    for (size_t r = 0; r < rows(); ++r) {
        for (size_t c = 0; c < cols(); ++c) {
            // A cell is an element if all quadrature points fall inside the
            // object or if the fraction exceeds the cell overlap threshold.
            size_t cell = get1DCellIndex(r, c);
            Scalar overlap  = cellOverlaps[cell];
            if ((overlap == 1.0) || (overlap > m_cellOverlapThreshold)) {
                m_elementForCell[cell] = numElements++;
                m_elementOverlap.push_back(overlap);
            }
        }
    }

    // Invert m_elementForCell to get m_cellForElement, mark nodes
    m_cellForElement.resize(numElements);
    AdjacencyVec cellVerts;
    const size_t numVerts = numVertices();
    std::vector<bool> isNode(numVerts);
    for (size_t cell = 0; cell < m_elementForCell.size(); ++cell) {
        int e = m_elementForCell[cell];
        if (e >= 0) {
            assert((size_t) e < numElements);
            m_cellForElement[e] = cell;
            cellVertices(cell, cellVerts);
            // All corners of an element cell are nodes.
            for (size_t i = 0; i < (size_t) cellVerts.rows(); ++i) {
                assert((size_t) cellVerts[i] < numVerts);
                isNode[cellVerts[i]] = true;
            }
        }
    }

    // Compute m_vertexForNode, m_nodeForVertex
    m_nodeForVertex.assign(numVerts, -1);
    size_t numNodes = 0;
    for (size_t v = 0; v < numVerts; ++v) {
        if (isNode[v])
            m_nodeForVertex[v] = numNodes++;
    }
    m_vertexForNode.resize(numNodes);
    for (size_t v = 0; v < numVerts; ++v) {
        int n = m_nodeForVertex[v];
        if (n >= 0) {
            assert((size_t) n < numNodes);
            m_vertexForNode[n] = v;
        }
    }
}
