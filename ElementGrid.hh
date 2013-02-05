////////////////////////////////////////////////////////////////////////////////
// ElementGrid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a regular grid of rectangular, axis-aligned elements
//      surrounding some implicitly-defined geometry. Only
//      grid cells overlapping the geometry are considered elements.
//      Assumes that piecewise bi/tri-linear basis functions will be used.
//
//      Terminology:
//          Cell:    One of Nx * Ny rectangular boxes making up the grid.
//                   The cell at (r, c) has index (r * Nx + c)
//          Vertex:  One of the (Nx + 1) * (Ny + 1) distinct cell corners
//          Element: A cell overlapping the geometry (thus in the support of the
//                   functions to be integrated). All four element corners are
//                   nodes.
//          Node:    A vertex whose basis function's support overlaps the
//                   geometry. These are exactly the vertices at the corners
//                   of some element.
//
//          Full Element: A cell completely contained within the geometry.
//          Boundary Element: A cell only partially contained within the
//                            geometry.
//
//      Note: We determine if a cell is an element by checking every quadrature
//      point. This will be less accurate when fewer quadrature points are used.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/03/2013 12:21:08
////////////////////////////////////////////////////////////////////////////////
#ifndef ELEMENT_GRID_HH
#define ELEMENT_GRID_HH

#include "GlobalTypes.hh"
#include <Eigen/Dense>
#include <vector>

template<typename Model>
class ElementGrid2D {
public:
    typedef Eigen::Vector4i AdjacencyVec;

    ElementGrid2D(size_t Nx, size_t Ny, const Quadrature2D &q,
                  const Model &model)
        : m_Nx(Nx), m_Ny(Ny), m_quadrature(q), m_model(model)
    {
        update();
    }

    // Should be called whenever the quadrature or model changes
    void update();

    void setGridSize(size_t Nx, size_t Ny) {
        m_Nx = Nx;
        m_Ny = Ny;
        update();
    }

    void getGridSize(size_t &Nx, size_t &Ny) const {
        Nx = m_Nx;
        Ny = m_Ny;
    }

    size_t numElements() const  {
        return m_cellForElement.size();
    }

    size_t numNodes() const {
        return m_vertexForNode.size();
    }

    BBox_t elementBoundingBox(size_t i) const
    {
        assert(i < m_cellForElement.size());
        return m_cellBoundingBox(m_cellForElement[i]);
    }

    Vector nodePosition(size_t i) const
    {
        assert(i < m_vertexForNode.size());
        return m_vertexPosition(m_vertexForNode[i]);
    }

    AdjacencyVec elementCorners(size_t ei) const {
        assert(ei < m_elementForCell.size());
        AdjacencyVec result;
        m_cellVertices(m_cellForElement[ei], result);
        for (size_t i = 0; i < (size_t) result.rows(); ++i) {
            result[i] = m_nodeForVertex[result[i]];
            assert((result[i] >= 0) && ((size_t) result[i] < numNodes()));
        }
        return result;
    }

    // Query the grid to count how many neighboring nodes a node has.
    // Useful for NNZ computation.
    size_t numNodesAdjacentNode(size_t ni) const {
        assert(ni < m_vertexForNode.size());
        size_t row, col;
        m_get2DVertexIndex(ni, row, col);
        size_t adjacencyCount;
        for (size_t r  = row - 1; r <= row + 1; ++r) {
            if (r > m_Ny) continue;
            for (size_t c  = col  - 1; c <= col + 1; ++c) {
                if (c > m_Nx) continue;
                size_t v = m_get1DVertexIndex(r, c);
                if (m_cellForElement[v] >= 0)
                    ++adjacencyCount;
            }
        }

        // Don't count the node itself.
        return adjacencyCount - 1;
    }

    AdjacencyVec elementsAdjacentNode(size_t ni) const {
        // TODO implement this
        assert(ni < numNodes());
        assert(false);
        return AdjacencyVec(AdjacencyVec::Zero());
    }

    bool elementIsFull(size_t i) const
    {
        assert(i < m_isFullElement.size());
        return m_isFullElement[i];
    }

    ~ElementGrid2D();

private:
    ////////////////////////////////////////////////////////////////////////////
    // Member Variables
    ////////////////////////////////////////////////////////////////////////////
    typedef std::vector<int> IndexVector;
    size_t m_Nx, m_Ny;
    BBox_t m_bbox;

    // Maps between node/vertex indices and element/cell indices
    IndexVector m_nodeForVertex, m_vertexForNode,
                m_elementForCell, m_cellForElement;
    std::vector<bool> m_isFullElement;

    const Quadrature2D &m_quadrature;
    const Model &m_model;

    ////////////////////////////////////////////////////////////////////////////
    // Private Member Functions
    ////////////////////////////////////////////////////////////////////////////
    void m_get2DCellIndex(size_t i, size_t &row, size_t &col) const {
        assert(i < m_numCells());
        row = i / m_Nx;
        col = i % m_Nx;
    }

    size_t m_get1DCellIndex(size_t row, size_t col) const {
        assert((row < m_Ny) && (col < m_Nx));
        return row * m_Nx + col;
    }

    void m_get2DVertexIndex(size_t i, size_t &row, size_t &col) const {
        assert(i < m_numVertices());
        row = i / (m_Nx + 1);
        col = i % (m_Nx + 1);
    }

    size_t m_get1DVertexIndex(size_t row, size_t col) const {
        assert((row < m_Ny + 1) && (col < m_Nx + 1));
        return row * (m_Nx + 1) + col;
    }

    Vector m_vertexPosition(size_t i) const {
        size_t row, col;
        m_get2DVertexIndex(i, row, col);
        return m_vertexPosition(row, col);
    }

    Vector m_vertexPosition(size_t row, size_t col) const {
        assert((row < m_Ny + 1) && (col < m_Nx + 1));
        return m_bbox.interpolatePoint(Vector((1.0 * col) / m_Nx,
                                              (1.0 * row) / m_Ny));
    }

    BBox_t m_cellBoundingBox(size_t i) const {
        size_t row, col;
        m_get2DCellIndex(i, row, col);
        return m_cellBoundingBox(row, col);
    }

    BBox_t m_cellBoundingBox(size_t row, size_t col) const {
        assert((row < m_Ny) && (col < m_Nx));
        return BBox_t(m_vertexPosition(row, col),
                      m_vertexPosition(row + 1, col + 1));
    }
    
    void m_cellVertices(size_t i, AdjacencyVec &adj) const {
        size_t row, col;
        m_get2DCellIndex(i, row, col);
        adj[0] = m_get1DVertexIndex(row    , col    );
        adj[1] = m_get1DVertexIndex(row    , col + 1);
        adj[2] = m_get1DVertexIndex(row + 1, col + 1);
        adj[3] = m_get1DVertexIndex(row + 1, col    );
    }

    size_t m_numVertices() const { return (m_Nx + 1) * (m_Ny + 1); }
    size_t m_numCells()    const { return m_Nx * m_Ny; }
};

#include "ElementGrid.inl"

#endif // ELEMENT_GRID_HH
