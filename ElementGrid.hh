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
#include "Quadrature.hh"
#include "Grid.hh"
#include <Eigen/Dense>
#include <vector>

template<typename Model>
class ElementGrid2D : public Grid2D {
public:
    typedef Eigen::Vector4i AdjacencyVec;

    ElementGrid2D(size_t Nx, size_t Ny, double cellOverlapThreshold,
                  const Quadrature2D &q, const Model &model)
        : Grid2D(Nx, Ny, model.boundingBox()),
          m_cellOverlapThreshold(cellOverlapThreshold), m_quadrature(q),
          m_model(model)
    {
        update();
    }

    // Should be called whenever the grid size, quadrature, or model changes
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

    void setCellOverlapThreshold(double eps) {
        m_cellOverlapThreshold = eps;
        update();
    }

    double getCellOverlapThreshold() const {
        return m_cellOverlapThreshold;
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
        return cellBoundingBox(m_cellForElement[i]);
    }

    Vector nodePosition(size_t i) const
    {
        assert(i < m_vertexForNode.size());
        return vertexPosition(m_vertexForNode[i]);
    }

    void elementCorners(size_t ei, AdjacencyVec &corners) const {
        assert(ei < m_elementForCell.size());
        cellVertices(m_cellForElement[ei], corners);
        for (size_t i = 0; i < (size_t) corners.rows(); ++i) {
            corners[i] = m_nodeForVertex[corners[i]];
            assert((corners[i] >= 0) && ((size_t) corners[i] < numNodes()));
        }
    }

    AdjacencyVec elementCorners(size_t ei) const {
        AdjacencyVec result;
        elementCorners(ei, result);
        return result;
    }

    // Query the grid to count how many neighboring nodes a node has.
    // Useful for NNZ computation.
    size_t numNodesAdjacentNode(size_t ni) const {
        assert(ni < m_vertexForNode.size());
        size_t row, col;
        get2DVertexIndex(ni, row, col);
        size_t adjacencyCount;
        for (size_t r  = row - 1; r <= row + 1; ++r) {
            if (r > m_Ny) continue;
            for (size_t c  = col  - 1; c <= col + 1; ++c) {
                if (c > m_Nx) continue;
                size_t v = get1DVertexIndex(r, c);
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

    bool elementIsFull(size_t i) const {
        assert(i < m_isFullElement.size());
        return m_isFullElement[i];
    }

    ~ElementGrid2D();

private:
    ////////////////////////////////////////////////////////////////////////////
    // Member Variables
    ////////////////////////////////////////////////////////////////////////////
    typedef std::vector<int> IndexVector;

    // How much a cell must overlap the object to be considered an element.
    Scalar m_cellOverlapThreshold;

    // Maps between node/vertex indices and element/cell indices
    IndexVector m_nodeForVertex, m_vertexForNode,
                m_elementForCell, m_cellForElement;
    std::vector<bool> m_isFullElement;

    const Quadrature2D &m_quadrature;
    const Model &m_model;

    ////////////////////////////////////////////////////////////////////////////
    // Private Member Functions
    ////////////////////////////////////////////////////////////////////////////
};

#include "ElementGrid.inl"

#endif // ELEMENT_GRID_HH
