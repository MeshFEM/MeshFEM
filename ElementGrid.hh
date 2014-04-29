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
    using Grid2D::_BBox;
    using Grid2D::AdjacencyVec;

    ElementGrid2D(size_t Nx, size_t Ny, double cellOverlapThreshold,
                  const Quadrature2D &q, const Model &model, size_t borderWidth)
        : Grid2D(Nx, Ny, model.boundingBox(), borderWidth),
          m_cellOverlapThreshold(cellOverlapThreshold), m_boundingBoxLocked(false),
          m_updatesEnabled(true), m_updatePending(false),
          m_quadrature(q), m_model(model)
    {
        update();
    }

    // Construct ElementGrid2D quickly using (model, bbox, cellOverlaps) to
    // accelerate
    ElementGrid2D(size_t Nx, size_t Ny, double cellOverlapThreshold,
                  const Quadrature2D &q, const Model &model, const _BBox &bbox,
                  size_t borderWidth,
                  std::vector<Scalar> cellOverlaps = std::vector<Scalar>())
        : Grid2D(Nx, Ny, bbox, borderWidth),
          m_cellOverlapThreshold(cellOverlapThreshold), m_boundingBoxLocked(false),
          m_updatesEnabled(true), m_updatePending(false),
          m_quadrature(q), m_model(model)
    {
        update(cellOverlaps);
    }

    bool boundingBoxIsLocked() const { return m_boundingBoxLocked; }
    void setBoundingBoxLocked(bool locked) { m_boundingBoxLocked = locked; }

    // Allow masking of update() calls. This allows us to avoid wasting time
    // when multiple sequential changes are made and we don't want to update the
    // grid in the intermediate states. In other words, a sequence of updates
    // can be coalesced into a single update() call.
    bool updatesEnabled() const { return m_updatesEnabled; }
    void setUpdatesEnabled(bool enabled) { m_updatesEnabled = enabled; }
    // Check whether a call to update() was masked. This is useful for seeing if
    // there were actually updates coalesced as described above.
    bool updatePending() const { return m_updatePending; }

    // Should be called whenever the grid size, quadrature, or model changes. If
    // cellOverlaps is passed and is of size numCells(), it is used to determine
    // cell classification without running inside/outside queries (much faster!)
    void update(std::vector<Scalar> cellOverlaps = std::vector<Scalar>());

    void setBorderWidth(size_t borderWidth) {
        Grid2D::setBorderWidth(borderWidth);
        update();
    }

    void setGridSize(size_t Nx, size_t Ny) {
        Grid2D::setGridSize(Nx, Ny);
        update();
    }

    void setCellOverlapThreshold(double eps) {
        m_cellOverlapThreshold = eps;
        update();
    }

    double getCellOverlapThreshold() const {
        return m_cellOverlapThreshold;
    }

    size_t numElements() const { return m_cellForElement.size(); }
    size_t numNodes()    const { return m_vertexForNode.size(); }

    _BBox elementBoundingBox(size_t i) const
    {
        assert(i < m_cellForElement.size());
        return cellBoundingBox(m_cellForElement[i]);
    }

    Scalar elementOverlap(size_t i) const
    {
        assert(i < m_elementOverlap.size());
        return m_elementOverlap[i];
    }

    void getCellOverlaps(std::vector<Scalar> &cellOverlaps) const {
        cellOverlaps.assign(numCells(), 0.0);
        for (size_t i = 0; i < numElements(); ++i)
            cellOverlaps[m_cellForElement[i]] = elementOverlap(i);
    }

    Vector2D nodePosition(size_t i) const
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
        get2DVertexIndex(m_vertexForNode[ni], row, col);
        size_t adjacencyCount;
        for (size_t r  = row - 1; r <= row + 1; ++r) {
            if (r > rows()) continue;
            for (size_t c  = col  - 1; c <= col + 1; ++c) {
                if (c > cols()) continue;
                size_t v = get1DVertexIndex(r, c);
                if (m_cellForElement[v] >= 0)
                    ++adjacencyCount;
            }
        }

        // Don't count the node itself.
        return adjacencyCount - 1;
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the elements above, right, below, and left of this element.
    //  @param[in]  ei      element to query
    //  @param[out] adj     vector of adjacent element indices
    *///////////////////////////////////////////////////////////////////////////
    void elementsAdjacentElement(size_t ei, std::vector<size_t> &adj) const {
        assert(ei < numElements());
        size_t row, col;
        get2DCellIndex(m_cellForElement[ei], row, col);
        adj.clear();
        adj.reserve(4);
        if (row + 1 < rows()) {
            int ej = m_elementForCell[get1DCellIndex(row + 1, col)];
            if (ej >= 0) adj.push_back(ej);
        }
        if (col + 1 < cols()) {
            int ej = m_elementForCell[get1DCellIndex(row, col + 1)];
            if (ej >= 0) adj.push_back(ej);
        }
        if (row > 0) {
            int ej = m_elementForCell[get1DCellIndex(row - 1, col)];
            if (ej >= 0) adj.push_back(ej);
        }
        if (col > 0) {
            int ej = m_elementForCell[get1DCellIndex(row, col - 1)];
            if (ej >= 0) adj.push_back(ej);
        }
    }

    AdjacencyVec elementsAdjacentNode(size_t ni) const {
        // TODO implement this
        assert(ni < numNodes());
        assert(false);
        return AdjacencyVec(AdjacencyVec::Zero());
    }

    void elementsAroundPoint(const Vector2D &pt, Scalar radius,
                          std::vector<size_t> &elements) const {
        elements.clear();

        std::vector<size_t> cells;
        cellsAroundPoint(pt, radius, cells);

        for (size_t i = 0; i < cells.size(); ++i) {
            assert(cells[i] < m_elementForCell.size());
            int elem = m_elementForCell[cells[i]];
            if (elem >= 0)
                elements.push_back(elem);
        }
    }

    bool elementIsFull(size_t i) const {
        assert(i < m_elementOverlap.size());
        return m_elementOverlap[i] == 1.0;
    }

    ~ElementGrid2D() { };

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
    std::vector<Scalar> m_elementOverlap;

    bool m_boundingBoxLocked;
    bool m_updatesEnabled, m_updatePending;

    const Quadrature2D &m_quadrature;
    const Model &m_model;
};

template<typename Model>
class ElementGrid3D : public Grid3D {
public:
    using Grid3D::_BBox;
    using Grid3D::AdjacencyVec;

    ElementGrid3D(size_t Nx, size_t Ny, size_t Nz, double cellOverlapThreshold,
                  const Quadrature3D &q, const Model &model, size_t borderWidth)
        : Grid3D(Nx, Ny, Nz, model.boundingBox(), borderWidth),
          m_cellOverlapThreshold(cellOverlapThreshold),
          m_quadrature(q), m_model(model)
    {
        update();
    }

    // Construct ElementGrid3D quickly using (model, bbox, cellOverlaps) to
    // accelerate
    ElementGrid3D(size_t Nx, size_t Ny, size_t Nz, double cellOverlapThreshold,
                  const Quadrature3D &q, const Model &model, const _BBox &bbox,
                  size_t borderWidth,
                  std::vector<Scalar> cellOverlaps = std::vector<Scalar>())
        : Grid3D(Nx, Ny, Nz, bbox, borderWidth),
          m_cellOverlapThreshold(cellOverlapThreshold),
          m_quadrature(q), m_model(model)
    {
        update(cellOverlaps);
    }

    // Should be called whenever the grid size, quadrature, or model changes. If
    // cellOverlaps is passed and is of size numCells(), it is used to determine
    // cell classification without running inside/outside queries (much faster!)
    void update(std::vector<Scalar> cellOverlaps = std::vector<Scalar>());

    void setBorderWidth(size_t borderWidth) {
        Grid3D::setBorderWidth(borderWidth);
        update();
    }

    void setGridSize(size_t Nx, size_t Ny, size_t Nz) {
        Grid3D::setGridSize(Nx, Ny, Nz);
        update();
    }

    void setCellOverlapThreshold(double eps) {
        m_cellOverlapThreshold = eps;
        update();
    }

    double getCellOverlapThreshold() const { return m_cellOverlapThreshold; }

    size_t numElements() const { return m_cellForElement.size(); }
    size_t numNodes()    const { return m_vertexForNode.size(); }

    _BBox elementBoundingBox(size_t i) const
    {
        assert(i < m_cellForElement.size());
        return cellBoundingBox(m_cellForElement[i]);
    }

    Scalar elementOverlap(size_t i) const
    {
        assert(i < m_elementOverlap.size());
        return m_elementOverlap[i];
    }

    void getCellOverlaps(std::vector<Scalar> &cellOverlaps) const {
        cellOverlaps.assign(numCells(), 0.0);
        for (size_t i = 0; i < numElements(); ++i)
            cellOverlaps[m_cellForElement[i]] = elementOverlap(i);
    }

    Vector3D nodePosition(size_t i) const
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

    void elementsAroundPoint(const Vector3D &pt, Scalar radius,
                          std::vector<size_t> &elements) const {
        elements.clear();

        std::vector<size_t> cells;
        cellsAroundPoint(pt, radius, cells);

        for (size_t i = 0; i < cells.size(); ++i) {
            assert(cells[i] < m_elementForCell.size());
            int elem = m_elementForCell[cells[i]];
            if (elem >= 0)
                elements.push_back(elem);
        }
    }

    bool elementIsFull(size_t i) const {
        assert(i < m_elementOverlap.size());
        return m_elementOverlap[i] == 1.0;
    }

    ~ElementGrid3D() { };

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
    std::vector<Scalar> m_elementOverlap;

    const Quadrature3D &m_quadrature;
    const Model &m_model;
};


#include "ElementGrid.inl"

#endif // ELEMENT_GRID_HH
