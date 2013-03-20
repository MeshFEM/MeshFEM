////////////////////////////////////////////////////////////////////////////////
// Grid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a regular grid of axis-aligned cells fit within the
//      axis-aligned bounding box of an object.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/18/2013 14:59:19
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include "GlobalTypes.hh"
#ifndef GRID_HH
#define GRID_HH

class Grid2D {
public:
    typedef Eigen::Vector4i AdjacencyVec;

    Grid2D(size_t Nx, size_t Ny, const BBox_t &bbox)
        : m_Nx(Nx), m_Ny(Ny), m_bbox(bbox) { }

    void setBoundingBox(BBox_t bbox) {
        m_bbox = bbox;
    }

    void setGridSize(size_t Nx, size_t Ny) {
        m_Nx = Nx;
        m_Ny = Ny;
    }

    Vector cellSize() const {
        Vector sizes = m_bbox.dimensions();
        sizes[0] /= m_Nx, sizes[1] /= m_Ny;
        return sizes;
    }

    void get2DCellIndex(size_t i, size_t &row, size_t &col) const {
        assert(i < numCells());
        row = i / m_Nx;
        col = i % m_Nx;
    }

    size_t get1DCellIndex(size_t row, size_t col) const {
        assert((row < m_Ny) && (col < m_Nx));
        return row * m_Nx + col;
    }

    void get2DVertexIndex(size_t i, size_t &row, size_t &col) const {
        assert(i < numVertices());
        row = i / (m_Nx + 1);
        col = i % (m_Nx + 1);
    }

    size_t get1DVertexIndex(size_t row, size_t col) const {
        assert((row < m_Ny + 1) && (col < m_Nx + 1));
        return row * (m_Nx + 1) + col;
    }

    Vector vertexPosition(size_t i) const {
        size_t row, col;
        get2DVertexIndex(i, row, col);
        return vertexPosition(row, col);
    }

    Vector vertexPosition(size_t row, size_t col) const {
        assert((row < m_Ny + 1) && (col < m_Nx + 1));
        return m_bbox.interpolatePoint(Vector((1.0 * col) / m_Nx,
                                              (1.0 * row) / m_Ny));
    }

    Vector cellMidpointPosition(size_t i) const {
        size_t row, col;
        get2DCellIndex(i, row, col);
        return cellMidpointPosition(row, col);
    }

    Vector cellMidpointPosition(size_t row, size_t col) const {
        return cellBoundingBox(row, col).interpolatePoint(Vector(.5, .5));
    }

    BBox_t cellBoundingBox(size_t i) const {
        size_t row, col;
        get2DCellIndex(i, row, col);
        return cellBoundingBox(row, col);
    }

    BBox_t cellBoundingBox(size_t row, size_t col) const {
        assert((row < m_Ny) && (col < m_Nx));
        return BBox_t(vertexPosition(row, col),
                      vertexPosition(row + 1, col + 1));
    }
    
    void cellVertices(size_t i, AdjacencyVec &adj) const {
        size_t row, col;
        get2DCellIndex(i, row, col);
        adj[0] = get1DVertexIndex(row    , col    );
        adj[1] = get1DVertexIndex(row    , col + 1);
        adj[2] = get1DVertexIndex(row + 1, col + 1);
        adj[3] = get1DVertexIndex(row + 1, col    );
    }

    size_t numVertices() const { return (m_Nx + 1) * (m_Ny + 1); }
    size_t numCells()    const { return m_Nx * m_Ny; }

    ~Grid2D() { }

protected:
    size_t m_Nx, m_Ny;
    BBox_t m_bbox;
};

#endif // GRID_HH
