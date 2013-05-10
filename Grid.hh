////////////////////////////////////////////////////////////////////////////////
// Grid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a regular grid of Ny x Nx axis-aligned cells fit within
//      the axis-aligned bounding box of an object.
//
//      This grid has a border m_borderWidth cells wide filled with additional
//      cells of the same size. This border is useful during marching squares
//      and shape optimization, where we need to be able to compute quantities
//      outside the original object's bounding box.
//
//      These border cells are treated the same as any other, so we effectively
//      have a grid of size (Ny + 2 borderWidth) x (Nx + 2 borderWidth) such
//      that the inner Ny x Nx block fits tightly within the object's bounding
//      box.
//
//      Cells are numbered 0..(Nx + 2 borderWidth)*(Ny + 2 borderWidth) - 1
//      Verts are numbered 0..(Nx + 2borderWidth + 1)*(Ny + 2borderWidth + 1)-1
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/18/2013 14:59:19
////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <algorithm>
#include "GlobalTypes.hh"
#include "Geometry.hh"
#ifndef GRID_HH
#define GRID_HH

class Grid2D {
public:
    typedef Eigen::Vector4i AdjacencyVec;

    Grid2D(size_t Nx, size_t Ny, const BBox_t &bbox, size_t borderWidth = 0)
        : m_Nx(Nx), m_Ny(Ny), m_borderWidth(borderWidth), m_bbox(bbox) { }

    void setBoundingBox(BBox_t bbox) {
        m_bbox = bbox;
    }

    size_t interiorCols() const { return m_Nx; }
    size_t interiorRows() const { return m_Ny; }

    size_t cols() const {
        return m_Nx + 2 * m_borderWidth;
    }

    size_t rows() const {
        return m_Ny + 2 * m_borderWidth;
    }

    size_t vertexCols() const {
        return cols() + 1;
    }

    size_t vertexRows() const {
        return rows() + 1;
    }

    void setGridSize(size_t Nx, size_t Ny) {
        m_Nx = Nx;
        m_Ny = Ny;
    }

    void getGridSize(size_t &Nx, size_t &Ny) const {
        Nx = m_Nx;
        Ny = m_Ny;
    }

    // Add borderWidth cells around this grid, maintaining cell size.
    void setBorderWidth(size_t borderWidth) {
        m_borderWidth = borderWidth;
    }

    size_t getBorderWidth() const {
        return m_borderWidth;
    }

    Vector cellSize() const {
        // Note: cell size computed ignoring border
        Vector sizes = m_bbox.dimensions();
        sizes[0] /= m_Nx, sizes[1] /= m_Ny;
        return sizes;
    }

    void get2DCellIndex(size_t i, size_t &row, size_t &col) const {
        assert(i < numCells());
        row = i / cols();
        col = i % cols();
    }

    size_t get1DCellIndex(size_t row, size_t col) const {
        assert((row < rows()) && (col < cols()));
        return row * cols() + col;
    }

    void get2DVertexIndex(size_t i, size_t &row, size_t &col) const {
        assert(i < numVertices());
        row = i / vertexCols();
        col = i % vertexCols();
    }

    size_t get1DVertexIndex(size_t row, size_t col) const {
        assert((row < vertexRows()) && (col < vertexCols()));
        return row * vertexCols() + col;
    }

    Vector vertexPosition(size_t i) const {
        size_t row, col;
        get2DVertexIndex(i, row, col);
        return vertexPosition(row, col);
    }

    Vector vertexPosition(size_t row, size_t col) const {
        assert((row < vertexRows()) && (col < vertexCols()));
        Vector icoord;
        icoord[0] = (((double) col) - ((double) m_borderWidth)) / m_Nx;
        icoord[1] = (((double) row) - ((double) m_borderWidth)) / m_Ny;
        return m_bbox.interpolatePoint(icoord);
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
        assert((row < rows()) && (col < cols()));
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

    void cellsAroundPoint(const Vector &pt, Scalar radius,
                          std::vector<size_t> &cells) const {
        cells.clear();

        // Get the minimum and maximum corner in interpolation coordinates
        Vector icoord = m_bbox.interpolationCoordinates(pt);
        Vector iradius = radius / m_bbox.dimensions().array();
        Vector minCorner = icoord - iradius;
        Vector maxCorner = icoord + iradius;

        // Shift so the interpolation coordinates corresponding to cells are in
        // [0.0, cols() / m_Nx], [0.0, rows() / Ny]
        Vector iborderWidth(((double) m_borderWidth) / m_Nx,
                            ((double) m_borderWidth) / m_Ny);
        minCorner += iborderWidth;
        maxCorner += iborderWidth;

        size_t gridStartX, gridStartY, gridEndX, gridEndY;
        gridStartX = std::max((long) floor(m_Nx * minCorner[0]), (long) 0);
        gridStartY = std::max((long) floor(m_Ny * minCorner[1]), (long) 0);
        gridEndX   = std::min((long)  ceil(m_Nx * maxCorner[0]), (long) cols());
        gridEndY   = std::min((long)  ceil(m_Ny * maxCorner[1]), (long) rows());

        for (size_t row = gridStartY; row < gridEndY; ++row) {
            for (size_t col = gridStartX; col < gridEndX; ++col) {
                BBox_t candidate = cellBoundingBox(row, col);
                if (candidate.intersectsCircle(pt, radius)) {
                    cells.push_back(get1DCellIndex(row, col));
                }
            }
        }
    }

    size_t numVertices() const { return vertexCols() * vertexRows(); }
    size_t numCells()    const { return rows() * cols(); }

    ~Grid2D() { }

private:
    size_t m_Nx, m_Ny;
protected:
    size_t m_borderWidth;
    BBox_t m_bbox;
};

#endif // GRID_HH
