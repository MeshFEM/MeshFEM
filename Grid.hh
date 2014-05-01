////////////////////////////////////////////////////////////////////////////////
// Grid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a regular grid of Ny * Nx or Nx * Ny * Nz axis-aligned cells
//      fit within the axis-aligned bounding box of an object.
//
//      This grid has a border m_borderWidth cells wide filled with additional
//      cells of the same size. This border is useful during marching squares
//      and shape optimization, where we need to be able to compute quantities
//      outside the original object's bounding box.
//
//      These border cells are treated the same as any other, so we effectively
//      have a grid of size
//      (Ny + 2 borderWidth) * (Nx + 2 borderWidth) [* (Ny + 2 borderWidth)]
//      such that the inner Ny * Nx [* Nz] block fits tightly within the object's
//      bounding box.
//
//      In 2D:
//      Cells and vertices are arranged into rows (y) and cols (x)
//      Cells are numbered 0..(Nx + 2 borderWidth)*(Ny + 2 borderWidth) - 1
//      Verts are numbered 0..(Nx + 2borderWidth + 1)*(Ny + 2borderWidth + 1)-1
//      Grid indices are column-major: 1D index = row * cols() + col
//
//      In 3D:
//      Cells and vertices are arranged into slices (z), rows (y), and cols (x)
//      Cells are numbered 0 to
//          (Nx + 2 borderWidth)*(Ny + 2 borderWidth)*(Nz + 2 borderWidth) - 1
//      Vertices are numbered 0 to
//          (Nx + 2 borderWidth + 1)*(Ny + 2 borderWidth + 1)*
//          (Nz + 2 borderWidth + 1) - 1
//      Grid indices are row-column major:
//          1D index = slice * rows() * cols() + row * cols() + col
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
    typedef BBox<Vector2D> _BBox;
    typedef Eigen::Vector4i AdjacencyVec;

    Grid2D(size_t Nx, size_t Ny, const _BBox &bbox, size_t borderWidth = 0)
        : m_Nx(Nx), m_Ny(Ny), m_borderWidth(borderWidth), m_bbox(bbox) { }

    void   setBoundingBox(_BBox bbox) { m_bbox = bbox; }
    _BBox getBoundingBox() const      { return m_bbox; }

    size_t interiorCols() const { return m_Nx; }
    size_t interiorRows() const { return m_Ny; }

    size_t rows() const { return m_Ny + 2 * m_borderWidth; }
    size_t cols() const { return m_Nx + 2 * m_borderWidth; }

    size_t vertexRows() const { return rows() + 1; }
    size_t vertexCols() const { return cols() + 1; }

    void setGridSize(size_t Nx, size_t Ny) {
        m_Nx = Nx;
        m_Ny = Ny;
    }

    void getGridSize(size_t &Nx, size_t &Ny) const {
        Nx = m_Nx;
        Ny = m_Ny;
    }

    // Add borderWidth cells around this grid, maintaining cell size.
    void setBorderWidth(size_t borderWidth) { m_borderWidth = borderWidth; }
    size_t getBorderWidth() const { return m_borderWidth; }

    Vector2D cellSize() const {
        // Note: cell size computed ignoring border
        Vector2D sizes = m_bbox.dimensions();
        sizes[0] /= m_Nx, sizes[1] /= m_Ny;
        return sizes;
    }

    Scalar cellVolume() const {
        Vector2D size = cellSize();
        return size[0] * size[1];
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

    Vector2D vertexPosition(size_t i) const {
        size_t row, col;
        get2DVertexIndex(i, row, col);
        return vertexPosition(row, col);
    }

    Vector2D vertexPosition(size_t row, size_t col) const {
        assert((row < vertexRows()) && (col < vertexCols()));
        Vector2D icoord;
        icoord[0] = (((double) col) - ((double) m_borderWidth)) / m_Nx;
        icoord[1] = (((double) row) - ((double) m_borderWidth)) / m_Ny;
        return m_bbox.interpolatePoint(icoord);
    }

    Vector2D cellMidpointPosition(size_t i) const {
        size_t row, col;
        get2DCellIndex(i, row, col);
        return cellMidpointPosition(row, col);
    }

    Vector2D cellMidpointPosition(size_t row, size_t col) const {
        return cellBoundingBox(row, col).interpolatePoint(Vector2D(.5, .5));
    }

    _BBox cellBoundingBox(size_t i) const {
        size_t row, col;
        get2DCellIndex(i, row, col);
        return cellBoundingBox(row, col);
    }

    _BBox cellBoundingBox(size_t row, size_t col) const {
        assert((row < rows()) && (col < cols()));
        return _BBox(vertexPosition(row, col),
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

    void cellsAroundPoint(const Vector2D &pt, Scalar radius,
                          std::vector<size_t> &cells) const {
        cells.clear();

        // Get the minimum and maximum corner in interpolation coordinates
        Vector2D icoord = m_bbox.interpolationCoordinates(pt);
        Vector2D iradius = radius / m_bbox.dimensions().array();
        Vector2D minCorner = icoord - iradius;
        Vector2D maxCorner = icoord + iradius;

        // Shift so the interpolation coordinates corresponding to cells are in
        // [0.0, cols() / Nx], [0.0, rows() / Ny]
        // = [0.0, 1.0 + 2 * borderWidth / Nx], ...
        // (Currently it's [-borderWidth / Nx, 1 + borderWidth / Nx]...)
        Vector2D iborderWidth(((double) m_borderWidth) / m_Nx,
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
                _BBox candidate = cellBoundingBox(row, col);
                if (candidate.intersectsCircle(pt, radius))
                    cells.push_back(get1DCellIndex(row, col));
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
    _BBox m_bbox;
};

class Grid3D {
public:
    typedef BBox<Vector3D> _BBox;
    typedef Eigen::Matrix<int, 8, 1> AdjacencyVec;

    Grid3D(size_t Nx, size_t Ny, size_t Nz, const _BBox &bbox,
           size_t borderWidth = 0)
        : m_Nx(Nx), m_Ny(Ny), m_Nz(Nz), m_borderWidth(borderWidth), m_bbox(bbox)
    { }

    size_t interiorSlices() const { return m_Nz; }
    size_t interiorRows()   const { return m_Ny; }
    size_t interiorCols()   const { return m_Nx; }

    size_t slices() const { return m_Nz + 2 * m_borderWidth; }
    size_t rows()   const { return m_Ny + 2 * m_borderWidth; }
    size_t cols()   const { return m_Nx + 2 * m_borderWidth; }

    size_t vertexSlices() const { return slices() + 1; }
    size_t vertexCols()   const { return cols() + 1; }
    size_t vertexRows()   const { return rows() + 1; }

    void   setBoundingBox(_BBox bbox) { m_bbox = bbox; }
    _BBox getBoundingBox() const      { return m_bbox; }

    void setGridSize(size_t Nx, size_t Ny, size_t Nz) {
        m_Nx = Nx;
        m_Ny = Ny;
        m_Nz = Nz;
    }

    void getGridSize(size_t &Nx, size_t &Ny, size_t &Nz) const {
        Nx = m_Nx;
        Ny = m_Ny;
        Nz = m_Nz;
    }

    // Add borderWidth cells around this grid, maintaining cell size.
    void setBorderWidth(size_t borderWidth) { m_borderWidth = borderWidth; }
    size_t getBorderWidth() const { return m_borderWidth; }

    Vector3D cellSize() const {
        // Note: cell size computed ignoring border
        Vector3D sizes = m_bbox.dimensions();
        sizes[0] /= m_Nx, sizes[1] /= m_Ny; sizes[3] /= m_Nz;
        return sizes;
    }

    Scalar cellVolume() const {
        Vector3D size = cellSize();
        return size[0] * size[1] * size[2];
    }

    void get3DCellIndex(size_t i, size_t &slice, size_t &row, size_t &col) const
    {
        assert(i < numCells());
        slice = i / (rows() * cols());
        i = i % (rows() * cols());
        row = i / cols();
        col = i % cols();
    }

    size_t get1DCellIndex(size_t slice, size_t row, size_t col) const {
        assert((slice < slices()) && (row < rows()) && (col < cols()));
        return slice * rows() * cols() + row * cols() + col;
    }

    void get3DVertexIndex(size_t i, size_t &slice, size_t &row,
                          size_t &col) const {
        assert(i < numVertices());
        row = i / vertexCols();
        col = i % vertexCols();

        slice = i / (vertexRows() * vertexCols());
        i = i % (vertexRows() * vertexCols());
        row = i / vertexCols();
        col = i % vertexCols();
    }

    size_t get1DVertexIndex(size_t slice, size_t row, size_t col) const {
        assert((slice < vertexSlices()) && (row < vertexRows()) &&
               (col < vertexCols()));
        return slice * vertexRows() * vertexCols() + row * vertexCols() + col;
    }

    Vector3D vertexPosition(size_t i) const {
        size_t slice, row, col;
        get3DVertexIndex(i, slice, row, col);
        return vertexPosition(slice, row, col);
    }

    Vector3D vertexPosition(size_t slice, size_t row, size_t col) const {
        assert((slice < vertexSlices()) && (row < vertexRows()) &&
               (col < vertexCols()));
        Vector3D icoord;
        icoord[0] = (((double) col)   - ((double) m_borderWidth)) / m_Nx;
        icoord[1] = (((double) row)   - ((double) m_borderWidth)) / m_Ny;
        icoord[2] = (((double) slice) - ((double) m_borderWidth)) / m_Nz;
        return m_bbox.interpolatePoint(icoord);
    }

    Vector3D cellMidpointPosition(size_t i) const {
        size_t slice, row, col;
        get3DCellIndex(i, slice, row, col);
        return cellMidpointPosition(slice, row, col);
    }

    Vector3D cellMidpointPosition(size_t slice, size_t row, size_t col) const {
        return cellBoundingBox(slice, row, col).
                interpolatePoint(Vector3D(.5, .5, .5));
    }

    _BBox cellBoundingBox(size_t i) const {
        size_t slice, row, col;
        get3DCellIndex(i, slice, row, col);
        return cellBoundingBox(slice, row, col);
    }

    _BBox cellBoundingBox(size_t slice, size_t row, size_t col) const {
        assert((slice < rows()) && (row < rows()) && (col < cols()));
        return _BBox(vertexPosition(slice, row, col),
                      vertexPosition(slice + 1, row + 1, col + 1));
    }
    
    void cellVertices(size_t i, AdjacencyVec &adj) const {
        size_t slice, row, col;
        get3DCellIndex(i, slice, row, col);
        adj[0] = get1DVertexIndex(slice    , row    , col    );
        adj[1] = get1DVertexIndex(slice    , row    , col + 1);
        adj[2] = get1DVertexIndex(slice    , row + 1, col + 1);
        adj[3] = get1DVertexIndex(slice    , row + 1, col    );
        adj[4] = get1DVertexIndex(slice + 1, row    , col    );
        adj[5] = get1DVertexIndex(slice + 1, row    , col + 1);
        adj[6] = get1DVertexIndex(slice + 1, row + 1, col + 1);
        adj[7] = get1DVertexIndex(slice + 1, row + 1, col    );
    }

    void cellsAroundPoint(const Vector3D &pt, Scalar radius,
                          std::vector<size_t> &cells) const {
        cells.clear();

        // Get the minimum and maximum corner in interpolation coordinates
        Vector3D icoord = m_bbox.interpolationCoordinates(pt);
        Vector3D iradius = radius / m_bbox.dimensions().array();
        Vector3D minCorner = icoord - iradius;
        Vector3D maxCorner = icoord + iradius;

        // Shift so the interpolation coordinates corresponding to cells are in
        // [0.0, cols() / Nx], [0.0, rows() / Ny], [0.0, slices() / Nz]
        // = [0.0, 1.0 + 2 * borderWidth / Nx], ...
        // (Currently it's [-borderWidth / Nx, 1 + borderWidth / Nx]...)
        Vector3D iborderWidth(((double) m_borderWidth) / m_Nx,
                              ((double) m_borderWidth) / m_Ny,
                              ((double) m_borderWidth) / m_Nz);
        minCorner += iborderWidth;
        maxCorner += iborderWidth;

        size_t gridStartX, gridStartY, gridStartZ, gridEndX, gridEndY, gridEndZ;
        gridStartX = std::max((long) floor(m_Nx * minCorner[0]), (long) 0);
        gridStartY = std::max((long) floor(m_Ny * minCorner[1]), (long) 0);
        gridStartZ = std::max((long) floor(m_Nz * minCorner[2]), (long) 0);
        gridEndX   = std::min((long)  ceil(m_Nx * maxCorner[0]), (long) cols());
        gridEndY   = std::min((long)  ceil(m_Ny * maxCorner[1]), (long) rows());
        gridEndZ   = std::min((long)  ceil(m_Nz * maxCorner[2]), (long) slices());

        for (size_t slice = gridStartZ; slice < gridEndZ; ++slice) {
            for (size_t row = gridStartY; row < gridEndY; ++row) {
                for (size_t col = gridStartX; col < gridEndX; ++col) {
                    _BBox candidate = cellBoundingBox(slice, row, col);
                    if (candidate.intersectsCircle(pt, radius))
                        cells.push_back(get1DCellIndex(slice, row, col));
                }
            }
        }
    }

    size_t numVertices() const { return vertexSlices() * vertexRows() * vertexCols(); }
    size_t numCells()    const { return slices() * rows() * cols(); }

private:
    private:
        size_t m_Nx, m_Ny, m_Nz;
    protected:
        size_t m_borderWidth;
        _BBox m_bbox;
};

#endif // GRID_HH
