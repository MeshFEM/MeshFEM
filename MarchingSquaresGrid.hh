////////////////////////////////////////////////////////////////////////////////
// MarchingSquaresGrid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements an axis-aligned grid that can extract the polygon of an
//      object with marching cubes.
//      
//      The grid is padded with a border of outside-object vertices to handle
//      the case where the object bounding box is tight (otherwise no boundary
//      would be found). This is done by making the actual grid size
//      Nx + 2, Ny + 2 and by growing the bounding box accordingly.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/18/2013 23:07:39
////////////////////////////////////////////////////////////////////////////////
#ifndef MARCHING_SQUARES_GRID_HH
#define MARCHING_SQUARES_GRID_HH

#include <vector>
#include "Grid.hh"

class MarchingSquaresGrid : public Grid2D {
public:
    MarchingSquaresGrid(size_t Nx, size_t Ny)
        : Grid2D(Nx + 2, Ny + 2, BBox_t()) { }

    void setGridSize(size_t Nx, size_t Ny) {
        m_Nx = Nx + 2;
        m_Ny = Ny + 2;
    }

    void m_mergePolygon(Scalar mergeThreshold, Polygon_t &boundary) const;

    template<typename Model>
    void extractBoundaryPolygons(const Model &model, std::vector<Polygon_t> &p,
                                 typename Model::Real mergeThreshold = .10);

private:
    template<typename Model>
    Polygon_t m_extractPolygon(const Model &model, size_t ci,
                               const std::vector<char> &cellCornerCase,
                               std::vector<char> &visitCount);
};

#endif // MARCHING_SQUARES_GRID_HH
