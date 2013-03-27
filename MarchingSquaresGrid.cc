////////////////////////////////////////////////////////////////////////////////
// MarchingSquaresGrid.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements the marching squares algorithm for extracting model boundaries.
//  This implementation uses the following numbering of cell cases:
//
//      16 Cases of the corners:
//      .---.  .---.  .---.  .---.
//      | 0 |  | 1 |  | 2 |  | 3 |
//      '---'  *---'  '---*  *---*
//      .---*  .---*  .---*  .---*
//      | 4 |  | 5 |  | 6 |  | 7 |
//      '---'  *---'  '---*  *---*
//      *---.  *---.  *---.  *---.
//      | 8 |  | 9 |  |10 |  |11 |
//      '---'  *---'  '---*  *---*
//      *---*  *---*  *---*  *---*
//      |12 |  |13 |  |14 |  |15 |
//      '---'  *---'  '---*  *---*
//
//      Disambiguations: The cell center must be sampled. Also, entrance
//      (previous) direction determines action, so encode it in the state...
//
//      Corner case 5 disambiguation:
//      .---*           .---*    .---*
//      | 5 |  ==>      | * |    | O | 
//      *---'           *---'    *---'
//      Entrance Dir:   D: 16    D: 18
//                      U: 17    U: 19
//
//      Corner case 10 disambiguation:
//      *---.           *---.    *---.
//      |10 |  ==>      | * |    | O | 
//      '---*           '---*    '---*
//      Entrance Dir:   R: 20    R: 22
//                      L: 21    L: 23
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/19/2013 00:19:25
////////////////////////////////////////////////////////////////////////////////
#include "MarchingSquaresGrid.hh"
#include "GlobalTypes.hh"
#include <cassert>

using namespace std;

typedef enum { MS_UP, MS_LEFT, MS_DOWN, MS_RIGHT, MS_NONE } Direction;

// Note: movement directions also indicate which new border point to add.
const Direction movement[24] = {
        // 16 Cases of the Corners
        MS_NONE,  MS_LEFT,  MS_DOWN, MS_LEFT,
        MS_RIGHT, MS_NONE,  MS_DOWN, MS_LEFT,
        MS_UP,    MS_UP,    MS_NONE, MS_UP,
        MS_RIGHT, MS_RIGHT, MS_DOWN, MS_NONE,
        // Disambiguation of case 5
        MS_LEFT,  MS_RIGHT, // Middle point inside,  Previous Dir: D, U
        MS_RIGHT, MS_LEFT,  // Middle point outside, Previous Dir: D, U
        // Disambiguation of case 10
        MS_DOWN, MS_UP,     // Middle point inside,  Previous Dir: R, L
        MS_UP,   MS_DOWN    // Middle point outside, Previous Dir: R, L
};

// Records the boundary of the model component containing cell ci, starting at
// ci. Traverses counter-clockwise.
//
// Assumes cellCornerCase was constructed from a grid where all the boundary
// vertices are outside the object (to keep the marching squares algorithm
// within the grid). This is done by extractBoundaryPolygons.
template<typename Model>
Polygon_t MarchingSquaresGrid::
m_extractPolygon(const Model &model, size_t ci,
                 const vector<char> &cellCornerCase,
                 vector<char> &visitCount)
{
    assert(visitCount.size() == cellCornerCase.size());

    Polygon_t border;

    Direction prevDir = MS_NONE;

    while (true) {
        assert(ci < cellCornerCase.size());
        int cellCase = cellCornerCase[ci];
        bool diagonalCase = ((cellCase == 5) || (cellCase == 10));

        // When we get to a seen cell again, (unless it's a "diagonal" case
        // requiring two visits).
        if ((visitCount[ci] > 0) || (diagonalCase && (visitCount[ci] > 1)))
            break;

        // We are visiting this cell now!
        ++visitCount[ci];

        // Encode disambiguating state for the diagonal cases
        if (cellCase == 5) {
            cellCase = 16 + (model.isInside(cellMidpointPosition(ci)) ? 0 : 2)
                          + (prevDir == MS_DOWN ? 0 : 1);
        }
        else if (cellCase == 10) {
            cellCase = 20 + (model.isInside(cellMidpointPosition(ci)) ? 0 : 2)
                          + (prevDir == MS_RIGHT ? 0 : 1);
        }

        // Now decode case to get movement and new point generation
        size_t row, col;
        get2DCellIndex(ci, row, col);
        Vector newPoint;
        Direction dir = movement[cellCase];

        Scalar a, b, s = 0.5;
        Grid2D::AdjacencyVec corners;
        cellVertices(ci, corners);

        // Assumes corner numbering:
        //      3--2
        //      |  |
        //      0--1
        switch(dir) {
            case MS_LEFT:
                --col;
                a = model.signedDistance(vertexPosition(corners[0]));
                b = model.signedDistance(vertexPosition(corners[3]));
                s = a / (a - b); // Approximate zero crossing
                assert(s <= 1.0 && s >= 0.0);
                newPoint = Vector(0, s);
                break;
            case MS_DOWN:
                --row;
                a = model.signedDistance(vertexPosition(corners[0]));
                b = model.signedDistance(vertexPosition(corners[1]));
                s = a / (a - b); // Approximate zero crossing
                newPoint = Vector(s, 0);
                break;
            case MS_RIGHT:
                ++col;
                a = model.signedDistance(vertexPosition(corners[1]));
                b = model.signedDistance(vertexPosition(corners[2]));
                s = a / (a - b); // Approximate zero crossing
                newPoint = Vector(1.0, s);
                break;
            case MS_UP:
                ++row;
                a = model.signedDistance(vertexPosition(corners[3]));
                b = model.signedDistance(vertexPosition(corners[2]));
                s = a / (a - b); // Approximate zero crossing
                newPoint = Vector(s, 1.0);
                break;
            default:
                // We better be making a valid movement!
                assert(false);
        }

        border.addPoint(cellBoundingBox(ci).interpolatePoint(newPoint));
        // Move to the next cell
        ci = get1DCellIndex(row, col);
        prevDir = dir;
    }

    return border;
}

template<typename Model>
void MarchingSquaresGrid::extractBoundaryPolygons(const Model &model,
                                  vector<Polygon_t> &p,
                                  typename Model::Real mergeThreshold)
{
    typedef typename Model::Real Real;

    m_bbox = model.boundingBox();
    // Expand the bounding box to account for the fact that a border has been
    // added. Also expand outward slightly more to avoid instabilities/issues
    // when the bounding box is tight.
    Scalar eps = 0.0; // 1e-2;
    m_bbox.expand(Vector((2.0 + eps) / (m_Nx - 2.0),
                         (2.0 + eps) / (m_Ny - 2.0)));
    
    vector<bool> vertexInside(numVertices(), false);
    for (size_t v = 0; v < numVertices(); ++v) {
        size_t row, col;
        get2DVertexIndex(v, row, col);
        // Note: vertex indices range from (0, 0) to (Nx, Ny)
        bool gridBorder = (row == 0)    || (col == 0) ||
                          (row == m_Ny) || (col == m_Nx);
        // All grid border vertices are marked as outside the object.
        vertexInside[v] = gridBorder ? false
                                     : model.isInside(vertexPosition(v));
    }

    // Compute the (pre-disambiguation) corner case.
    // Assumes corner numbering:
    //      3--2
    //      |  |
    //      0--1
    //  Then corner c on/off is mapped to bit c on/off.
    vector<char> cellCornerCase(numCells(), 0);
    Grid2D::AdjacencyVec corners;
    for (size_t ci = 0; ci < numCells(); ++ci) {
        cellVertices(ci, corners);
        for (size_t c = 0; c < (size_t) corners.rows(); ++c) {
            if (vertexInside[corners[c]])
                cellCornerCase[ci] |= (1 << c);
        }
    }

    vector<char> visitCount(numCells(), 0);

    p.clear();
    for (size_t ci = 0; ci < numCells(); ++ci) {
        char cCase = cellCornerCase[ci];
        if ((cCase > 0) && (cCase < 15) && (visitCount[ci] == 0)) {
            Polygon_t boundary;
            if ((cCase == 5) || (cCase == 10)) {
                // Case 5 and 10 are the difficult diagonal cases whose exit
                // directions depend on the entrance direction... Let's not
                // start there.
                continue;
            }
            boundary = m_extractPolygon(model, ci, cellCornerCase, visitCount);

            // Merge pairs of points that are within the merging threshold of
            // each other. The following code assumes mergeThreshold < 1.0, so
            // no more than two adjacent points will be close enough for
            // merging.
            Real mthresh = mergeThreshold * cellSize().minCoeff();
            std::vector<Vector> &pts = boundary.points;
            size_t numMergedPoints = 0;
            // Handle all points except the last
            for (size_t i = 0; i < pts.size() - 1; ++i) {
                Vector p = pts[i];
                Vector pNext = pts[i + 1];
                if ((pNext - p).norm() < mthresh) {
                    // Merged point becomes the average
                    p += pNext;
                    p *= .5;
                    ++i; // Skip the point we merged with
                }

                pts[++numMergedPoints] = p;
            }
            // Handle the last point
            if ((pts.back() - pts[0]).norm() < mthresh) {
                // Merging with the first point repositions it
                pts[0] += pts.back();
                pts[0] *= .5;
            }
            else {
                pts[++numMergedPoints] = pts.back();
            }

            pts.resize(numMergedPoints);
            p.push_back(boundary);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Template instantiations
////////////////////////////////////////////////////////////////////////////////
template void MarchingSquaresGrid::
extractBoundaryPolygons<CSGTree_t>(const CSGTree_t &model,
                                   vector<Polygon_t> &p,
                                   typename CSGTree_t::Real mergeThreshold);
