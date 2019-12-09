////////////////////////////////////////////////////////////////////////////////
// CollisionGrid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Accelerates nearby point queries by binning objects into a (sparse) grid
//      of cells and pruning cells not overlapping the query sphere. This is a
//      quick hack to get around needing a true AABB or similar datastructure.
//
//      (This is a grid centered around the origin with lattice width cellSize)
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/16/2014 07:22:33
////////////////////////////////////////////////////////////////////////////////
#ifndef COLLISIONGRID_HH
#define COLLISIONGRID_HH

#include <map>
#include <vector>
#include <cassert>
#include <utility>

template<typename Real, typename Point>
class CollisionGrid {
public:
    constexpr size_t Dim() const { return Point::RowsAtCompileTime; }

    CollisionGrid(Real cellSize) {
        assert(Dim() <= 3);
        reset(cellSize);
    }
    void reset() { m_cells.clear(); }
    void reset(Real cellsize) { m_cellSize = cellsize; reset(); }

    ////////////////////////////////////////////////////////////////////////////
    /*! Add a tagged point to the collection, creating a new bin for it if
    //  necessary.
    *///////////////////////////////////////////////////////////////////////////
    void addPoint(const Point &p, size_t tag) {
        CellIdx idx;
        for (size_t i = 0; i < Dim(); ++i)
            idx[i] = floor(p[i] / m_cellSize);
        auto it = m_cells.find(idx);
        if (it != m_cells.end())
            it->second.emplace_back(std::make_pair(p, tag));
        else {
            m_cells.emplace(idx, Bin { std::make_pair(p, tag) } );
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Find the closest point to p, only allowing points within distance eps
    //  @param[in]  p   query point
    //  @param[in]  eps distance p must be within to consider it matched
    //  @return     tag, point closest point and its tag if found.
    //                          "tag" is -1 if not found.
    *///////////////////////////////////////////////////////////////////////////
    std::pair<int, Point> getClosestPoint(const Point &p, Real eps) const {
        CellIdx idxMin, idxMax;
        for (size_t i = 0; i < Dim(); ++i) {
            idxMin[i] = floor((p[i] - eps) / m_cellSize);
            idxMax[i] = floor((p[i] + eps) / m_cellSize);
        }
        int closestTag = -1;
        Real closestDist = eps;
        Point closestPoint(Point::Zero()); // Zero-init to avoid warnings
        for (int i = idxMin[0]; i <= idxMax[0]; ++i) {
            for (int j = idxMin[1]; j <= idxMax[1]; ++j) {
                for (int k = idxMin[2]; k <= idxMax[2]; ++k) {
                    CellIdx bin(i, j, k);
                    auto it = m_cells.find(bin);
                    if (it != m_cells.end()) {
                        const Bin &b = it->second;
                        for (size_t l = 0; l < b.size(); ++l) {
                            Real dist = (p - b[l].first).norm();
                            if (dist <= closestDist) {
                                closestDist = dist;
                                closestPoint = b[l].first;
                                closestTag = b[l].second;
                            }
                        }
                    }
                }
            }
        }

        return std::make_pair(closestTag, closestPoint);
    }


    // In {1,2}D, higher-dim components are effectively ignored since they all
    // equal (0)
    struct CellIdx {
        CellIdx(int a = 0, int b = 0, int c = 0) {
            idx[0] = a; idx[1] = b; idx[2] = c;
        }
        int idx[3];
        // Lexicographic ordering.
        bool operator<(const CellIdx &b) const {
            for (size_t i = 0; i < 3; ++i) {
                if (idx[i] < b.idx[i]) return true;
                if (idx[i] > b.idx[i]) return false;
            }
            return false;
        }

              int &operator[](size_t i)       { assert(i < 3); return idx[i]; }
        const int &operator[](size_t i) const { assert(i < 3); return idx[i]; }
    };

private:
    Real m_cellSize;
    typedef std::vector<std::pair<Point, size_t> > Bin;
    std::map<CellIdx, Bin> m_cells;
};

#endif /* end of include guard: COLLISIONGRID_HH */
