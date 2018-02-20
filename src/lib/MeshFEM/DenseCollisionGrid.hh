////////////////////////////////////////////////////////////////////////////////
// DenseCollisionGrid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Dense grid to accelerate queries for the aabbs overlapping a point.
//      The application in mind is finding which cells (triangles or tetrahedra)
//      from a mesh a point is in: these cells will densely cover a large
//      fraction of the mesh's bounding box (for "nice" meshes), so a dense grid
//      is appropriate.
//      
//      For this application, the cells' bounding boxes will be added to the
//      grid, and a list of the query point's enclosing bounding boxes is
//      returned. Then it's up to the caller to determine which of these
//      bounding boxes' elements actually contains the point.
//
//      Grid cells are indexed by their min corner grid position.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/18/2016 11:38:04
////////////////////////////////////////////////////////////////////////////////
#ifndef DENSECOLLISIONGRID_HH
#define DENSECOLLISIONGRID_HH
#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <MeshFEM/Geometry.hh>

template<size_t N>
struct DenseGridStorage;

// template<size_t N>
// using GridVector = Eigen::Matrix<Real, 2, 1, Eigen::DontAlign>

template<>
struct DenseGridStorage<2> {
    using BinIndex = std::array<size_t, 2>;
    using BBox = ::BBox<VectorND<2>>;
    using TaggedBBox = std::pair<BBox, size_t>;
    using Bin = std::vector<TaggedBBox, Eigen::aligned_allocator<TaggedBBox>>;
    DenseGridStorage(size_t size) : data(size, std::vector<Bin>(size)) { }
    const Bin &getBin(const BinIndex &i) const { return data.at(i[0]).at(i[1]); }
          Bin &getBin(const BinIndex &i)       { return data.at(i[0]).at(i[1]); }
    void addBoxToBinRange(const BBox &bb, size_t tag,
                          const BinIndex &minIdx, const BinIndex &maxIdx) {
        for (    size_t i = minIdx[0]; i <= maxIdx[0]; ++i)
            for (size_t j = minIdx[1]; j <= maxIdx[1]; ++j)
                getBin(BinIndex{{i, j}}).push_back({bb, tag});
    }

    std::vector<size_t> binSizeHistogram() const {
        std::vector<size_t> counts;
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                size_t count = data[i][j].size();
                if (counts.size() <= count) counts.resize(count + 1);
                ++counts.at(count);
            }
        }
        return counts;
    }

    std::vector<std::vector<Bin>> data;
};

template<>
struct DenseGridStorage<3> {
    using BinIndex = std::array<size_t, 3>;
    using BBox = ::BBox<VectorND<3>>;
    using TaggedBBox = std::pair<BBox, size_t>;
    using Bin = std::vector<TaggedBBox, Eigen::aligned_allocator<TaggedBBox>>;

    DenseGridStorage(size_t size) : data(size, std::vector<std::vector<Bin>>(size, std::vector<Bin>(size))) { }
    const Bin &getBin(const BinIndex &i) const { return data[i[0]][i[1]][i[2]]; }
          Bin &getBin(const BinIndex &i)       { return data[i[0]][i[1]][i[2]]; }
    void addBoxToBinRange(const BBox &bb, size_t tag,
                          const BinIndex &minIdx, const BinIndex &maxIdx) {
        for (        size_t i = minIdx[0]; i <= maxIdx[0]; ++i)
            for (    size_t j = minIdx[1]; j <= maxIdx[1]; ++j)
                for (size_t k = minIdx[2]; k <= maxIdx[2]; ++k)
                    getBin(BinIndex{{i, j, k}}).push_back({bb, tag});
    }

    std::vector<size_t> binSizeHistogram() const {
        std::vector<size_t> counts;
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                for (size_t k = 0; k < data[i][j].size(); ++k) {
                    size_t count = data[i][j][k].size();
                    if (counts.size() <= count) counts.resize(count + 1);
                    ++counts.at(count);
                }
            }
        }
        return counts;
    }

    std::vector<std::vector<std::vector<Bin>>> data;
};

template<size_t N>
class DenseCollisionGrid {
public:
    using Point      = VectorND<N>;
    using Vector     = VectorND<N>;
    using BBox       = ::BBox<Point>;
    using TaggedBBox = std::pair<BBox, size_t>;
    using Bin        = std::vector<TaggedBBox>;
    using BinIndex   = typename DenseGridStorage<N>::BinIndex;

    DenseCollisionGrid(size_t size, const BBox &bb)
        : m_bb(bb), m_storage(size), m_gridSize(size), m_cellSize(m_bb.dimensions() / Real(size)) { }

    void addBox(const BBox &bb, size_t tag) {
        m_storage.addBoxToBinRange(bb, tag, binIndexForPoint(bb.minCorner),
                                            binIndexForPoint(bb.maxCorner));
    }

    std::vector<size_t> enclosingBoxes(const Point &p) const {
        std::vector<size_t> result;
        const auto &bin = m_storage.getBin(binIndexForPoint(p));
        for (const auto &box : bin)
            if (box.first.containsPoint(p)) result.push_back(box.second);

        return result;
    }

    BinIndex binIndexForPoint(const Point &p) const {
        BinIndex idx;
        for (size_t c = 0; c < N; ++c) {
            int i = floor((p[c] - m_bb.minCorner[c]) / m_cellSize[c]);
            idx[c] = size_t(std::max(i, 0));
            idx[c] = std::min(idx[c], m_gridSize - 1);
        }

        // std::cout << "m_bb.minCorner[i]
        // std::cout << "bin idx for pt " << p[0] << ", " << p[1]
        //           << ": " << idx[0] << ", " << idx[1] << std::endl;

        return idx;
    }

    void printBinSizeHistogram() const {
        auto counts = m_storage.binSizeHistogram();
        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] != 0)
                std::cout << i << ": " << counts[i] << std::endl;
        }
    }
    
private:
    BBox m_bb;
    DenseGridStorage<N> m_storage;
    size_t m_gridSize;
    Vector m_cellSize;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif /* end of include guard: DENSECOLLISIONGRID_HH */
