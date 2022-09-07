////////////////////////////////////////////////////////////////////////////////
// VoxelBoundaryMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Extracts a triangulation of the boundary of a voxel grid with optional
//  binary mask specifing voxels that should be removed. This is helpful
//  for efficient visualization of voxel designs.
//
//  The voxel grid must be a 3D array conforming to the
//  `pybind11::unchecked<3>` array interface.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/02/2022 17:06:42
////////////////////////////////////////////////////////////////////////////////
#ifndef VOXELBOUNDARYMESH_HH
#define VOXELBOUNDARYMESH_HH

#include <MeshFEM/Types.hh>
#include "MeshConversion.hh"
#include <map>

struct VoxelBoundaryMesh {
    using A3i = Eigen::Array<int, 3, 1>;
    using FType = Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using VType = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

    template<class NumpyArray3t, class NumpyArray3b>
    static std::unique_ptr<VoxelBoundaryMesh> construct_numpy(const NumpyArray3t &numpy_grid,
               const Eigen::Array3d &dx, const NumpyArray3b *mask_ptr = nullptr,
               char order = 'C') {
        const bool hasMask = mask_ptr && (mask_ptr->ndim() != 0); // apparently pybind11 is passing an unspecified `mask` as an empty one, rather than `nullptr` :/
        if ((numpy_grid.ndim() != 3) || (hasMask && (mask_ptr->ndim() != 3))) throw std::runtime_error("Grids must be 3D");
        auto grid = numpy_grid.template unchecked<3>();
        if (!hasMask)
            return std::make_unique<VoxelBoundaryMesh>(grid, dx, (const NumpyArray3b *) nullptr, order);
        auto mask = mask_ptr->template unchecked<3>();
        return std::make_unique<VoxelBoundaryMesh>(grid, dx, &mask, order);
    }

    struct GridPtCompare {
        bool operator()(const A3i lhs, const A3i rhs) const {
            return std::lexicographical_compare(lhs.data(), lhs.data() + lhs.size(), rhs.data(), rhs.data() + rhs.size());
        }
    };

    template<class Array3t, class Array3b>
    VoxelBoundaryMesh(const Array3t &grid, const Eigen::Array3d &dx,
                      const Array3b *mask_ptr = nullptr,
                      char order = 'C') {
        m_numVoxels = grid.size();
        size_t numFaces = 0;
        s_visitBoundaryFaces([&numFaces](const A3i &/* idx */, size_t /* face */) { ++numFaces; }, grid, mask_ptr);

        std::map<A3i, size_t, GridPtCompare> vtxForIdx;
        const size_t ntris = 2 * numFaces;
        m_F.resize(ntris, 3);
        m_voxelForTri.resize(ntris);

        // We use the following corner numbering convention (differing from GMSH)
        // to simplify 1d => 3d index flattening
        //        y
        // 2----------3
        // |\     ^   |.
        // | \    |   | .
        // |  \   |   |  .
        // |   6------+---7
        // |   |  +-- |-- | -> x
        // 0---+---\--1   |
        //  \  |    \  \  |
        //   \ |     \  \ |
        //    \|      z  \|
        //     4----------5
        // Nodes of faces [x_min, x_max, y_min, y_max, z_min, z_max]
        // oriented outward.
        Eigen::Array<size_t, 6, 4> faces;
        faces << 0, 4, 6, 2,
                 1, 3, 7, 5,
                 0, 1, 5, 4,
                 2, 6, 7, 3,
                 0, 2, 3, 1,
                 4, 5, 7, 6;

        std::vector<Eigen::Vector3f> verts;
        auto insertCorner = [&](const A3i &voxel_idx, size_t c) {
            A3i vert_idx = voxel_idx + A3i(bool(c & (1 << 0)), bool(c & (1 << 1)), bool(c & (1 << 2)));
            auto it = vtxForIdx.find(vert_idx);
            if (it == vtxForIdx.end()) {
                vtxForIdx.emplace(vert_idx, verts.size());
                verts.emplace_back((vert_idx.cast<double>() * dx).template cast<float>().matrix().eval());
                return verts.size() - 1;
            }
            return it->second;
        };

        size_t tri_back = 0;

        A3i idxIncrement;
        if (order == 'C')
            idxIncrement << grid.shape(1) * grid.shape(2), grid.shape(2), 1;
        else if (order == 'F')
            idxIncrement << 1, grid.shape(0), grid.shape(0) * grid.shape(1);
        else throw std::runtime_error("Unknown array storage order");

        s_visitBoundaryFaces([&](const A3i &idx, size_t face) {
                Eigen::Array<size_t, 4, 1> vidxs;
                for (size_t c = 0; c < 4; ++c)
                    vidxs[c] = insertCorner(idx, faces(face, c));
                m_F.row(tri_back++) << vidxs[0], vidxs[1], vidxs[2];
                m_F.row(tri_back++) << vidxs[2], vidxs[3], vidxs[0];

                int flatIdx = (idxIncrement * idx).sum();
                assert(tri_back - 1 < m_voxelForTri.size());
                m_voxelForTri[tri_back - 2] = flatIdx;
                m_voxelForTri[tri_back - 1] = flatIdx;
            }, grid, mask_ptr);

        const size_t nv = verts.size();
        m_V.resize(nv, 3);
        for (size_t i = 0; i < nv; ++i)
            m_V.row(i) = verts[i];
    }

    const VType &vertices() const { return m_V; }
    const FType &   faces() const { return m_F; }
    VType         normals() const {
        const size_t ntris = m_F.rows();
        VType result(ntris, 3);
        for (size_t i = 0; i < ntris; ++i)
            result.row(i) = (m_V.row(m_F(i, 1)) - m_V.row(m_F(i, 0))).cross(m_V.row(m_F(i, 2)) - m_V.row(m_F(i, 0))).normalized();
        return result;
    }

    template<typename T>
    Eigen::MatrixXf visualizationField(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &f) const {
        if (f.rows() != m_numVoxels) throw std::runtime_error("Unexpected field shape");
        const size_t ntris = m_F.rows();
        Eigen::MatrixXf result(ntris, f.cols());
        for (size_t i = 0; i < ntris; ++i)
            result.row(i) = f.row(m_voxelForTri[i]).template cast<float>();
        return result;
    }

private:
    FType m_F;
    VType m_V;
    size_t m_numVoxels;
    Eigen::ArrayXi m_voxelForTri;

    template<class Array3t, class F, class VoxelPresent>
    static void s_visitBoundaryFaces(const F &f, const Array3t &grid, const VoxelPresent &present) {
        A3i shape{int(grid.shape(0)), int(grid.shape(1)), int(grid.shape(2))};
        A3i idx;
        for (idx[0] = 0; idx[0] < shape[0]; ++idx[0]) {
            for (idx[1] = 0; idx[1] < shape[1]; ++idx[1]) {
                for (idx[2] = 0; idx[2] < shape[2]; ++idx[2]) {
                    if (!present(idx)) continue;
                    // Iterate over faces [x_min, x_max, y_min, y_max, z_min, z_max]
                    for (size_t face = 0; face < 6; ++face) {
                        // Neighbor directions `s e_d` for `s in {-1, 1}`
                        size_t d = face / 2;
                        int s = 2 * (face % 2) - 1;
                        A3i nidx = idx;
                        nidx[d] += s;
                        bool neighborMissing = ((nidx[d] < 0) || (nidx[d] >= shape[d]) || !present(nidx));
                        if (neighborMissing)
                            f(idx, face);
                    }
                }
            }
        }
    }

    template<class Array3t, class Array3b, class F>
    static void s_visitBoundaryFaces(const F &f, const Array3t &grid, const Array3b *mask_ptr = nullptr) {
        if (mask_ptr == nullptr)
            s_visitBoundaryFaces(f, grid, [](const A3i &/* idx */) { return true; });
        else {
            const Array3b &mask = *mask_ptr;
            s_visitBoundaryFaces(f, grid, [&mask](const A3i &idx) { return mask(idx[0], idx[1], idx[2]); });
        }
    }
};

#endif /* end of include guard: VOXELBOUNDARYMESH_HH */
