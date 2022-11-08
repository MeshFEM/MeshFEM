////////////////////////////////////////////////////////////////////////////////
// VoxelBoundaryMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Extracts a triangulation of the boundary of a voxel grid with optional
//  binary mask specifing voxels that should be removed. This is helpful
//  for efficient visualization of voxel designs.
//
//  The 2D case is handled by considering every pixel to comprise a single
//  "boundary face" (so that the full grid is drawn if `mask` is None).
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
    using FType = Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using VType = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

    template<class NumpyArrayb>
    static std::unique_ptr<VoxelBoundaryMesh> construct_numpy(const Eigen::ArrayXi &grid_shape,
               const Eigen::ArrayXd &dx, const NumpyArrayb *mask_ptr = nullptr,
               char order = 'C') {
        const bool hasMask = mask_ptr && (mask_ptr->ndim() != 0); // apparently pybind11 is passing an unspecified `mask` as an empty one, rather than `nullptr` :/
        int dim = grid_shape.size();
        if ((dim != 2) && (dim != 3)) throw std::runtime_error("Grids must be 2D or 3D");
        if (hasMask && (mask_ptr->ndim() != dim)) throw std::runtime_error("Mask must be of the same dimension as the grid");
        if (!hasMask) {
            if (dim == 2) return std::make_unique<VoxelBoundaryMesh>(grid_shape.head<2>().eval(), dx.head<2>().eval(), (const NumpyArrayb *) nullptr, order);
            return               std::make_unique<VoxelBoundaryMesh>(grid_shape.head<3>().eval(), dx.head<3>().eval(), (const NumpyArrayb *) nullptr, order);
        }

        if (dim == 2) { auto mask = mask_ptr->template unchecked<2>(); return std::make_unique<VoxelBoundaryMesh>(grid_shape.head<2>().eval(), dx.head<2>().eval(), &mask, order); }
                        auto mask = mask_ptr->template unchecked<3>(); return std::make_unique<VoxelBoundaryMesh>(grid_shape.head<3>().eval(), dx.head<3>().eval(), &mask, order);

    }

    template<int N, class Array_Nd_b> // `N` must be `int` for deduction from `Eigen::Array` template parameters
    VoxelBoundaryMesh(const Eigen::Array<int, N, 1> &grid_shape, const Eigen::Array<double, N, 1> &dx,
                      const Array_Nd_b *mask_ptr = nullptr,
                      char order = 'C') {
        using ANi = Eigen::Array<int, N, 1>;
        m_numVoxels     = grid_shape.prod();
        m_numGridPoints = (grid_shape + 1).prod();
        size_t numFaces = 0;
        s_visitBoundaryFaces([&numFaces](const ANi &/* idx */, size_t /* face */) { ++numFaces; }, grid_shape, mask_ptr);

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
        if (N == 3) {
            faces << 0, 4, 6, 2,
                     1, 3, 7, 5,
                     0, 1, 5, 4,
                     2, 6, 7, 3,
                     0, 2, 3, 1,
                     4, 5, 7, 6;
        }
        else {
            faces.row(0) << 0, 1, 3, 2;
        }

        std::vector<Eigen::Vector3f> verts;
        ANi gridPtIdxIncrement, voxIdxIncrement;
        if (N == 3) {
            if (order == 'C') {
                voxIdxIncrement    <<  grid_shape[1] *       grid_shape[2],       grid_shape[2],      1;
                gridPtIdxIncrement << (grid_shape[1] + 1) * (grid_shape[2] + 1), (grid_shape[2] + 1), 1;
            } else if (order == 'F') {
                voxIdxIncrement    << 1,  grid_shape[0]     ,  grid_shape[0]      *  grid_shape[1]     ;
                gridPtIdxIncrement << 1, (grid_shape[0] + 1), (grid_shape[0] + 1) * (grid_shape[1] + 1);
            }
            else throw std::runtime_error("Unknown array storage order");
        }
        if (N == 2) {
            if (order == 'C') {
                voxIdxIncrement    <<  grid_shape[1]     , 1;
                gridPtIdxIncrement << (grid_shape[1] + 1), 1;
            } else if (order == 'F') {
                voxIdxIncrement    << 1,  grid_shape[0]     ;
                gridPtIdxIncrement << 1, (grid_shape[0] + 1);
            }
            else throw std::runtime_error("Unknown array storage order");
        }

        auto flattenedGridPtIdx = [&](auto idxND) { return (gridPtIdxIncrement * idxND).sum(); };
        auto    flattenedVoxIdx = [&](auto idxND) { return (   voxIdxIncrement * idxND).sum(); };

        auto insertCorner = [&](const ANi &voxel_idx, size_t c) {
            ANi grid_pt = voxel_idx;
            for (int i = 0; i < N; ++i)
                grid_pt[i] += bool(c & (1 << i));
            size_t vert_idx = flattenedGridPtIdx(grid_pt);
            auto it = m_vtxForGridPt.find(vert_idx);
            if (it == m_vtxForGridPt.end()) {
                m_vtxForGridPt.emplace(vert_idx, verts.size());
                verts.emplace_back(padTo3D((grid_pt.template cast<double>() * dx).template cast<float>().matrix().eval()));
                return verts.size() - 1;
            }
            return it->second;
        };

        int tri_back = 0;

        s_visitBoundaryFaces([&](const ANi &idx, size_t face) {
                Eigen::Array<size_t, 4, 1> vidxs;
                for (size_t c = 0; c < 4; ++c)
                    vidxs[c] = insertCorner(idx, faces(face, c));
                m_F.row(tri_back++) << vidxs[0], vidxs[1], vidxs[2];
                m_F.row(tri_back++) << vidxs[2], vidxs[3], vidxs[0];

                int flatIdx = flattenedVoxIdx(idx);
                assert(tri_back - 1 < m_voxelForTri.size());
                m_voxelForTri[tri_back - 2] = flatIdx;
                m_voxelForTri[tri_back - 1] = flatIdx;
            }, grid_shape, mask_ptr);

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
        if (size_t(f.rows()) == m_numVoxels) {
            const size_t ntris = m_F.rows();
            Eigen::MatrixXf result(ntris, f.cols());
            for (size_t i = 0; i < ntris; ++i)
                result.row(i) = f.row(m_voxelForTri[i]).template cast<float>();
            return result;
        }
        if (size_t(f.rows()) == m_numGridPoints) {
            Eigen::MatrixXf result(m_vtxForGridPt.size(), f.cols());
            for (auto &entry : m_vtxForGridPt) {
                result.row(entry.second) = f.row(entry.first).template cast<float>();
            }
            return result;
        }
        throw std::runtime_error("Unexpected field shape");
    }

private:
    FType m_F;
    VType m_V;
    size_t m_numVoxels, m_numGridPoints;
    Eigen::ArrayXi m_voxelForTri;
    std::map<size_t, size_t> m_vtxForGridPt; // sparse map from grid point indices to the output vertex

    template<int N, class F, class VoxelPresent>
    static void s_visitBoundaryFaces(const F &f, const Eigen::Array<int, N, 1> &grid_shape, const VoxelPresent &present) {
        using ANi = Eigen::Array<int, N, 1>;
        ANi idx;
        for (idx[0] = 0; idx[0] < grid_shape[0]; ++idx[0]) {
            for (idx[1] = 0; idx[1] < grid_shape[1]; ++idx[1]) {
                if (N == 3) {
                    for (idx[2] = 0; idx[2] < grid_shape[2]; ++idx[2]) {
                        if (!present(idx)) continue;
                        // Iterate over faces [x_min, x_max, y_min, y_max, z_min, z_max]
                        for (size_t face = 0; face < 6; ++face) {
                            // Neighbor directions `s e_d` for `s in {-1, 1}`
                            size_t d = face / 2;
                            int s = 2 * (face % 2) - 1;
                            ANi nidx = idx;
                            nidx[d] += s;
                            bool neighborMissing = ((nidx[d] < 0) || (nidx[d] >= grid_shape[d]) || !present(nidx));
                            if (neighborMissing)
                                f(idx, face);
                        }
                    }
                }
                if (N == 2) {
                    if (!present(idx)) continue;
                    f(idx, 0);
                }
            }
        }
    }

    template<class Array_Nd_b, class ANi, size_t... Idxs>
    static auto s_accessNDArray(const Array_Nd_b &a, const ANi &idx, std::index_sequence<Idxs...>) {
        return a(idx[Idxs]...);
    }

    template<int N, class Array_Nd_b, class F>
    static void s_visitBoundaryFaces(const F &f, const Eigen::Array<int, N, 1> &grid_shape, const Array_Nd_b *mask_ptr = nullptr) {
        using ANi = Eigen::Array<int, N, 1>;
        if (mask_ptr == nullptr)
            s_visitBoundaryFaces<N>(f, grid_shape, [](const ANi &/* idx */) { return true; });
        else {
            const Array_Nd_b &mask = *mask_ptr;
            s_visitBoundaryFaces<N>(f, grid_shape, [&mask](const ANi &idx) { return s_accessNDArray(mask, idx, std::make_index_sequence<N>()); });
        }
    }
};

#endif /* end of include guard: VOXELBOUNDARYMESH_HH */
