////////////////////////////////////////////////////////////////////////////////
// SystemAssembler.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Functionality for efficient sparse FEM matrix assembly and block sparsity
//  pattern generation.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  06/19/2023 18:35:21
*///////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <array>
#include <atomic>
#include <utility>
#include <limits>
#include <MeshFEM/Handles/FEMMeshHandles.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Flattening.hh>

// Represents the block (vector) structure of variables in the optimization
// problem. We assume that the scalar variables of the optimization problem are
// grouped into vectors of either all the same dimension or a small number of
// distinct dimensions (usually at most 2). In the latter case, the variables
// of each different dimension are collected together for efficiency.
// The dimensions are specified by the `BlockDimensions_` template parameter(s).
template<size_t... BlockDimensions_>
struct OptimizationVarStructure {
    static constexpr size_t NumBlockTypes = sizeof...(BlockDimensions_);
    static constexpr std::array<size_t, NumBlockTypes> BlockDimensions{BlockDimensions_...};
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    struct Block { size_t start, size; };
    Block blockInfo(size_t blockIndex) const {
        if constexpr (NumBlockTypes == 1) { return Block{BlockDimensions[0] * blockIndex, BlockDimensions[0]}; }
        else {
            for (size_t i = 0; i < NumBlockTypes; ++i)
                if (blockIndex < m_typeBlockOffsets[i + 1]) return Block{m_typeVarOffsets[i] + (blockIndex - m_typeBlockOffsets[i]) * BlockDimensions[i], BlockDimensions[i]};
            return Block{NONE, NONE};
        }
    }

    // Query the block size of a given variable.
    size_t blockSize(size_t block) const {
        if constexpr (NumBlockTypes == 1) { return BlockDimensions[0]; }
        else {
            for (size_t i = 0; i < NumBlockTypes; ++i)
                if (block < m_typeBlockOffsets[i + 1]) return BlockDimensions[i];
            return NONE;
        }
    }

    size_t offsetForBlock(size_t block) const {
        if constexpr (NumBlockTypes == 1) { return BlockDimensions[0] * block; }
        else {
            for (size_t i = 0; i < NumBlockTypes; ++i) {
                if (block < m_typeBlockOffsets[i + 1])
                    return m_typeVarOffsets[i] + (block - m_typeBlockOffsets[i]) * BlockDimensions[i];
            }
            return NONE;
        }
    }

    template <typename... Args>
    OptimizationVarStructure(Args... args)
        : m_numBlocksPerType{args...}
    {
        m_typeBlockOffsets[0] = 0;
        m_typeVarOffsets[0] = 0;
        for (size_t i = 0; i < NumBlockTypes; ++i) {
            m_typeBlockOffsets[i + 1] = m_typeBlockOffsets[i] + m_numBlocksPerType[i];
            m_typeVarOffsets  [i + 1] = m_typeVarOffsets  [i] + m_numBlocksPerType[i] * BlockDimensions[i];
        }

        m_numBlocks     = m_typeBlockOffsets[NumBlockTypes];
        m_numScalarVars = m_typeVarOffsets[NumBlockTypes];
    }

    size_t numVars() const { return m_numScalarVars; }
    size_t numBlocks() const { return m_numBlocks; }

private:
    size_t m_numBlocks, m_numScalarVars;
    std::array<size_t, NumBlockTypes> m_numBlocksPerType;
    std::array<size_t, NumBlockTypes + 1> m_typeBlockOffsets;
    std::array<size_t, NumBlockTypes + 1> m_typeVarOffsets;
};

#if 0
template<typename Real_, bool Flatten, size_t numElemLocalVars>
struct PerElementHessian;

// Use a dense numElemLocalVars x numElemLocalVars matrix if not flattening.
template<typename Real_, size_t numElemLocalVars>
struct PerElementHessian<Real_, false, numElemLocalVars> : public Eigen::Matrix<Real_, numElemLocalVars, numElemLocalVars> {
    template<class Derived>
    void addBlock(size_t i, size_t j, const Eigen::MatrixBase<Derived> &b) {
        (*this).template block<Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>(i, j) += b;
    }

    template<class Derived>
    void addStrip(size_t i, size_t j, const Eigen::MatrixBase<Derived> &b) {
        (*this).template segment<Derived::RowsAtCompileTime>(i, j) += b;
    }

    template<size_t Rows>
    auto getStrip(size_t i, size_t j) const {
        Eigen::Matrix<Real_, Rows, 1> result;
        for (size_t bi = 0; bi < Rows; ++bi) {
            result[bi] = (*this)(i + bi, j);
        }
        return (*this).template segment<Rows>(i, j);
    }
};

template<typename Real_, size_t numElemLocalVars>
struct PerElementHessian<Real_, true, numElemLocalVars> : public Eigen::Matrix<Real_, flatLen(numElemLocalVars), 1> {
    using Base = Eigen::Matrix<Real_, flatLen(numElemLocalVars), 1>;
    static constexpr size_t flatten(size_t i, size_t j) {
        return (i < j) ? i + (j * (j + 1)) / 2
                       : j + (i * (i + 1)) / 2;
    }

          Real &operator()(size_t i, size_t j)       { return (*this)[flatten(i, j)]; }
    const Real &operator()(size_t i, size_t j) const { return (*this)[flatten(i, j)]; }

    template<class Derived>
    void addBlock(size_t i, size_t j, const Eigen::MatrixBase<Derived> &b) {
        static_assert(Derived::RowsAtCompileTime > 0 && Derived::ColsAtCompileTime > 0, "Must be a fixed-size block");
        for (size_t bj = 0; bj < Derived::ColsAtCompileTime; ++bj) {
            for (size_t bi = 0; bi < Derived::RowsAtCompileTime; ++bi) {
                (*this)(i + bi, j + bj) += b(bi, bj);
            }
        }
    }

    template<class Derived>
    void addStrip(size_t i, size_t j, const Eigen::MatrixBase<Derived> &b) {
        static_assert(Derived::RowsAtCompileTime > 0 && Derived::ColsAtCompileTime == 1, "Must be a fixed-size column vector");
        for (size_t bi = 0; bi < b.rows(); ++bi) {
            (*this)(i + bi, j) += b[bi];
        }
    }

    template<size_t Rows>
    Eigen::Matrix<Real_, Rows, 1> getStrip(size_t i, size_t j) const {
        Eigen::Matrix<Real_, Rows, 1> result;
        for (size_t bi = 0; bi < Rows; ++bi) {
            result[bi] = (*this)(i + bi, j);
        }
        return result;
    }
};
#endif

template<size_t... BlockDimensions_>
struct SystemAssembler {
    using index_type = SuiteSparse_long;
    using CSCMat = CSCMatrix<index_type, double>;

    template <typename... Args>
    SystemAssembler(Args... args)
        : m_vars(args...)
    {
        static_assert(sizeof...(Args) > 0, "Variables must be initialized!");
        size_t numLocks = m_vars.numBlocks();
        m_varLocks = std::make_unique<std::vector<std::atomic<bool>>>(numLocks);
        for (size_t i = 0; i < numLocks; ++i)
            atomic_init(&(*m_varLocks)[i], false);
    }

    template<class FEMMesh_>
    CSCMat blockSparsityPatternForMesh(const FEMMesh_ &m) const {
        return blockSparsityPattern(m.numNodes(), m.numElements(),
                [&](size_t ei) {
                    std::array<size_t, FEMMesh_::NumNodesPerElement> blockVarsForElement;
                    auto e = m.element(ei);
                    for (const auto n_b : e.nodes()) { blockVarsForElement[n_b.localIndex()] = n_b.index(); }
                    return blockVarsForElement;
                });
    }

    template<class ElemBlockVarsForElement>
    CSCMat blockSparsityPattern(size_t numBlockVars, size_t numElems, const ElemBlockVarsForElement &blockVarsForElement) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("blockSparsityPattern");

#if 0
        struct SparsityTriplet { size_t i, j; };
        std::vector<SparsityTriplet> nz;

        size_t numEntriesPerElement;
        {
            size_t numVarsPerElement = blockVarsForElement(0).size();
            numEntriesPerElement = ((numVarsPerElement + 1) * numVarsPerElement) / 2;
        }
        nz.resize(numElems * numEntriesPerElement);

        // for (size_t ei = 0; ei < numElems; ++ei) {
        parallel_for_range(numElems, [&](size_t ei) {
            const auto &bvars = blockVarsForElement(ei);
            size_t back = ei * numEntriesPerElement;
            for (size_t v_b : bvars) {
                for (size_t v_a : bvars) {
                    if (v_a > v_b) continue;
                    nz[back++] = SparsityTriplet{v_a, v_b};
                }
            }
        });

        // TODO: generating the binned columns directly in a
        // std::vector<std::vector<int>> adjacency list?
        // (Possibly do a counting pass first)

        CSCMat result(numBlockVars, numBlockVars);
        result.symmetry_mode = CSCMat::SymmetryMode::UPPER_TRIANGLE;
        BENCHMARK_SCOPED_TIMER_SECTION timer2("ToCSC");
        sparsityPatternToCSC(numBlockVars, nz, result.Ap, result.Ai);
#else
        CSCMat result(numBlockVars, numBlockVars);
        auto &Ap = result.Ap;
        auto &Ai = result.Ai;
        const size_t n = numBlockVars;
        result.symmetry_mode = CSCMat::SymmetryMode::UPPER_TRIANGLE;

        std::vector<size_t> bucketStart(n + 1);
        {
            size_t *sizes = bucketStart.data() + 1;
            // BENCHMARK_SCOPED_TIMER_SECTION timer1("calc size");
            for (size_t ei = 0; ei < numElems; ++ei) {
                auto bvars = blockVarsForElement(ei);
#if 0 // This seems slower
                std::sort(bvars.begin(), bvars.end());
                for (size_t i = 0; i < bvars.size(); ++i)
                    sizes[bvars[i]] += (i + 1);
#else
                for (size_t v_b : bvars)
                    for (size_t v_a : bvars)
                        if (v_a <= v_b) ++sizes[v_b]; // sizes[v_b] += (v_a <= v_b);
#endif
            }
        }

        size_t origNNZ = 0;
        {
            // Next, compute bucketStart[2:] = cumsum(bucketStart[1:])
            for (size_t j = 1; j <= n; ++j) {
                size_t colsize_j = bucketStart[j];
                bucketStart[j] = origNNZ;
                origNNZ += colsize_j;
            }
        }

        Eigen::Matrix<size_t, Eigen::Dynamic, 1> columnBuckets(origNNZ);
        {
            // BENCHMARK_SCOPED_TIMER_SECTION timer1("fill adjacency");
            // Fill the index buckets; note incrementing the offsets in
            // bucketStart[1:] by the size of each bucket converts these into the
            // end offsets.
            size_t *bucketBack = bucketStart.data() + 1;
            parallel_for_range(numElems, [&](size_t ei) {
                const auto &bvars = blockVarsForElement(ei);
                for (size_t v_b : bvars) {
                    while ((*m_varLocks)[v_b].exchange(true, std::memory_order_acquire)); // lock column gvar_j
                    size_t back = bucketBack[v_b];
                    for (size_t v_a : bvars)
                        if (v_a <= v_b) columnBuckets[back++] = v_a;
                    bucketBack[v_b] = back;
                    (*m_varLocks)[v_b].store(false, std::memory_order_release); // unlock column gvar_j
                }
            });
        }

        // BENCHMARK_SCOPED_TIMER_SECTION timer1("Generate CSCMat");

        Ap.resize(n + 1);

        // Sort each bucket in parallel and deduplicate.
        parallel_for_range(n, [&](size_t j) {
            auto start = columnBuckets.data() + bucketStart[j];
            auto end   = columnBuckets.data() + bucketStart[j + 1];
            std::sort(start, end);
            end = std::unique(start, end);
            Ap[j] = std::distance(start, end); // Write deduplicated bucket size
        });

        // Calculate column pointer array using cumulative sum.
        size_t newNNZ = 0;
        for (size_t j = 0; j < n; ++j) {
            size_t colsize_j = Ap[j];
            Ap[j] = newNNZ;
            newNNZ += colsize_j;
        }
        Ap[n] = newNNZ;

        // Fill row index array `Ai`
        Ai.resize(newNNZ);
        // for (size_t j = 0; j < n; ++j) { // could be parallelized
        parallel_for_range(n, [&](size_t j) {
            size_t offset = bucketStart[j];
            for (index_type ii = Ap[j]; ii < Ap[j + 1]; ++ii)
                Ai[ii] = columnBuckets[offset++];
        });
#endif

        return result;
    }

    CSCMat blockHessianSparsityPatternToScalar(const CSCMat &blockHsp, double val = 0) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("blockHessianSparsityPatternToScalar");
        if (blockHsp.symmetry_mode != CSCMat::SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only SymmetryMode::UPPER_TRIANGLE is supported");
        CSCMat result(m_vars.numVars(), m_vars.numVars());
        result.symmetry_mode = CSCMat::SymmetryMode::UPPER_TRIANGLE;
        CSCMat::InOrderBuilder builder(result, [&blockHsp, this](index_type *colSizes) {
                // Count the number of nonzeros in each column of the scalar Hessian sparsity pattern.
                for (index_type block_j = 0; block_j < blockHsp.n; ++block_j) {
                    auto [gvar_j, bsj] = m_vars.blockInfo(block_j);
                    for (index_type ii = blockHsp.Ap[block_j]; ii < blockHsp.Ap[block_j + 1]; ++ii) {
                        index_type block_i = blockHsp.Ai[ii];
                        if (block_i < block_j) {
                            auto [gvar_i, bsi] = m_vars.blockInfo(block_i);
                            for (size_t c_j = 0; c_j < bsj; ++c_j)
                                colSizes[gvar_j + c_j] += bsi;
                        }
                        else {
                            for (size_t c_j = 0; c_j < bsj; ++c_j)
                                colSizes[gvar_j + c_j] += (c_j + 1);
                        }
                    }
                }
            });

        // BENCHMARK_SCOPED_TIMER_SECTION timer2("builderFiller");
        // Filling out the index arrays (can be done in parallel)
        for (index_type block_j = 0; block_j < blockHsp.n; ++block_j) {
            auto [gvar_j, bsj] = m_vars.blockInfo(block_j);
            for (index_type ii = blockHsp.Ap[block_j]; ii < blockHsp.Ap[block_j + 1]; ++ii) {
                index_type block_i = blockHsp.Ai[ii];
                if (block_i < block_j) {
                    auto [gvar_i, bsi] = m_vars.blockInfo(block_i);
                    for (size_t c_j = 0; c_j < bsj; ++c_j)
                        for (size_t c_i = 0; c_i < bsi; ++c_i)
                            builder.insert(gvar_i + c_i, gvar_j + c_j, val);
                }
                else {
                    for (size_t c_j = 0; c_j < bsj; ++c_j)
                        for (size_t c_i = 0; c_i <= c_j; ++c_i)
                            builder.insert(gvar_j + c_i, gvar_j + c_j, val);
                }
            }
        }

        return result;
    }

    // Assemble the per-element Hessian `eval_HE(ei)` for element ei in 0..ne.
    // The element's global block variable indices are obtained by calling `element(ei)`,
    // which should either return a FEMMesh element handle or an array of variable indices.
    template<class SPMat, class PEHEval, class ElementGetter>
    void assembleHessian(SPMat &H, size_t ne, const PEHEval &eval_He, const ElementGetter &element) const {
        get_hessian_assembly_arena().execute([&H, &eval_He, &element, ne, this]() {
            parallel_for_range(ne, [&H, &eval_He, &element, this](size_t ei) {
                m_assembleHessianContrib(H, eval_He(ei), element(ei));
            });
        });
    }

    // Convenience method for the typical case of assembling a per-element Hessian using
    // using nodal variables of a FEMMesh.
    template<class SPMat, class Mesh, class PEHEval>
    void assembleHessian(SPMat &H, const Mesh &m, const PEHEval &eval_He) const {
        get_hessian_assembly_arena().execute([&H, &eval_He, &m, this]() {
            parallel_for_range(m.numElements(), [&H, &eval_He, &m, this](size_t ei) {
                m_assembleHessianContrib(H, eval_He(ei), m.element(ei));
            });
        });
    }

private:
    template<class SPMat, class PEH, class ElemBlockVars>
    void m_assembleHessianContrib(SPMat &H, const PEH &H_e, const ElemBlockVars &blockVars) const {
        size_t lvar_j = 0;
        for (size_t bj : blockVars) {
            auto [gvar_j, bsj] = m_vars.blockInfo(bj);
            size_t lvar_i = 0;
            while ((*m_varLocks)[bj].exchange(true, std::memory_order_acquire)); // lock column gvar_j
            for (size_t bi : blockVars) {
                auto [gvar_i, bsi] = m_vars.blockInfo(bi);
                if (gvar_i < gvar_j) {
                    index_type idx = H.findEntry(gvar_i, gvar_j);
                    for (size_t c = 0; c < bsj; ++c) {
                        typename SPMat::DataMap(H.Ax.data() + idx, bsj) += H_e.col(lvar_j + c).segment(lvar_i, bsj);
                        // Advance to the start of the block in the next columnn
                        // (assuming the next column has an identical sparsity
                        // pattern in rows 0...gvar_i)
                        idx += H.col_nnz(gvar_j + c);
                    }
                }
                else if (gvar_i == gvar_j) {
                    index_type idx = H.findDiagEntry(gvar_i); // Top of strip to add
                    for (size_t c = 0; c < bsj; ++c) {
                        typename SPMat::DataMap(H.Ax.data() + idx, c + 1) += H_e.col(lvar_j + c).segment(lvar_i, c + 1);
                        idx += H.col_nnz(gvar_j + c);
                    }
                }

                lvar_i += bsi;
            }
            (*m_varLocks)[bj].store(false, std::memory_order_release); // unlock column gvar_j
            lvar_j += bsj;
        }
    }

    template<class SPMat, class PEH, class _Mesh>
    void m_assembleHessianContrib(SPMat &H, const PEH &H_e, const _FEMMeshHandles::EHandle<_Mesh> &e) const {
        std::array<size_t, _FEMMeshHandles::EHandle<_Mesh>::numNodes()> elemBlockVars;
        for (const auto n_b : e.nodes()) { elemBlockVars[n_b.localIndex()] = n_b.index(); }
        m_assembleHessianContrib(H, H_e, elemBlockVars);
    }

    mutable std::unique_ptr<std::vector<std::atomic<bool>>> m_varLocks;
    OptimizationVarStructure<BlockDimensions_...> m_vars;
};
