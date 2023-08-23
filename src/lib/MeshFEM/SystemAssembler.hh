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
#ifndef SYSTEMASSEMBLER_HH
#define SYSTEMASSEMBLER_HH

#include <vector>
#include <array>
#include <atomic>
#include <utility>
#include <limits>
#include <tuple>
#include <MeshFEM/Handles/FEMMeshHandles.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Flattening.hh>
#include "Utilities/static_sort.hh"

// Represents the block (vector) structure of variables in the optimization
// problem. We assume that the scalar variables of the optimization problem are
// grouped into vectors of either all the same dimension or a small number of
// distinct dimensions (usually at most 2). In the latter case, the variables
// of each different dimension are collected together for efficiency.
// The dimensions are specified by the `BlockDimensions_` template parameter(s).
template<size_t... BlockDimensions_>
struct OptimizationVarStructure {
    static constexpr size_t FirstBlockDim  = std::get<0>(std::make_tuple(BlockDimensions_...));
    static constexpr size_t NumBlockTypes  = sizeof...(BlockDimensions_);
    static constexpr size_t MinBlockDim    = std::min({BlockDimensions_...});
    static constexpr size_t MaxBlockDim    = std::max({BlockDimensions_...});
    static constexpr bool   SingleBlockDim = (MinBlockDim == MaxBlockDim);
    static constexpr std::array<size_t, NumBlockTypes> BlockDimensions{{BlockDimensions_...}};
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    struct Block { size_t start, size; };
    Block blockInfo(size_t blockIndex) const {
        if constexpr (SingleBlockDim) { return Block{FirstBlockDim * blockIndex, FirstBlockDim}; }
        else {
            for (size_t i = 0; i < NumBlockTypes; ++i)
                if (blockIndex < m_typeBlockOffsets[i + 1]) return Block{m_typeVarOffsets[i] + (blockIndex - m_typeBlockOffsets[i]) * BlockDimensions[i], BlockDimensions[i]};
            return Block{NONE, NONE};
        }
    }

    // Query the block size of a given variable.
    size_t blockSize(size_t block) const {
        if constexpr (SingleBlockDim) { return FirstBlockDim; }
        else {
            for (size_t i = 0; i < NumBlockTypes; ++i)
                if (block < m_typeBlockOffsets[i + 1]) return BlockDimensions[i];
            return NONE;
        }
    }

    size_t offsetForBlock(size_t block) const {
        if constexpr (SingleBlockDim) { return FirstBlockDim * block; }
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
        : m_numBlocksPerType{{args...}}
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

    size_t offsetForType(size_t type_id) const { return m_typeVarOffsets[type_id]; }
    size_t numVarsOfType(size_t type_id) const { return m_typeVarOffsets[type_id + 1] - m_typeVarOffsets[type_id]; }

    size_t numVars() const { return m_numScalarVars; }
    size_t numBlocks() const { return m_numBlocks; }

    template<class Derived> auto variablesOfType(      Eigen::MatrixBase<Derived> &x, size_t type_id) const { return x.segment(offsetForType(type_id), numVarsOfType(type_id)); }
    template<class Derived> auto variablesOfType(const Eigen::MatrixBase<Derived> &x, size_t type_id) const { return x.segment(offsetForType(type_id), numVarsOfType(type_id)); }

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
    using VarStructure = OptimizationVarStructure<BlockDimensions_...>;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;

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

    const VarStructure &vars() const { return m_vars; }

    template<class FEMMesh_>
    CSCMat blockSparsityPatternForMesh(const FEMMesh_ &m) const {
        return blockSparsityPattern(m.numElements(),
                [&](size_t ei) {
                    std::array<size_t, FEMMesh_::NumNodesPerElement> blockVarsForElement;
                    auto e = m.element(ei);
                    for (const auto n_b : e.nodes()) { blockVarsForElement[n_b.localIndex()] = n_b.index(); }
                    return blockVarsForElement;
                });
    }

    template<class ElemBlockVarsForElement>
    CSCMat blockSparsityPattern(size_t numElems, const ElemBlockVarsForElement &blockVarsForElement) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("blockSparsityPattern");

        const size_t numBlockVars = m_vars.numBlocks();
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
                for (decltype(bvars.size()) v_b_i = 0; v_b_i < bvars.size(); ++v_b_i) {
                    auto v_b = bvars[v_b_i];
                    for (decltype(bvars.size()) v_a_i = 0; v_a_i < bvars.size(); ++v_a_i)
                        if (bvars[v_a_i] <= v_b) ++sizes[v_b]; // sizes[v_b] += (v_a <= v_b);
                }
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
                for (decltype(bvars.size()) v_b_i = 0; v_b_i < bvars.size(); ++v_b_i) {
                    auto v_b = bvars[v_b_i];
                    m_lockVar(v_b);
                    size_t back = bucketBack[v_b];
                    for (decltype(bvars.size()) v_a_i = 0; v_a_i < bvars.size(); ++v_a_i) {
                        auto v_a = bvars[v_a_i];
                        if (v_a <= v_b) columnBuckets[back++] = v_a;
                    }
                    bucketBack[v_b] = back;
                    m_unlockVar(v_b);
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

        result.nz = newNNZ;
        // result.Ax.resize(newNNZ); // <--- Intentionally leave empty since we generally don't need to store data in the block pattern.

        return result;
    }

    static constexpr size_t NEW_ENTRIES = std::numeric_limits<size_t>::max();
    // Efficiently detect changes in the block sparsity pattern.
    // Returns `NEW_ENTRIES` if even a single new entry becomes nonzero;
    // otherwise returns the number of entries that have disappeared from the
    // sparsity pattern (if any).
    template<class ElemBlockVarsForElement>
    size_t detectChangedEntries(const CSCMat &oldBlockHsp, size_t numElems, const ElemBlockVarsForElement &blockVarsForElement) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("detectChangedEntries");
        if (numElems == 0) return oldBlockHsp.nz;
        if ((oldBlockHsp.nz == 0) && (numElems != 0)) return NEW_ENTRIES;

        bool hasNewEntries = false;
        m_sparsityChangeDetectionScratch.assign(oldBlockHsp.nz, false);

        parallel_for_range(numElems, [&](size_t ei) {
            if (hasNewEntries) return; // early exit
            auto bvars = blockVarsForElement(ei);
            size_t nv = bvars.size();
            for (size_t vj = 0; vj < nv; ++vj) {
                auto var_j = bvars[vj];
                for (size_t vi = 0; vi < nv; ++vi) {
                    auto var_i = bvars[vi];
                    if (var_i > var_j) continue;
                    SuiteSparseMatrix::index_type loc = oldBlockHsp.findEntry<true>(var_i, var_j);
                    if (loc == SuiteSparseMatrix::INDEX_NONE) { hasNewEntries = true; return; }
                    else m_sparsityChangeDetectionScratch[loc] = true;
                }
            }
        }, /* grain_size = */ 128, /* parallelism_threshold = */ 1000);
        if (hasNewEntries) return NEW_ENTRIES;

        // Check for missing entries
        size_t numDisappeared = 0;
        for (char i : m_sparsityChangeDetectionScratch) numDisappeared += (i == 0);
        return numDisappeared;
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

    ////////////////////////////////////////////////////////////////////////////
    // Scalar Hessian assembly.
    ////////////////////////////////////////////////////////////////////////////
    template <class PEH, class EVars>
    struct HessianElementAssemblyData {
        auto block(size_t a, size_t b, size_t bsa, size_t bsb) const { return getBlock(H_e, a, b, bsa, bsb).eval(); }
        PEH H_e;
        EVars evars;
    };

    // Fully customizable Hessian assembly:
    // For each element ei in 0..ne, obtain a data object from `edataGetter`
    // whose `elementVars` method reports the global block variables corresponding to the element
    // and whose `block` method provides accesses to blocks of the per-element Hessian.
    // Note that this `block` method enables additional computation to be
    // performed at assembly time, e.g., to implement chain rule expressions.
    template<class SPMat, class ElementAssemblyDataGetter>
    void assembleHessian(SPMat &H, size_t ne, const ElementAssemblyDataGetter &edataGetter) const {
        get_hessian_assembly_arena().execute([&H, &edataGetter, ne, this]() {
            parallel_for_range(ne, [&H, &edataGetter, this](size_t ei) {
                auto edata = edataGetter(ei);
                m_assembleHessianContrib(H, [&edata](size_t a, size_t b, size_t bsa, size_t bsb) {
                    return edata.block(a, b, bsa, bsb);
                }, edata.evars);
            }, 1, 32);
        });
    }

    // For each ei in 0..ne, evaluate per-element Hessian H_e = eval(ei) and
    // then assemble it into H[element(ei), element(ei)] by accessing its blocks
    // with `He_block(H_e, lni_a, lni_b, bsa, bsb)`.
    template<class SPMat, class PEHEval, class HEBlock, class ElementGetter>
    void assembleHessian(SPMat &H, size_t ne, const PEHEval &eval_He, const HEBlock &He_block, const ElementGetter &element) const {
        using PEH = decltype(eval_He(0));
        using EVars = decltype(element(0));
        using HEAD = HessianElementAssemblyData<PEH, EVars>;
        assembleHessian(H, ne, [&](size_t ei) { return HEAD{eval_He(ei), element(ei)}; });
    }

    // Assemble the per-element Hessian `eval_He(ei)` for element ei in 0..ne.
    // The element's global block variable indices are obtained by calling
    // `element(ei)`, which should return an array of variable indices.
    template<class SPMat, class PEHEval, class ElementGetter>
    void assembleHessian(SPMat &H, size_t ne, const PEHEval &eval_He, const ElementGetter &element) const {
        assembleHessian(H, ne, eval_He, [](const auto &H_e, size_t a, size_t b, size_t bsa, size_t bsb) {
            return getBlock(H_e, a, b, bsa, bsb).eval();
        }, element);
    }

    // Convenience method for the typical case of assembling a per-element Hessian using
    // using nodal variables of a FEMMesh.
    template<class SPMat, class Mesh, class PEHEval>
    void assembleHessian(SPMat &H, const Mesh &m, const PEHEval &eval_He) const {
        assembleHessian(H, m.numElements(), eval_He, [&m](size_t ei) { return m.elementNodeIndices(ei); });
    }

    ////////////////////////////////////////////////////////////////////////////
    // Block Hessian assembly.
    ////////////////////////////////////////////////////////////////////////////
    template<class PEH>
    static auto getBlock(const PEH &H_e, size_t a, size_t b, size_t bsa = VarStructure::MaxBlockDim, size_t bsb = VarStructure::MaxBlockDim) {
        static constexpr size_t N = VarStructure::MaxBlockDim;
        if constexpr (VarStructure::SingleBlockDim) {
            UNUSED(bsa); UNUSED(bsb);
            return H_e.template block<N, N>(a, b);
        }
        else {
            return H_e.block(a, b, bsa, bsb);
        }
    }

    template<class ElemBlockVars>
    static auto argsort(const ElemBlockVars &blockVars) {
        static constexpr size_t nbv = std::tuple_size_v<ElemBlockVars>;
        std::array<size_t, nbv> order;
        for (size_t i = 0; i < nbv; ++i) { order[i] = i; }
        StaticTimSort<nbv> timBoseNelsonSort;
        timBoseNelsonSort(order, [&blockVars](size_t a, size_t b) { return blockVars[a] < blockVars[b]; });
        return order;
    }

    template<bool InParallel = true, class SPMat, class Mesh, class PEH, class ElemBlockVars>
    void assembleHessianBlockContrib(SPMat &H, const Mesh &m, const PEH &H_e, const ElemBlockVars &blockVars) const {
        static_assert(SingleBlockDim, "Only implemented for SingleBlockDim case");
        static constexpr size_t N = VarStructure::FirstBlockDim;
        static constexpr size_t nbv = std::tuple_size_v<ElemBlockVars>;

        auto order = argsort(blockVars);

        SuiteSparse_long *Ap = H.Ap.data();
        SuiteSparse_long *Ai = H.Ai.data();

        for (size_t lbj_i = 0; lbj_i < nbv; ++lbj_i) {
            size_t lbj = order[lbj_i];
            auto bj = blockVars[lbj];

            SuiteSparse_long head = Ap[bj];
            SuiteSparse_long colEnd = Ap[bj + 1];

            if constexpr (InParallel) m_lockVar(bj);
#if 1
            // Insert the blocks for column `bj`
            for (size_t lbi_i = 0; lbi_i < lbj_i; ++lbi_i) {
                size_t lbi = order[lbi_i];
                auto bi = blockVars[lbi];
                if (lbi < lbj) H.addNZ(bi, bj, getBlock(H_e, N * lbi, N * lbj));
                else           H.addNZ(bi, bj, getBlock(H_e, N * lbj, N * lbi).transpose());
            }
#else
            // Merge in the blocks for column `bj`
            for (size_t lbi_i = 0; lbi_i < lbj_i; ++lbi_i) {
                size_t lbi = order[lbi_i];
                SuiteSparse_long bi = blockVars[lbi];
                head = binary_search(bi, Ai, head, colEnd);
                // while (Ai[head] < bi) ++head;
                if (lbi < lbj) H.Ax[head] += getBlock(N * lbi, N * lbj);
                else           H.Ax[head] += getBlock(N * lbj, N * lbi).transpose();
            }
#endif
            H.Ax[colEnd - 1].template triangularView<Eigen::Upper>() += getBlock(H_e, N * lbj, N * lbj); // Add diagonal entry.
            if constexpr (InParallel) m_unlockVar(bj);
        }
    }

    template<class SPMat, class Mesh, class PEHEval>
    void assembleBlockHessian(SPMat &H, const Mesh &m, const PEHEval &eval_He) const {
        static_assert(SingleBlockDim, "Only implemented for SingleBlockDim case");
        static constexpr size_t N = VarStructure::FirstBlockDim;
        if (get_max_num_tbb_threads() == 1) {
            const size_t ne = m.numElements();
            for (size_t ei = 0; ei < ne; ++ei)
                assembleHessianBlockContrib</* InParallel = */ false>(H, m, eval_He(ei), m.elementNodeIndices(ei));
        }
        else {
            get_hessian_assembly_arena().execute([&H, &eval_He, &m, this]() {
                parallel_for_range(m.numElements(), [&H, &eval_He, &m, this](size_t ei) {
                    assembleHessianBlockContrib(H, m, eval_He(ei), m.elementNodeIndices(ei));
                }, 1, 32);
            });
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Block-accelerated scalar Hessian assembly.
    // (Construct the scalar Hessian but use a block sparsity pattern for
    // acceleration).
    ////////////////////////////////////////////////////////////////////////////
    template<bool InParallel = true, class SPMatScalar, class SPMatBlock, class Mesh, class PEH, class ElemBlockVars>
    void assembleHessianContribBlockAccelerated(SPMatScalar &H, const SPMatBlock &blockH, const Mesh &m, const PEH &H_e, const ElemBlockVars &blockVars) const {
        static_assert(SingleBlockDim, "Only implemented for SingleBlockDim case");
        static constexpr size_t N = VarStructure::MaxBlockDim;
        static constexpr size_t nbv = std::tuple_size_v<ElemBlockVars>;

        auto order = argsort(blockVars);

        typename SPMatScalar::value_type *Ax = H.Ax.data();
        using StripMap = typename SPMatScalar::template SizedDataMap<N>;

        for (size_t lbj_i = 0; lbj_i < nbv; ++lbj_i) {
            size_t lbj = order[lbj_i];
            auto bj = blockVars[lbj];

            SuiteSparse_long colStart  = H.Ap[N * bj];
            SuiteSparse_long colStride = H.col_nnz(N * bj);

            if constexpr (InParallel) m_lockVar(bj);
            for (size_t lbi_i = 0; lbi_i < lbj_i; ++lbi_i) {
                size_t lbi = order[lbi_i];
                auto bi = blockVars[lbi];

                auto addBlock = [&](auto block) {
                    // Convert block matrix location into scalar matrix location.
                    SuiteSparse_long loc = colStart + N * (blockH.findEntry(bi, bj) - blockH.Ap[bj]);
                    for (size_t c = 0; c < N; ++c) {
                        StripMap(Ax + loc) += block.col(c);
                        loc += colStride + c; // each subsequent column has an extra entry...
                    }
                };

                if (lbi < lbj) addBlock(getBlock(H_e, N * lbi, N * lbj));
                else           addBlock(getBlock(H_e, N * lbj, N * lbi).transpose());
            }

            // Add (upper triangle of) diagonal block
            SuiteSparse_long loc = H.Ap[N * bj + 1] - 1;
            auto block = getBlock(H_e, N * lbj, N * lbj);
            for (size_t c = 0; c < N; ++c) {
                typename SPMatScalar::DataMap(Ax + loc, c + 1) += block.col(c).topRows(c + 1);
                loc += colStride + c; // each subsequent column has an extra entry...
            }

            if constexpr (InParallel) m_unlockVar(bj);
        }
    }

    template<class SPMatScalar, class SPMatBlock, class Mesh, class PEHEval>
    void assembleHessianBlockAccelerated(SPMatScalar &H, const SPMatBlock &blockH, const Mesh &m, const PEHEval &eval_He) const {
        static_assert(SingleBlockDim, "Only implemented for SingleBlockDim case");
        static constexpr size_t N = VarStructure::FirstBlockDim;
        if (get_max_num_tbb_threads() == 1) {
            const size_t ne = m.numElements();
            for (size_t ei = 0; ei < ne; ++ei)
                assembleHessianContribBlockAccelerated</* InParallel = */ false>(H, blockH, m, eval_He(ei), m.elementNodeIndices(ei));
        }
        else {
            get_hessian_assembly_arena().execute([&H, &blockH, &eval_He, &m, this]() {
                parallel_for_range(m.numElements(), [&H, &blockH, &eval_He, &m, this](size_t ei) {
                    assembleHessianContribBlockAccelerated(H, blockH, m, eval_He(ei), m.elementNodeIndices(ei));
                }, 1, 32);
            });
        }
    }

    // *Accumulate* to `g` the per-element gradient `eval_ge(ei)` for element ei in 0..ne.
    // The element's global block variable indices are obtained by calling `element(ei)`,
    // which should return an array of variable indices.
    template<class Result, class PEGEval, class ElementGetter>
    void assembleGradient(Result &g, size_t ne, const PEGEval &eval_ge, const ElementGetter &element) const {
        auto accumulate_per_element_contrib = [&element, &eval_ge, this](size_t ei, Result &g_out) {
            const auto blockVars = element(ei);
            const auto ge = eval_ge(ei);

            if constexpr (SingleBlockDim) {
                UNUSED(this); // Work around spurious unused warning in clang...
                for (decltype(blockVars.size()) lbi = 0; lbi < blockVars.size(); ++lbi) {
                    g_out .template segment<VarStructure::FirstBlockDim>(VarStructure::FirstBlockDim * blockVars[lbi]) +=
                        ge.template segment<VarStructure::FirstBlockDim>(VarStructure::FirstBlockDim * lbi);
                }
            }
            else {
                size_t lvar = 0;
                for (decltype(blockVars.size()) lbi = 0; lbi < blockVars.size(); ++lbi) {
                    auto bi = blockVars[lbi];
                    auto [gvar, bs] = m_vars.blockInfo(bi);
                    g_out.segment(gvar, bs) += ge.segment(lvar, bs);
                    lvar += bs;
                }
            }
        };
        assemble_parallel(accumulate_per_element_contrib, g, ne);
    }

    template<class Result, class Mesh, class PEGEval>
    void assembleGradient(Result &g, const Mesh &m, const PEGEval &eval_ge) const {
        return assembleGradient(g, m.numElements(), eval_ge, [&m](size_t ei) { return m.elementNodeIndices(ei); });
    }

    using VXd = Eigen::VectorXd;
    mutable VXd m_pregatherGradient;
    using NodeLocalNodeAdjacencyMatrix = CSCMatrix<SuiteSparse_long, char>;
    mutable std::shared_ptr<NodeLocalNodeAdjacencyMatrix> m_localNodesForNode;
    template<class Result, class Mesh, class PEGEval>
    void assembleGradientScatterGather(Result &g, const Mesh &m, const PEGEval &eval_ge) const {
        static_assert(VarStructure::SingleBlockDim, "Only SingleBlockDim case is implemented");
        constexpr size_t N = VarStructure::FirstBlockDim;
        constexpr size_t numElemLocalVars = N * Mesh::NumNodesPerElement;

        // Cache vertex => (element, local index) map in a CSCMatrix<Char>
        if (!m_localNodesForNode) {
            TripletMatrix<Triplet<char>> localNodesForNodeTrip(numElemLocalVars * m.numElements(), m.numNodes());
            for (auto e : m.elements())
                for (auto n : e.nodes())
                    localNodesForNodeTrip.addNZ(numElemLocalVars * e.index() + N * n.localIndex(), n.index(), 1);

            m_localNodesForNode = std::make_unique<NodeLocalNodeAdjacencyMatrix>(localNodesForNodeTrip);
        }
        m_pregatherGradient.resize(m.numElements() * numElemLocalVars);

        BENCHMARK_SCOPED_TIMER_SECTION timer("assembleGradientScatterGather");
        parallel_for_range(m.numElements(), [this, &eval_ge](size_t ei) {
            m_pregatherGradient.template segment<numElemLocalVars>(ei * numElemLocalVars) = eval_ge(ei);
        }, 32, 100);

        if (size_t(g.size()) != N * m.numNodes()) throw std::runtime_error("Unexpected g size");
        SuiteSparse_long *Ai = m_localNodesForNode->Ai.data();
        SuiteSparse_long *Ap = m_localNodesForNode->Ap.data();
        parallel_for_range(m.numNodes(), [&g, Ai, Ap, this](size_t ni) {
                SuiteSparse_long *idxPtr = Ai + Ap[ni];
                SuiteSparse_long *colEnd = Ai + Ap[ni + 1];
                VecN_T<Real, N> g_n = m_pregatherGradient.template segment<N>(*idxPtr);
                for (++idxPtr; idxPtr < colEnd; ++idxPtr)
                    g_n += m_pregatherGradient.template segment<N>(*idxPtr);
                g.template segment<N>(N * ni) = g_n;
            }, 100, 100);
    }

private:
    template<class SPMat, class HeBlock, class ElemBlockVars>
    void m_assembleHessianContrib(SPMat &H, const HeBlock &He_block, const ElemBlockVars &blockVars) const {
        size_t lvar_j = 0;

        for (decltype(blockVars.size()) lbj = 0; lbj < blockVars.size(); ++lbj) {
            auto bj = blockVars[lbj];
            auto [gvar_j, bsj] = m_vars.blockInfo(bj);
            size_t lvar_i = 0;
            m_lockVar(bj);
            for (decltype(blockVars.size()) lbi = 0; lbi < blockVars.size(); ++lbi) {
                auto bi = blockVars[lbi];
                auto [gvar_i, bsi] = m_vars.blockInfo(bi);
                bool localUpperTri = lbi < lbj;

                decltype(He_block(lvar_i, lvar_j, bsi, bsj)) block;
                if (localUpperTri) block = He_block(lvar_i, lvar_j, bsi, bsj);
                else               block = He_block(lvar_j, lvar_i, bsj, bsi).transpose();

                if (gvar_i < gvar_j) {
                    index_type idx = H.findEntry(gvar_i, gvar_j);
                    for (size_t c = 0; c < bsj; ++c) {
                        if constexpr (SingleBlockDim) typename SPMat::template SizedDataMap<VarStructure::FirstBlockDim>(H.Ax.data() + idx) += block.col(c);
                        else                          typename SPMat::DataMap(H.Ax.data() + idx, bsi) += block.col(c);

                        // Advance to the start of the block in the next columnn
                        // (assuming the next column has an identical sparsity
                        // pattern in rows 0...gvar_i)
                        idx += H.col_nnz(gvar_j + c);
                    }
                }
                else if (gvar_i == gvar_j) {
                    index_type idx = H.findDiagEntry(gvar_i); // Top of strip to add
                    for (size_t c = 0; c < bsj; ++c) {
                        typename SPMat::DataMap(H.Ax.data() + idx, c + 1) += block.col(c).topRows(c + 1);
                        idx += H.col_nnz(gvar_j + c);
                    }
                }

                lvar_i += bsi;
            }
            m_unlockVar(bj);
            lvar_j += bsj;
        }
    }

    void   m_lockVar(size_t var) const { while ((*m_varLocks)[var].exchange(true, std::memory_order_acquire)); }
    void m_unlockVar(size_t var) const {        (*m_varLocks)[var].store  (false, std::memory_order_release);  }

    mutable std::vector<char> m_sparsityChangeDetectionScratch;
    mutable std::unique_ptr<std::vector<std::atomic<bool>>> m_varLocks;
    VarStructure m_vars;
};

#endif /* end of include guard: SYSTEMASSEMBLER_HH */
