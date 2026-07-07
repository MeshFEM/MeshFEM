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
#include <atomic>
#include <tuple>
#include <functional>
#include <MeshFEMSparse/SparseMatrices.hh>
#include <MeshFEMSparse/Utilities/argsort.hh>

#include <MeshFEMSparse/VarStructure.hh>
#include <MeshFEMSparse/BlockCSCHessian.hh>
#include <MeshFEMSparse/BorderedSparseHessian.hh>

namespace MeshFEM {

struct VarLocks {
    void init(size_t numLocks) {
        if (m_varLocks) return;
        m_varLocks = std::make_unique<std::vector<std::atomic<bool>>>(numLocks);
        for (size_t i = 0; i < numLocks; ++i)
            atomic_init(&(*m_varLocks)[i], false);
    }

    void   lock(size_t var) { while ((*m_varLocks)[var].exchange(true, std::memory_order_acquire)); }
    void unlock(size_t var) {        (*m_varLocks)[var].store  (false, std::memory_order_release);  }
private:
    std::unique_ptr<std::vector<std::atomic<bool>>> m_varLocks;
};

// Assemble into a scalar-valued `Ax` array but use the block sparsity
// pattern in `blockH` for acceleration. This is the correct assembly routine
// to use for the `BlockCSCHessian` format.
template<bool UseBlockMergeAlgorithm> // Whether to use our proposed sort + merge algorithm (as opposed to a binary search-based approach)
struct ElementHessianContribAssembler;

template<>
struct ElementHessianContribAssembler<true> {
    template<bool InParallel = true, class Real_, class SPMatBlock, class HeBlock, class ElemBlockVars, class VarStructure>
    static void run(Real_ *Ax, const SPMatBlock &blockH, const HeBlock &He_block, const ElemBlockVars &blockVars, VarStructure &vars, VarLocks &varLocks) {
        PerElementBlockOffsetCalculation<VarStructure, ElemBlockVars> blockOffsetCalc(vars, blockVars);

        auto order = argsort(blockVars);

        for (size_t lbj_i = 0; lbj_i < blockVars.size(); ++lbj_i) {
            size_t lbj = order[lbj_i];
            auto bj = blockVars[lbj];
            const size_t lbo_j = blockOffsetCalc.offset(lbj);

            auto colScanner = blockH.columnScanner(bj);
            const size_t bs_j = colScanner.colBlockSize();
            if constexpr (InParallel) varLocks.lock(bj);

            for (size_t lbi_i = 0; lbi_i < lbj_i; ++lbi_i) {
                size_t lbi = order[lbi_i];
                auto bi = blockVars[lbi];
                const size_t lbo_i = blockOffsetCalc.offset(lbi);
                const size_t bs_i = blockOffsetCalc.blockSize(lbi);

                if (lbi < lbj) colScanner.advanceToAndAddBlock(Ax, bi, He_block(lbo_i, lbo_j, bs_i, bs_j));
                else           colScanner.advanceToAndAddBlock(Ax, bi, He_block(lbo_j, lbo_i, bs_i, bs_j).transpose());
            }

            // Add (upper triangle of) diagonal block
            auto loc = colScanner.diagBlockScalarLoc();
            auto block = He_block(lbo_j, lbo_j, bs_j, bs_j);
            for (size_t c = 0; c < bs_j; ++c) {
                Eigen::Map<Eigen::Matrix<Real_, Eigen::Dynamic, 1>>(Ax + loc, c + 1) += block.col(c).topRows(c + 1);
                loc += colScanner.diagBlockColStride(c);
            }

            if constexpr (InParallel) varLocks.unlock(bj);
        }
    }
};

// Same as above, but using a traditional binary-search-based approach
template<>
struct ElementHessianContribAssembler<false> {
    template<bool InParallel = true, class Real_, class SPMatBlock, class HeBlock, class ElemBlockVars, class VarStructure>
    static void run(Real_ *Ax, const SPMatBlock &blockH, const HeBlock &He_block, const ElemBlockVars &blockVars, VarStructure &vars, VarLocks &varLocks) {
        PerElementBlockOffsetCalculation<VarStructure, ElemBlockVars> blockOffsetCalc(vars, blockVars);

        for (size_t lbj = 0; lbj < blockVars.size(); ++lbj) {
            auto bj = blockVars[lbj];
            const size_t lbo_j = blockOffsetCalc.offset(lbj);

            auto colScanner = blockH.columnScanner(bj);
            const size_t bs_j = colScanner.colBlockSize();
            if constexpr (InParallel) varLocks.lock(bj);

            for (size_t lbi = 0; lbi < blockVars.size(); ++lbi) {
                auto bi = blockVars[lbi];
                if (bi >= bj) continue; // Skip lower triangle

                const size_t lbo_i = blockOffsetCalc.offset(lbi);
                const size_t bs_i  = blockOffsetCalc.blockSize(lbi);

                if (lbi < lbj) colScanner.addBlock(Ax, bi, He_block(lbo_i, lbo_j, bs_i, bs_j));
                else           colScanner.addBlock(Ax, bi, He_block(lbo_j, lbo_i, bs_i, bs_j).transpose());
            }

            // Add (upper triangle of) diagonal block
            auto loc = colScanner.diagBlockScalarLoc();
            auto block = He_block(lbo_j, lbo_j, bs_j, bs_j);
            for (size_t c = 0; c < bs_j; ++c) {
                Eigen::Map<Eigen::Matrix<Real_, Eigen::Dynamic, 1>>(Ax + loc, c + 1) += block.col(c).topRows(c + 1);
                loc += colScanner.diagBlockColStride(c);
            }

            if constexpr (InParallel) varLocks.unlock(bj);
        }
    }
};

struct MESHFEM_EXPORT SystemAssemblerBase {
    using index_type = SuiteSparse_long;
    using CSCMat = CSCMatrix<index_type, double>;

    virtual ~SystemAssemblerBase() = default;

    virtual size_t      numVars() const = 0;
    virtual size_t numBlockVars() const = 0;

    // Construct a block sparsity pattern consistent this assembler's variable
    // structure from dynamically-accessed element stencils (obtained by calling
    // `elementGetter(ei)`) containing block-variable indices, where the element
    // variable blocks vb_i are all of a uniform size `blockSize`.
    //
    // Each of these vb_i must fit entirely within a single block b_i of the
    // variable structure, otherwise an exception will be thrown. The block
    // will effectively be expanded to the size of b_i in the variable
    // structure when entering into the sparsity pattern. For example,
    // when `blockSize == 1`, the caller is asking insert a single scalar
    // entry into the sparsity pattern, but the entire block containing
    // that scalar will be marked nonzero.
    //
    // This method should be used only as a last resort (e.g., for an objective
    // term that does not know the problem block structure), as it is less
    // efficient and typesafe than the `blockSparsityPattern` method templates of
    // the derived classes.
    using DynamicElementGetter = std::function<std::vector<size_t>(size_t)>;
    std::unique_ptr<BlockCSCHessianBase> blockSparsityPattern(size_t numElements, size_t blockSize, const DynamicElementGetter &elementGetter) const {
        return m_blockSparsityPatternDynamicImpl(numElements, blockSize, elementGetter);
    }

    // TODO: dynamic-sized version of Hessian assembly.
    struct DynamicHessianElementAssemblyData {
        virtual Eigen::Map<const Eigen::MatrixXd> block(size_t a, size_t b) const = 0;
        std::vector<size_t> blockVars;
        virtual ~DynamicHessianElementAssemblyData() = default;
    };

private:
    virtual std::unique_ptr<BlockCSCHessianBase> m_blockSparsityPatternDynamicImpl(size_t numElements, size_t blockSize, const DynamicElementGetter &elementGetter) const = 0;
};

template<size_t... BlockDimensions_>
struct MESHFEM_EXPORT SystemAssembler : public SystemAssemblerBase {
    using index_type = SuiteSparse_long;
    using CSCMat = CSCMatrix<index_type, double>;
    using VarStructure = OptimizationVarStructure<BlockDimensions_...>;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;
    static constexpr bool UseBlockMergeAlgorithmDefault = true;

    // Construct given a number of variables for each type.
    template <typename... Args>
    SystemAssembler(Args... args)
        : m_vars(args...)
    {
        static_assert(sizeof...(Args) > 0, "Variables must be initialized!");
    }

    virtual ~SystemAssembler() = default;

    const VarStructure &varStructure() const { return m_vars; }
    size_t      numVars() const override { return varStructure().numVars(); }
    size_t numBlockVars() const override { return varStructure().numBlocks(); }

    size_t blockSizeOfType(size_t type) const { return VarStructure::BlockDimensions[type]; }

    using BCSCMat = BlockCSCHessian<VarStructure>;

    std::unique_ptr<BCSCMat> emptyBlockSparsityPattern() const { return BCSCMat::construct(m_vars); }

    ////////////////////////////////////////////////////////////////////////////
    // BlockCSCHessian sparsity pattern generation
    ////////////////////////////////////////////////////////////////////////////
    template<class ElemBlockVarsForElement>
    std::unique_ptr<BCSCMat> blockSparsityPattern(size_t numElems, const ElemBlockVarsForElement &blockVarsForElement) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("blockSparsityPattern");
        const bool parallel = (get_max_num_tbb_threads() > 1) && (numElems >= 1024);

        const size_t numBlockVars = m_vars.numBlocks();
        auto result = emptyBlockSparsityPattern();
        auto &Ap = result->Ap;
        auto &Ai = result->Ai;
        const size_t n = numBlockVars;

        std::vector<size_t> bucketStart;
        {
            // BENCHMARK_SCOPED_TIMER_SECTION timer1("calc size");
            bucketStart.resize(n + 1);
            size_t *sizes = bucketStart.data() + 1;
            for (size_t ei = 0; ei < numElems; ++ei) {
                auto bvars = blockVarsForElement(ei);
#if 1
                static_sort_with_fallback(bvars);
                for (decltype(bvars.size()) i = 0; i < bvars.size(); ++i)
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
            // BENCHMARK_SCOPED_TIMER_SECTION timer1("cumsum");
            // Next, compute bucketStart[2:] = cumsum(bucketStart[1:])
            for (size_t j = 1; j <= n; ++j) {
                size_t colsize_j = bucketStart[j];
                bucketStart[j] = origNNZ;
                origNNZ += colsize_j;
            }
        }

        using RowIndex = uint32_t; // Using a narrower integer type substantially reduces memory i/o
        Eigen::Matrix<RowIndex, Eigen::Dynamic, 1> columnBuckets;
        {
            // BENCHMARK_SCOPED_TIMER_SECTION timer1("fill adjacency");

            columnBuckets.resize(origNNZ);
            // Fill the index buckets; note incrementing the offsets in
            // bucketStart[1:] by the size of each bucket converts these into the
            // end offsets.
            size_t *bucketBack = bucketStart.data() + 1;

#define VARLOCK_PARALLELIZATION 0 // Whether to use simple column-lock based parallelism or atomically resized buckets.

            if (parallel) {
#if VARLOCK_PARALLELIZATION
                m_varLocks.init(n);
                parallel_for_range(numElems, [&](size_t ei) {
                    const auto &bvars = blockVarsForElement(ei);
                    for (decltype(bvars.size()) v_b_i = 0; v_b_i < bvars.size(); ++v_b_i) {
                        auto v_b = bvars[v_b_i];
                        m_varLocks.lock(v_b);
                        size_t back = bucketBack[v_b];
                        for (decltype(bvars.size()) v_a_i = 0; v_a_i < bvars.size(); ++v_a_i) {
                            auto v_a = bvars[v_a_i];
                            if (v_a <= v_b) columnBuckets[back++] = v_a;
                        }
                        bucketBack[v_b] = back;
                        m_varLocks.unlock(v_b);
                    }
                });
#else
                // std::vector<std::atomic<int>> bucketSizeAtomic(n);
                parallel_for_range(numElems, [&](size_t ei) {
                    auto bvars = blockVarsForElement(ei);
                    static_sort_with_fallback(bvars);
                    for (decltype(bvars.size()) v_b_i = 0; v_b_i < bvars.size(); ++v_b_i) {
                        auto v_b = bvars[v_b_i];
                        // The standards-compliant way of doing the following would be std::atomic_ref, which requires C++20.
                        // size_t back = __atomic_add_fetch(bucketBack + v_b, v_b_i + 1, __ATOMIC_RELAXED); // Works on Clang/GCC, no UB (assuming proper alignment)
                        size_t back = reinterpret_cast<std::atomic<size_t> *>(bucketBack + v_b)->fetch_add(v_b_i + 1, std::memory_order_relaxed); // Technically UB but works
                        // size_t back = bucketBack[v_b] + bucketSizeAtomic[v_b].fetch_add(v_b_i + 1, std::memory_order_relaxed);
                        for (decltype(bvars.size()) v_a_i = 0; v_a_i <= v_b_i; ++v_a_i)
                            columnBuckets[back++] = bvars[v_a_i];
                    }
                });
                // for (size_t j = 0; j < n; ++j)
                //     bucketBack[j] += bucketSizeAtomic[j].load(std::memory_order_relaxed);
#endif
            }
            else {
                for (size_t ei = 0; ei < numElems; ++ei) {
#if 0
                    const auto &bvars = blockVarsForElement(ei);
                    for (decltype(bvars.size()) v_b_i = 0; v_b_i < bvars.size(); ++v_b_i) {
                        auto v_b = bvars[v_b_i];
                        size_t back = bucketBack[v_b];
                        for (decltype(bvars.size()) v_a_i = 0; v_a_i < bvars.size(); ++v_a_i) {
                            auto v_a = bvars[v_a_i];
                            if (v_a <= v_b) columnBuckets[back++] = v_a;
                        }
                        bucketBack[v_b] = back;
                    }
#else
                    auto bvars = blockVarsForElement(ei);
                    static_sort_with_fallback(bvars);
                    for (decltype(bvars.size()) v_b_i = 0; v_b_i < bvars.size(); ++v_b_i) {
                        auto v_b = bvars[v_b_i];
                        size_t back = bucketBack[v_b];
                        for (decltype(bvars.size()) v_a_i = 0; v_a_i <= v_b_i; ++v_a_i)
                            columnBuckets[back++] = bvars[v_a_i];
                        bucketBack[v_b] = back;
                    }
#endif
                }
            }
        }

        // BENCHMARK_SCOPED_TIMER_SECTION timer1("Generate CSCMat");

        // BENCHMARK_START_TIMER_SECTION("Ap resize");
        Ap.resize(n + 1);
        // BENCHMARK_STOP_TIMER_SECTION("Ap resize");

        // BENCHMARK_START_TIMER_SECTION("Sort and deduplicate");
#if 1
        {
            // Deduplicate **first** and then sort; this is faster than sorting
            // the much larger duplicate-filled lists.
            // We use a thread-local "patternFlags" array to mark which row
            // indices have already been seen in the current column.
            tbb::enumerable_thread_specific<Eigen::Matrix<size_t, Eigen::Dynamic, 1>> threadPatternFlags;

            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                auto &patternFlags = threadPatternFlags.local();
                if (size_t(patternFlags.size()) != n)
                    patternFlags.setConstant(n, -1);
                for (size_t j = r.begin(); j != r.end(); ++j) {
                    RowIndex *start = columnBuckets.data() + bucketStart[j];
                    RowIndex *end   = columnBuckets.data() + bucketStart[j + 1];
                    RowIndex *back = start;
                    for (auto curr = start; curr < end; ++curr) {
                        size_t i = *curr;
                        if (patternFlags[i] != j) {
                            patternFlags[i] = j;
                            *back = i;
                            ++back;
                        }
                    }
                    std::sort(start, back); // Sort the deduplicated row indices.
                    Ap[j] = std::distance(start, back); // Write the deduplicated bucket size.
                }
            });
        }
#else
        // Sort each bucket in parallel and deduplicate.
        parallel_for_range(n, [&](size_t j) {
            auto start = columnBuckets.data() + bucketStart[j];
            auto end   = columnBuckets.data() + bucketStart[j + 1];
            std::sort(start, end);
            end = std::unique(start, end);
            Ap[j] = std::distance(start, end); // Write deduplicated bucket size
        });
#endif

        // BENCHMARK_STOP_TIMER_SECTION("Sort and deduplicate");

        // BENCHMARK_START_TIMER_SECTION("cumsum");
        // Calculate column pointer array using cumulative sum.
        size_t newNNZ = 0;
        for (size_t j = 0; j < n; ++j) {
            size_t colsize_j = Ap[j];
            Ap[j] = newNNZ;
            newNNZ += colsize_j;
        }
        Ap[n] = newNNZ;
        // BENCHMARK_STOP_TIMER_SECTION("cumsum");

        // Fill row index array `Ai`
        // BENCHMARK_START_TIMER_SECTION("Ai resize");
        Ai.resize(newNNZ);
        // BENCHMARK_STOP_TIMER_SECTION("Ai resize");

        // BENCHMARK_START_TIMER_SECTION("Ai fill");
        // Parallelizing this loop is especially important
        // when `Ai.resize` does not zero-initialize (i.e., when `Ai` is not a
        // std::vector). In that case, the loop's writes will incur first-touch
        // page faults on unmapped memory, and parallelization helps spread
        // this overhead across threads.
        parallel_for_range(n, [&](size_t j) {
            // Note that the following can't be a std::memcpy
            // unless `RowIndex` is the same as `decltype(Ai[0])`...
            size_t offset = bucketStart[j];
            for (index_type ii = Ap[j]; ii < Ap[j + 1]; ++ii)
                Ai[ii] = columnBuckets[offset++];
        });
        // BENCHMARK_STOP_TIMER_SECTION("Ai fill");

        // BENCHMARK_SCOPED_TIMER_SECTION timer2("finalize");
        columnBuckets.resize(0); // this deallocation takes significant time!
        result->nz = newNNZ;
        // result->Ax.resize(newNNZ); // <--- Intentionally leave empty since we generally don't need to store data in the block pattern.

        result->finalize();
        return result;
    }

    template<class FEMMesh_>
    std::unique_ptr<BCSCMat> blockSparsityPatternForMesh(const FEMMesh_ &m) const { return blockSparsityPattern(m.numElements(), [&](size_t ei) { return m.elementNodeIndices(ei); }); }

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

    ////////////////////////////////////////////////////////////////////////////
    // Scalar Hessian assembly.
    ////////////////////////////////////////////////////////////////////////////
    template <class PEH, class EVars>
    struct HessianElementAssemblyData {
        auto block(size_t a, size_t b, size_t bsa, size_t bsb) const { return getBlock(H_e, a, b, bsa, bsb); } // TODO: compare timing with and without eval()...
        // Version where block size is known
        auto block(size_t a, size_t b) const { return getBlock(H_e, a, b); }                                   // TODO: compare timing with and without eval()...
        PEH H_e;
        EVars evars;
    };

    ////////////////////////////////////////////////////////////////////////////
    // Block-accelerated scalar Hessian assembly.
    // (Construct the scalar Hessian but use a block sparsity pattern for
    // acceleration).
    ////////////////////////////////////////////////////////////////////////////
    template<bool UseBlockMergeAlgorithm = UseBlockMergeAlgorithmDefault, typename Real_, class SPMatBlock, class ElementAssemblyDataGetter>
    void assembleHessianBlockAccelerated(Real_ *Ax, const SPMatBlock &blockH, size_t numElements, const ElementAssemblyDataGetter &edataGetter) const {
        if (get_max_num_tbb_threads() == 1) {
            for (size_t ei = 0; ei < numElements; ++ei) {
                auto edata = edataGetter(ei);
                auto He_block = [&edata](size_t a, size_t b, size_t bsa, size_t bsb) { return edata.block(a, b, bsa, bsb); };
                ElementHessianContribAssembler<UseBlockMergeAlgorithm>::template run</* InParallel = */ false>(Ax, blockH, He_block, edata.evars, m_vars, m_varLocks);
            }
        }
        else {
            m_varLocks.init(numBlockVars());
            get_hessian_assembly_arena().execute([Ax, &blockH, &edataGetter, numElements, this]() {
                parallel_for_range(numElements, [Ax, &blockH, &edataGetter, this](size_t ei) {
                    auto edata = edataGetter(ei);
                    auto He_block = [&edata](size_t a, size_t b, size_t bsa, size_t bsb) { return edata.block(a, b, bsa, bsb); };
                    ElementHessianContribAssembler<UseBlockMergeAlgorithm>::template run</* InParallel = */ true>(Ax, blockH, He_block, edata.evars, m_vars, m_varLocks);
                }, 1, 32);
            });
        }
    }

    template<bool UseBlockMergeAlgorithm = UseBlockMergeAlgorithmDefault, typename Real_, class SPMatBlock, class PEHEval, class ElementGetter>
    void assembleHessianBlockAccelerated(Real_ *Ax, const SPMatBlock &blockH, size_t numElements, const PEHEval &eval_He, const ElementGetter &element) const {
        using PEH = decltype(eval_He(0));
        using EVars = decltype(element(0));
        using HEAD = HessianElementAssemblyData<PEH, EVars>;
        assembleHessianBlockAccelerated<UseBlockMergeAlgorithm>(Ax, blockH, numElements, [&](size_t ei) { return HEAD{eval_He(ei), element(ei)}; });
    }

    template<bool UseBlockMergeAlgorithm = UseBlockMergeAlgorithmDefault, typename Real_, class SPMatBlock, class Mesh, class PEHEval>
    void assembleHessianBlockAccelerated(Real_ *Ax, const SPMatBlock &blockH, const Mesh &m, const PEHEval &eval_He) const {
        assembleHessianBlockAccelerated<UseBlockMergeAlgorithm>(Ax, blockH, m.numElements(), eval_He, [&m](size_t ei) { return m.elementNodeIndices(ei); });
    }

    // Assemble Hessian using block acceleration.
    template<bool UseBlockMergeAlgorithm = UseBlockMergeAlgorithmDefault, typename... Args>
    void assembleHessian(BlockCSCHessianBase &H_base, Args&&... args) const {
        BCSCMat &H = BCSCMat::cast(H_base);
        if (H.isSparsityOnly()) H.setZero(); // Allocate Ax array if necessary
                                             // (and accumulate to existing Ax array otherwise)
        assembleHessianBlockAccelerated<UseBlockMergeAlgorithm>(H.Ax.data(), H, std::forward<Args>(args)...);
    }

    template<bool UseBlockMergeAlgorithm = UseBlockMergeAlgorithmDefault, typename... Args>
    void assembleHessian(const std::unique_ptr<BlockCSCHessianBase> &H_ss, Args&&... args) const {
        if (!H_ss) throw std::runtime_error("Attempted to assemble into a BorderedSparseHessian with no block sparsity pattern.");
        assembleHessian<UseBlockMergeAlgorithm>(BCSCMat::cast(*H_ss), std::forward<Args>(args)...);
    }

    template<bool UseBlockMergeAlgorithm = UseBlockMergeAlgorithmDefault, typename... Args>
    void assembleHessian(BorderedSparseHessian &BH, Args&&... args) const {
        if (!BH.H_ss) throw std::runtime_error("Attempted to assemble into a BorderedSparseHessian with no block sparsity pattern.");
        assembleHessian<UseBlockMergeAlgorithm>(*BH.H_ss, std::forward<Args>(args)...);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Legacy/baseline Hessian assembly routines.
    ////////////////////////////////////////////////////////////////////////////
    // Fully customizable Hessian assembly:
    // For each element ei in 0..ne, obtain a data object from `edataGetter`
    // whose `elementVars` method reports the global block variables corresponding to the element
    // and whose `block` method provides accesses to blocks of the per-element Hessian.
    // Note that this `block` method enables additional computation to be
    // performed at assembly time, e.g., to implement chain rule expressions.
    template<class SPMat, class ElementAssemblyDataGetter>
    void assembleHessian(SPMat &H, size_t ne, const ElementAssemblyDataGetter &edataGetter) const {
        if (get_max_num_tbb_threads() == 1) {
            for (size_t ei = 0; ei < ne; ++ei) {
                auto edata = edataGetter(ei);
                m_assembleHessianContrib</* InParallel = */ false>(H, [&edata](size_t a, size_t b, size_t bsa, size_t bsb) {
                    return edata.block(a, b, bsa, bsb);
                }, edata.evars);
            }
        }
        else {
            m_varLocks.init(numBlockVars());

            get_hessian_assembly_arena().execute([&H, &edataGetter, ne, this]() {
                parallel_for_range(ne, [&H, &edataGetter, this](size_t ei) {
                    auto edata = edataGetter(ei);
                    // edata.H_e.template triangularView<Eigen::StrictlyLower>().setZero();
                    // std::cout << "H_e: " << std::endl << edata.H_e << std::endl;
                    m_assembleHessianContrib</* InParallel = */ true>(H, [&edata](size_t a, size_t b, size_t bsa, size_t bsb) {
                        return edata.block(a, b, bsa, bsb);
                    }, edata.evars);
                }, 1, 32);
            });
        }
    }

    // Assemble the per-element Hessian `eval_He(ei)` for element ei in 0..ne.
    // The element's global block variable indices are obtained by calling
    // `element(ei)`, which should return an array of variable indices.
    template<class SPMat, class PEHEval, class ElementGetter>
    void assembleHessian(SPMat &H, size_t ne, const PEHEval &eval_He, const ElementGetter &element) const {
        using PEH = decltype(eval_He(0));
        using EVars = decltype(element(0));
        using HEAD = HessianElementAssemblyData<PEH, EVars>;
        assembleHessian(H, ne, [&](size_t ei) { return HEAD{eval_He(ei), element(ei)}; });
    }

    // Convenience method for the typical case of assembling a per-element Hessian using
    // using nodal variables of a FEMMesh.
    template<class SPMat, class Mesh, class PEHEval>
    void assembleHessian(SPMat &H, const Mesh &m, const PEHEval &eval_He) const {
        assembleHessian(H, m.numElements(), eval_He, [&m](size_t ei) { return m.elementNodeIndices(ei); });
    }

    ////////////////////////////////////////////////////////////////////////////
    // Block-valued CSC Hessian assembly (legacy/baseline).
    ////////////////////////////////////////////////////////////////////////////
    template<class PEH>
    static auto getBlock(const PEH &H_e, size_t a, size_t b, size_t bsa [[maybe_unused]] = VarStructure::MaxBlockDim, size_t bsb [[maybe_unused]] = VarStructure::MaxBlockDim) {
        static constexpr size_t N = VarStructure::MaxBlockDim;
        if constexpr (VarStructure::SingleBlockDim) {
            return H_e.template block<N, N>(a, b);
        }
        else {
            return H_e.block(a, b, bsa, bsb);
        }
    }

    template<class SPMat, class Mesh, class PEHEval>
    void assembleBlockHessian(SPMat &H, const Mesh &m, const PEHEval &eval_He) const {
        static_assert(SingleBlockDim, "Only implemented for SingleBlockDim case");
        static constexpr size_t N = VarStructure::FirstBlockDim;
        if (get_max_num_tbb_threads() == 1) {
            const size_t ne = m.numElements();
            for (size_t ei = 0; ei < ne; ++ei)
                m_assembleHessianBlockContrib</* InParallel = */ false>(H, eval_He(ei), m.elementNodeIndices(ei));
        }
        else {
            m_varLocks.init(numBlockVars());
            get_hessian_assembly_arena().execute([&H, &eval_He, &m, this]() {
                parallel_for_range(m.numElements(), [&H, &eval_He, &m, this](size_t ei) {
                    m_assembleHessianBlockContrib(H, eval_He(ei), m.elementNodeIndices(ei));
                }, 1, 32);
            });
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Gradient assembly.
    ////////////////////////////////////////////////////////////////////////////
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

    ////////////////////////////////////////////////////////////////////////////
    // Gather-based assembly: a more scalable alternative for high threadcount
    // settings.
    //
    // WARNING: this is only supported in cases where only a single, fixed
    // element set is used across the lifetime of the assembler object.
    // Supporting different/changing element sets would require maintaining
    // multiple GatherCache objects.
    //
    // Currently the implemenation also only supports the SingleBlockDim case
    // with compile-time-known per-element gradient sizes. These
    // are not fundamental limitations and could be lifted with additional
    // bookkeeping in the GatherCache.
    struct GatherCache {
        template<class ElementGetter>
        GatherCache(size_t ne, size_t N, size_t numBlockVarsPerElement, const ElementGetter &element) {
            // Cache vertex => (element, local index) map in a CSCMatrix<Char>
            int largest_var_idx = -1;
            TripletMatrix<Triplet<char>> localNodesForNodeTrip(ne * numBlockVarsPerElement * N, /* init_ncols = */ 0);
            for (size_t ei = 0; ei < ne; ++ei) {
                auto bvars = element(ei);
                for (size_t lni = 0; lni < numBlockVarsPerElement; ++lni) {
                    localNodesForNodeTrip.nz.emplace_back(N * (numBlockVarsPerElement * ei + lni), bvars[lni], 1); // bypasses size checks
                    largest_var_idx = std::max<int>(largest_var_idx, bvars[lni]);
                }
            }
            localNodesForNodeTrip.n = largest_var_idx + 1; // infer column size
            localBVarsForGlobalBVar = NodeLocalNodeAdjacencyMatrix(localNodesForNodeTrip);
        }

        Eigen::VectorXd pregatherGradient;
        using NodeLocalNodeAdjacencyMatrix = CSCMatrix<SuiteSparse_long, char>;
        NodeLocalNodeAdjacencyMatrix localBVarsForGlobalBVar; // TODO: we could use a more compact special-purpose datastructure here
    };

    mutable std::unique_ptr<GatherCache> m_gatherCache;

    template<bool Accumulate = true, class Result, class PEGEval, class ElementGetter>
    void assembleGradientGather(Result &g, size_t ne, const PEGEval &eval_ge, const ElementGetter &element) const {
        using PEG = decltype(eval_ge(0));
        static_assert(PEG::SizeAtCompileTime > 0, "Per-element gradient currently must be of compile-time-known size for scatter-gather assembly");
        static_assert(VarStructure::SingleBlockDim, "Only SingleBlockDim case is implemented");

        constexpr size_t numElemLocalVars = PEG::SizeAtCompileTime;
        constexpr size_t N = VarStructure::FirstBlockDim;
        static constexpr int numBlockVarsPerElement = numElemLocalVars / N;

        // Cache vertex => (element, local index) map in a CSCMatrix<Char>
        if (!m_gatherCache)
            m_gatherCache = std::make_unique<GatherCache>(ne, N, numBlockVarsPerElement, element);

        auto &pregatherGradient = m_gatherCache->pregatherGradient;
        pregatherGradient.resize(ne * numElemLocalVars);

        const auto &localBVarsForGlobalBVar = m_gatherCache->localBVarsForGlobalBVar;
        size_t num_bvars = localBVarsForGlobalBVar.n;

        parallel_for_range(ne, [&pregatherGradient, &eval_ge](size_t ei) {
            pregatherGradient.template segment<numElemLocalVars>(ei * numElemLocalVars) = eval_ge(ei);
        }, 32, 100);

        if (size_t(g.size()) < N * num_bvars) throw std::runtime_error("Gradient assembly destination is too small (elements stencils access out-of-bounds)");
        auto *Ai = localBVarsForGlobalBVar.Ai.data();
        auto *Ap = localBVarsForGlobalBVar.Ap.data();
        parallel_for_range(num_bvars, [&g, Ai, Ap, &pregatherGradient](size_t ni) {
                auto *idxPtr = Ai + Ap[ni];
                auto *colEnd = Ai + Ap[ni + 1];
                VecN_T<Real, N> g_n = pregatherGradient.template segment<N>(*idxPtr);
                for (++idxPtr; idxPtr < colEnd; ++idxPtr)
                    g_n += pregatherGradient.template segment<N>(*idxPtr);
                if constexpr (Accumulate) g.template segment<N>(N * ni) += g_n;
                else                      g.template segment<N>(N * ni)  = g_n;
            }, 100, 1000);
    }

    template<bool Accumulate = true, class Result, class Mesh, class PEGEval>
    void assembleGradientGather(Result &g, const Mesh &m, const PEGEval &eval_ge) const {
        return assembleGradientGather<Accumulate>(g, m.numElements(), eval_ge, [&m](size_t ei) { return m.elementNodeIndices(ei); });
    }

    // Due to the limitations of `assembleGradientGather`, we provide
    // this method that dispatches to either it or the regular
    // `assembleGradient` based on compile-time compatibility checks and
    // threading settings.
    // However, this method still only should be used when the caller is sure
    // that the same element set is reused across the lifetime of the
    // assembler object.
    template<bool Accumulate = true, class Result, class PEGEval, class ElementGetter>
    void assembleGradientConditionalGather(Result &g, size_t ne, const PEGEval &eval_ge, const ElementGetter &element) const {
        using PEG = decltype(eval_ge(0));
        if constexpr (PEG::SizeAtCompileTime > 0 && VarStructure::SingleBlockDim) {
            // Gather-based assembly is only beneficial in higher thread-count settings
            if (get_max_num_tbb_threads() >= 4) {
                assembleGradientGather<Accumulate>(g, ne, eval_ge, element);
                return;
            }
        }
        if constexpr (!Accumulate) g.setZero();
        assembleGradient(g, ne, eval_ge, element);
    }

    template<bool Accumulate = true, class Result, class Mesh, class PEGEval>
    void assembleGradientConditionalGather(Result &g, const Mesh &m, const PEGEval &eval_ge) const {
        return assembleGradientConditionalGather<Accumulate>(g, m.numElements(), eval_ge, [&m](size_t ei) { return m.elementNodeIndices(ei); });
    }

    struct ElementColoring {
        template<class ElementGetter>
        ElementColoring(size_t numBlockVars, size_t ne, const ElementGetter &element) {
            int maxColor = -1;
            colors.resize(ne);
            std::vector<std::vector<int>> colorsIncidentNode(numBlockVars);
            for (size_t ei = 0; ei < ne; ++ei) {
                auto bvars = element(ei);
                auto conflict = [&](int color) { // This could be made more efficient (avoiding a separate scan over nodes per color)
                    for (auto v : bvars) {
                        for (int c : colorsIncidentNode[v])
                            if (c == color) return true;
                    }
                    return false;
                };
                int color = 0;
                while (conflict(color)) ++color;

                colors[ei] = color;
                for (auto v : bvars)
                    colorsIncidentNode[v].push_back(color);
                maxColor = std::max(maxColor, color);
            }

            elementsForColor.resize(maxColor + 1);
            Eigen::VectorXi colorCounts = Eigen::VectorXi::Zero(maxColor + 1);
            for (size_t ei = 0; ei < ne; ++ei) ++colorCounts[colors[ei]];
            for (size_t ci = 0; ci < elementsForColor.size(); ++ci) elementsForColor[ci].resize(colorCounts[ci]);

            colorCounts.setZero();
            for (size_t ei = 0; ei < ne; ++ei) {
                int color = colors[ei];
                elementsForColor[color][colorCounts[color]++] = ei;
            }
        }

        Eigen::VectorXi colors;
        std::vector<Eigen::VectorXi> elementsForColor;
    };

    mutable std::unique_ptr<ElementColoring> elementColoring;

    template<class Result, class PEGEval, class ElementGetter>
    void assembleGradientColoring(Result &g, size_t ne, const PEGEval &eval_ge, const ElementGetter &element) const {
        // TODO: run serially for small stencil sets...
        if (!elementColoring)
            elementColoring = std::make_unique<ElementColoring>(numBlockVars(), ne, element);

        for (const auto &efc : elementColoring->elementsForColor) {
            parallel_for_range(efc.size(), [&g, &eval_ge, &element, &efc](size_t i) {
                size_t ei = efc[i];
                auto ge = eval_ge(ei);
                auto bvars = element(ei);
                for (decltype(bvars.size()) lbi = 0; lbi < bvars.size(); ++lbi) {
                    g.template segment<VarStructure::FirstBlockDim>(VarStructure::FirstBlockDim * bvars[lbi]) +=
                        ge.template segment<VarStructure::FirstBlockDim>(VarStructure::FirstBlockDim * lbi);
                }
            }, 32, 100);
        }
    }

private:
    // Assembly into a block-valued CSSC matrix using block sparsity pattern in `H`.
    template<bool InParallel = true, class SPMat, class PEH, class ElemBlockVars>
    void m_assembleHessianBlockContrib(SPMat &H, const PEH &H_e, const ElemBlockVars &blockVars) const {
        static_assert(SingleBlockDim, "Only implemented for SingleBlockDim case");
        static constexpr size_t N = VarStructure::FirstBlockDim;
        static constexpr size_t nbv = std::tuple_size_v<ElemBlockVars>;

        auto order = argsort(blockVars);

        auto *Ap = H.Ap.data();
        auto *Ai = H.Ai.data();

        for (size_t lbj_i = 0; lbj_i < nbv; ++lbj_i) {
            size_t lbj = order[lbj_i];
            auto bj = blockVars[lbj];

            SuiteSparse_long head = Ap[bj];
            SuiteSparse_long colEnd = Ap[bj + 1];

            if constexpr (InParallel) m_varLocks.lock(bj);
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
            if constexpr (InParallel) m_varLocks.unlock(bj);
        }
    }

    // Assembly into a scalar-valued CSC matrix using scalar sparsity pattern in `H`.
    template<bool InParallel = true, class SPMat, class HeBlock, class ElemBlockVars>
    void m_assembleHessianContrib(SPMat &H, const HeBlock &He_block, const ElemBlockVars &blockVars) const {
        PerElementBlockOffsetCalculation<VarStructure, ElemBlockVars> blockInfo(m_vars, blockVars);

        for (decltype(blockVars.size()) lbj = 0; lbj < blockVars.size(); ++lbj) {
            const auto bj = blockVars[lbj];
            if constexpr (InParallel) m_varLocks.lock(bj);
            const auto lvar_j = blockInfo.offset(lbj);
            const auto gvar_j = blockInfo.globalScalarVar(lbj, blockVars[lbj]);
            const auto bsj    = blockInfo.blockSize(lbj);
            for (decltype(blockVars.size()) lbi = 0; lbi < blockVars.size(); ++lbi) {
                const auto lvar_i = blockInfo.offset(lbi);
                const auto gvar_i = blockInfo.globalScalarVar(lbi, blockVars[lbi]);
                const auto bsi    = blockInfo.blockSize(lbi);

                if (gvar_i > gvar_j) { continue; }
                bool localUpperTri = lbi <= lbj;

                std::decay_t<decltype(He_block(lvar_i, lvar_j, bsi, bsj).eval())> block;
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
                    if (bsj == 1) { H.Ax[idx] += block.data()[0]; continue; }
                    for (size_t c = 0; c < bsj; ++c) {
                        typename SPMat::DataMap(H.Ax.data() + idx, c + 1) += block.col(c).topRows(c + 1);
                        idx += H.col_nnz(gvar_j + c);
                    }
                }
            }
            if constexpr (InParallel) m_varLocks.unlock(bj);
        }
    }

    // Implementation of the dynamic SystemAssemblerBase::blockSparsityPattern method.
    using DynamicElementGetter = SystemAssemblerBase::DynamicElementGetter;
    virtual std::unique_ptr<BlockCSCHessianBase> m_blockSparsityPatternDynamicImpl(size_t numElements, size_t blockSize, const DynamicElementGetter &elementGetter) const override {
        return blockSparsityPattern(numElements,
                [this, blockSize, &elementGetter](size_t ei) {
                    std::vector<size_t> elem = elementGetter(ei);
                    for (size_t i = 0; i < elem.size(); ++i) {
                        size_t vs = blockSize * elem[i]; // Scalar location of the start of the caller's block
                        size_t v = m_vars.blockContainingVar(vs);
                        auto [gvar, bs] = m_vars.blockInfo(v);
                        // Verify that the block the caller wants to insert
                        // fits inside block `v` of our VarStructure.
                        if (vs < gvar || vs + blockSize > gvar + bs) throw std::runtime_error("An element's block variable does not fit a single block of our VarStructure");
                        elem[i] = v;
                    }
                    return elem;
                });
    }

    mutable std::vector<char> m_sparsityChangeDetectionScratch;
    mutable VarLocks m_varLocks;
    VarStructure m_vars;
};

using ScalarSystemAssembler = SystemAssembler<1>;

template<class VS>
struct SystemAssemblerForVarStructure;

template<size_t... BlockDimensions_>
struct SystemAssemblerForVarStructure<OptimizationVarStructure<BlockDimensions_...>> : public SystemAssembler<BlockDimensions_...> { };

} // namespace MeshFEM

#endif /* end of include guard: SYSTEMASSEMBLER_HH */
