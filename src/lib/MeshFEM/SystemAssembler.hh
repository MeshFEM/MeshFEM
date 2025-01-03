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
#include <tuple>
#include <functional>
#include <MeshFEM/Handles/FEMMeshHandles.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Flattening.hh>
#include "Utilities/static_sort.hh"

#include "VarStructure.hh"
#include "BlockCSCHessian.hh"

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

private:
    virtual std::unique_ptr<BlockCSCHessianBase> m_blockSparsityPatternDynamicImpl(size_t numElements, size_t blockSize, const DynamicElementGetter &elementGetter) const = 0;
};

template<size_t... BlockDimensions_>
struct MESHFEM_EXPORT SystemAssembler : public SystemAssemblerBase {
    using index_type = SuiteSparse_long;
    using CSCMat = CSCMatrix<index_type, double>;
    using VarStructure = OptimizationVarStructure<BlockDimensions_...>;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;

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

    template<class FEMMesh_>
    std::unique_ptr<BCSCMat> blockSparsityPatternForMesh(const FEMMesh_ &m) const {
        return blockSparsityPattern(m.numElements(),
                [&](size_t ei) {
                    std::array<size_t, FEMMesh_::NumNodesPerElement> blockVarsForElement;
                    auto e = m.element(ei);
                    for (const auto n_b : e.nodes()) { blockVarsForElement[n_b.localIndex()] = n_b.index(); }
                    return blockVarsForElement;
                });
    }

    template<class ElemBlockVarsForElement>
    std::unique_ptr<BCSCMat> blockSparsityPattern(size_t numElems, const ElemBlockVarsForElement &blockVarsForElement) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("blockSparsityPattern");

        const bool parallel = get_max_num_tbb_threads() > 1;

        if (parallel) m_initVarLocks();

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
        auto result = emptyBlockSparsityPattern();
        auto &Ap = result->Ap;
        auto &Ai = result->Ai;
        const size_t n = numBlockVars;

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

            if (parallel) {
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
            else {
                for (size_t ei = 0; ei < numElems; ++ei) {
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
                }
            }
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

        result->nz = newNNZ;
        // result->Ax.resize(newNNZ); // <--- Intentionally leave empty since we generally don't need to store data in the block pattern.

        result->finalize();
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
            m_initVarLocks();

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
    // Block Hessian assembly.
    ////////////////////////////////////////////////////////////////////////////
    template<class PEH>
    static auto getBlock(const PEH &H_e, size_t a, size_t b, size_t bsa = VarStructure::MaxBlockDim, size_t bsb = VarStructure::MaxBlockDim) {
        static constexpr size_t N = VarStructure::MaxBlockDim;
        if constexpr (VarStructure::SingleBlockDim) {
            return H_e.template block<N, N>(a, b);
        }
        else {
            return H_e.block(a, b, bsa, bsb);
        }
    }

    template<typename T, size_t Size>
    static auto argsort(const std::array<T, Size> &blockVars) {
        std::array<size_t, Size> order;
        for (size_t i = 0; i < Size; ++i) { order[i] = i; }
        StaticTimSort<Size> timBoseNelsonSort;
        timBoseNelsonSort(order, [&blockVars](size_t a, size_t b) { return blockVars[a] < blockVars[b]; });
        return order;
    }

    template<size_t MinSize, size_t MaxSize>
    static auto argsort(const ElementBlockVarsWithSizeRange<MinSize, MaxSize> &blockVars) {
        ElementBlockVarsWithSizeRange<MinSize, MaxSize> order;
        order.resize(blockVars.size());
        for (size_t i = 0; i < blockVars.size(); ++i) { order[i] = i; }
        dispatchedStaticSort<MinSize, MaxSize>(order.data(), order.size(), [&blockVars](size_t a, size_t b) { return blockVars[a] < blockVars[b]; });
        return order;
    }

    // TODO: version for fully dynamic element sizes that uses std::sort
    //      std::sort(order.begin(), order.end(), [&blockVars](size_t a, size_t b) { return blockVars[a] < blockVars[b]; });

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
            m_initVarLocks();
            get_hessian_assembly_arena().execute([&H, &eval_He, &m, this]() {
                parallel_for_range(m.numElements(), [&H, &eval_He, &m, this](size_t ei) {
                    m_assembleHessianBlockContrib(H, eval_He(ei), m.elementNodeIndices(ei));
                }, 1, 32);
            });
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Block-accelerated scalar Hessian assembly.
    // (Construct the scalar Hessian but use a block sparsity pattern for
    // acceleration).
    ////////////////////////////////////////////////////////////////////////////
    template<typename Real_, class SPMatBlock, class ElementAssemblyDataGetter>
    void assembleHessianBlockAccelerated(Real_ *Ax, const SPMatBlock &blockH, size_t numElements, const ElementAssemblyDataGetter &edataGetter) const {
        if (get_max_num_tbb_threads() == 1) {
            for (size_t ei = 0; ei < numElements; ++ei) {
                auto edata = edataGetter(ei);
                auto He_block = [&edata](size_t a, size_t b, size_t bsa, size_t bsb) { return edata.block(a, b, bsa, bsb); };
                m_assembleHessianContribBlockAccelerated</* InParallel = */ false>(Ax, blockH, He_block, edata.evars);
            }
        }
        else {
            m_initVarLocks();
            get_hessian_assembly_arena().execute([Ax, &blockH, &edataGetter, numElements, this]() {
                parallel_for_range(numElements, [Ax, &blockH, &edataGetter, this](size_t ei) {
                    auto edata = edataGetter(ei);
                    auto He_block = [&edata](size_t a, size_t b, size_t bsa, size_t bsb) { return edata.block(a, b, bsa, bsb); };
                    m_assembleHessianContribBlockAccelerated(Ax, blockH, He_block, edata.evars);
                }, 1, 32);
            });
        }
    }

    template<typename Real_, class SPMatBlock, class PEHEval, class ElementGetter>
    void assembleHessianBlockAccelerated(Real_ *Ax, const SPMatBlock &blockH, size_t numElements, const PEHEval &eval_He, const ElementGetter &element) const {
        using PEH = decltype(eval_He(0));
        using EVars = decltype(element(0));
        using HEAD = HessianElementAssemblyData<PEH, EVars>;
        assembleHessianBlockAccelerated(Ax, blockH, numElements, [&](size_t ei) { return HEAD{eval_He(ei), element(ei)}; });
    }

    template<typename Real_, class SPMatBlock, class Mesh, class PEHEval>
    void assembleHessianBlockAccelerated(Real_ *Ax, const SPMatBlock &blockH, const Mesh &m, const PEHEval &eval_He) const {
        assembleHessianBlockAccelerated(Ax, blockH, m.numElements(), eval_He, [&m](size_t ei) { return m.elementNodeIndices(ei); });
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

    using VXd = Eigen::VectorXd;
    mutable VXd m_pregatherGradient;
    using NodeLocalNodeAdjacencyMatrix = CSCMatrix<SuiteSparse_long, char>;
    mutable std::shared_ptr<NodeLocalNodeAdjacencyMatrix> m_localNodesForNode;
    template<bool Accumulate = true, class Result, class Mesh, class PEGEval>
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
                if constexpr (Accumulate) g.template segment<N>(N * ni) += g_n;
                else                      g.template segment<N>(N * ni)  = g_n;
            }, 100, 100);
    }

private:
    void m_initVarLocks() const {
        if (m_varLocks) return;
        size_t numLocks = numBlockVars();
        m_varLocks = std::make_unique<std::vector<std::atomic<bool>>>(numLocks);
        for (size_t i = 0; i < numLocks; ++i)
            atomic_init(&(*m_varLocks)[i], false);
    }

    // Assembly into a block-valued CSSC matrix using block sparsity pattern in `H`.
    template<bool InParallel = true, class SPMat, class PEH, class ElemBlockVars>
    void m_assembleHessianBlockContrib(SPMat &H, const PEH &H_e, const ElemBlockVars &blockVars) const {
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

    // Assembly into a scalar-valued CSSC matrix using scalar sparsity pattern in `H`.
    template<bool InParallel = true, class SPMat, class HeBlock, class ElemBlockVars>
    void m_assembleHessianContrib(SPMat &H, const HeBlock &He_block, const ElemBlockVars &blockVars) const {
        PerElementBlockOffsetCalculation<VarStructure, ElemBlockVars> blockInfo(m_vars, blockVars);

        for (decltype(blockVars.size()) lbj = 0; lbj < blockVars.size(); ++lbj) {
            const auto bj = blockVars[lbj];
            if constexpr (InParallel) m_lockVar(bj);
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
            if constexpr (InParallel) m_unlockVar(bj);
        }
    }

    // Assemble into a scalar-valued `Ax` array but use the block sparsity
    // pattern in `H` for acceleration. This is the correct assembly routine
    // to use for the `BlockCSCHessian` format.
    template<bool InParallel = true, class Real_, class SPMatBlock, class HeBlock, class ElemBlockVars>
    void m_assembleHessianContribBlockAccelerated(Real_ *Ax, const SPMatBlock &blockH, const HeBlock &He_block, const ElemBlockVars &blockVars) const {
        PerElementBlockOffsetCalculation<VarStructure, ElemBlockVars> blockOffsetCalc(m_vars, blockVars);

        auto order = argsort(blockVars);

        for (size_t lbj_i = 0; lbj_i < blockVars.size(); ++lbj_i) {
            size_t lbj = order[lbj_i];
            auto bj = blockVars[lbj];
            const size_t lbo_j = blockOffsetCalc.offset(lbj);
            const size_t bs_j = blockOffsetCalc.blockSize(lbj);

            auto colScanner = blockH.columnScanner(bj);
            if constexpr (InParallel) m_lockVar(bj);

            for (size_t lbi_i = 0; lbi_i < lbj_i; ++lbi_i) {
                size_t lbi = order[lbi_i];
                auto bi = blockVars[lbi];
                const size_t lbo_i = blockOffsetCalc.offset(lbi);
                const size_t bs_i = blockOffsetCalc.blockSize(lbi);

                auto addBlock = [&](auto block) {
                    // Find offset in `Ax` of the block's upper-left corner.
#if 1
                    SuiteSparse_long loc = colScanner.advanceToBlock(bi);
#else
                    SuiteSparse_long loc = colScanner.findBlock(bi);
#endif
                    for (size_t c = 0; c < bs_j; ++c) {
                        if constexpr (SingleBlockDim)
                            Eigen::Map<Eigen::Matrix<Real_, VarStructure::MaxBlockDim, 1>>(Ax + loc) += block.col(c);
                        else
                            Eigen::Map<Eigen::Matrix<Real_, Eigen::Dynamic, 1>>(Ax + loc, bs_i) += block.col(c);

                        loc += colScanner.stride() + c; // each subsequent column has an extra entry...
                    }
                };

                if (lbi < lbj) addBlock(He_block(lbo_i, lbo_j, bs_i, bs_j));
                else           addBlock(He_block(lbo_j, lbo_i, bs_i, bs_j).transpose());
            }

            // Add (upper triangle of) diagonal block, starting at final entry
            // of first column.
            SuiteSparse_long loc = colScanner.diagBlockScalarLoc();
            auto block = He_block(lbo_j, lbo_j, bs_j, bs_j);
            for (size_t c = 0; c < bs_j; ++c) {
                Eigen::Map<Eigen::Matrix<Real_, Eigen::Dynamic, 1>>(Ax + loc, c + 1) += block.col(c).topRows(c + 1);
                loc += colScanner.stride() + c; // each subsequent column has an extra entry...
            }

            if constexpr (InParallel) m_unlockVar(bj);
        }
    }

    // Implementation of the dynamic SystemAssemblerBase::blockSparsityPattern method.
    using DynamicElementGetter = SystemAssemblerBase::DynamicElementGetter;
    virtual std::unique_ptr<BlockCSCHessianBase> m_blockSparsityPatternDynamicImpl(size_t numElements, size_t blockSize, const DynamicElementGetter &elementGetter) const override {
        return blockSparsityPattern(numElements,
                [this, blockSize, &elementGetter](size_t ei) {
                    std::vector<size_t> elem = elementGetter(ei);
                    for (size_t i = 0; i < elem.size(); ++i) {
                        size_t vb_i = elem[i];
                        size_t v = m_vars.blockContainingVar(vb_i * blockSize);
                        auto [gvar, bs] = m_vars.blockInfo(v);
                        if (gvar > vb_i || vb_i + blockSize > gvar + bs) throw std::runtime_error("An element's block variable does not fit a single block of our VarStructure");
                        elem[i] = v;
                    }
                    return elem;
                });
    }

    void   m_lockVar(size_t var) const { while ((*m_varLocks)[var].exchange(true, std::memory_order_acquire)); }
    void m_unlockVar(size_t var) const {        (*m_varLocks)[var].store  (false, std::memory_order_release);  }

    mutable std::vector<char> m_sparsityChangeDetectionScratch;
    mutable std::unique_ptr<std::vector<std::atomic<bool>>> m_varLocks;
    VarStructure m_vars;
};

template<class VS>
struct SystemAssemblerForVarStructure;

template<size_t... BlockDimensions_>
struct SystemAssemblerForVarStructure<OptimizationVarStructure<BlockDimensions_...>> : public SystemAssembler<BlockDimensions_...> { };

#endif /* end of include guard: SYSTEMASSEMBLER_HH */
