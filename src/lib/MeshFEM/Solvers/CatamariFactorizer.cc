#include "CatamariFactorizer.hh"
#include "MeshFEM/GlobalBenchmark.hh"
#include <limits>

#if MESHFEM_WITH_CATAMARI

#include "CholmodFactorizer.hh"
#include "amd.h"

#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include <specify.hpp>

#if MESHFEM_WITH_SCOTCH
#include "ScotchOrdering.hh"
#endif

#if CATAMARI_FINEGRAINED_TIMERS
#include <filesystem>
#endif

#include "CatamariConversionPlan.hh"

// The largest block size for which to instantiate a BlockCatamari solver.
#define MAX_INSTANTIATED_BLOCK_SIZE 3

template<size_t MaxBlockSize = MAX_INSTANTIATED_BLOCK_SIZE>
SuiteSparseMatrix expandSparsityPattern(const SuiteSparseMatrix &A, size_t blockSize) {
    if (blockSize == MaxBlockSize) return A.expandSparsityPattern<MaxBlockSize, /*AssumeDiagonalExists = */ true>();
    else {
        if constexpr (MaxBlockSize > 1) return expandSparsityPattern<MaxBlockSize - 1>(A, blockSize);
        else                            throw std::runtime_error("Unsupported block size");
    }
}

// Support for converting a SuiteSparseMatrix holding only the *upper triangle*
// of a CSC-format symmetric matrix into a catamari::CoordinateMatrix of the
// full matrix. The output format is essentially CSR but with the row indices
// also explicitly stored.
// The upper triangle values in column `j` can be copied directly to the output
// entries starting at `out.RowEntryOffset(j)`; these are in the *lower*
// triangle of the output matrix due to the CSC->CSR conversion.
// Then, the strict upper triangle entries should be copied also into the
// locations corresponding to their implied reflected copies in the input
// matrix's lower triangle. To prevent looking up these locations in each of
// the many conversions done with a fixed sparsity pattern, we cache their
// entry pointers in a lookup table.
struct CatamariConverter {
    using CMat = catamari::CoordinateMatrix<double>;
    using ConversionPlan = catamari::ConversionPlan;

    // Note `Asp_in` is the rowcol-reduced *block* sparsity pattern.
    // If `blockSize > 1`, then `Asp_in` will be expanded from a "block sparsity
    // pattern" to a "scalar sparsity pattern" before converting.
    // We retain this to support legacy-Catamari mode, but for best efficiency,
    // the caller should pass `blockSize = 1` and interpret the converter's
    // entries as representing blocks of the appropriate size.
    CatamariConverter(const SuiteSparseMatrix &Asp_in, const size_t blockSize, bool legacy, const std::vector<SuiteSparse_long> &entryForReducedEntry)
        : m_legacy(legacy)
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter");
        if (Asp_in.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");
        if (Asp_in.m != Asp_in.n) throw std::runtime_error("Only square matrices are supported");

        const SuiteSparseMatrix *Asp_ptr = &Asp_in;
        SuiteSparseMatrix A_scalar;
        if (blockSize > 1) { A_scalar = expandSparsityPattern<>(Asp_in, blockSize); Asp_ptr = &A_scalar; }
        const SuiteSparseMatrix &Asp = *Asp_ptr;

        // Convert upper-triangle sparsity pattern to a full symmetric sparsity
        // pattern in Catamari format.
        m_result.Resize(Asp_ptr->n, Asp_ptr->n);

        {
            // TODO: generalized version of `CSCMat::InOrderBuilder` that
            // enables us to build the catamari::CoordinateMatrix in-place...

            // Get an integer-valued sparse matrix where each entry holds the
            // index of the source upper triangle entry that generated it.
            BENCHMARK_START_TIMER_SECTION("toSymmetryMode");
            CSCMatrix<SuiteSparse_long, SuiteSparse_long> A_full = Asp_ptr->toSymmetryModeImpl<SuiteSparse_long>(SuiteSparseMatrix::SymmetryMode::NONE, [](size_t ii) { return ii; });
            BENCHMARK_STOP_TIMER_SECTION("toSymmetryMode");

            catamari::Buffer<catamari::MatrixEntry<typename SuiteSparseMatrix::value_type>> new_entries(A_full.nz);
            parallel_for_range(A_full.n, [&](size_t j) {
                for (SuiteSparse_long ii = A_full.Ap[j]; ii < A_full.Ap[j + 1]; ++ii) {
                    SuiteSparse_long i = A_full.Ai[ii];
                    new_entries[ii].row = j; // transpose: Catamari uses CSR storage
                    new_entries[ii].column = i;
                    // new_entries[ii].value = 1; // Value won't be referenced...
                }
            }, /* grain_size = */ 64, /* parallelism_threshold = */ 128);


            catamari::Buffer<catamari::Int> row_entry_offsets(A_full.Ap.size());
            for (size_t i = 0; i < A_full.Ap.size(); ++i)
                row_entry_offsets[i] = A_full.Ap[i];
            m_result.SetSortedEntries(std::move(new_entries), std::move(row_entry_offsets));

            m_sourceReducedEntryForFullMatrixEntry = std::move(A_full.Ax);
        }

        if (legacy) {
            // Determine where to find each entry of the Catamari matrix `m_result`
            // within the scalar values array of the original input matrix (pre row-col removal).
            m_sourceLocForCatamariInputEntry.assign(m_result.NumEntries(), -1);
            for (catamari::Int j = 0; j < Asp.n; ++j) {
                for (auto ii = Asp.Ap[j]; ii < Asp.Ap[j + 1]; ++ii) {
                    catamari::Int i = Asp.Ai[ii];
                    SuiteSparse_long loc = entryForReducedEntry.empty() ? ii : entryForReducedEntry[ii];
                    m_sourceLocForCatamariInputEntry[m_result.EntryOffset(i, j)] = loc;
                    if (i != j)
                        m_sourceLocForCatamariInputEntry[m_result.EntryOffset(j, i)] = loc;
                }
            }
            for (SuiteSparse_long loc : m_sourceLocForCatamariInputEntry)
                if (loc == -1) throw std::runtime_error("Missing source entry for full matrix entry");
        }
    }
    std::vector<SuiteSparse_long> m_sourceReducedEntryForFullMatrixEntry;

    // Note Ax_data is the scalar data before rowcol removal.
    // Legacy: convert and cache the numerical values of matrix `A` (assuming `A` has
    // an identical sparsity pattern to `m_Asp`).
    const CMat &convert(const double *Ax_data) {
        if (!m_legacy) throw std::runtime_error("convert() is for legacy mode only!");
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter.convert");
        catamari::Int nc = m_result.NumColumns();
        for (size_t i = 0; i < m_result.Entries().Size(); ++i)
            m_result.Entries()[i].value = Ax_data[m_sourceLocForCatamariInputEntry[i]];
        return m_result;
    }

    // Legacy: convert and cache the numerical values of matrix `A + sigma B` (assuming
    // `A` and `B` have identical sparsity patterns to `m_Asp`).
    // If `B_data == nullptr`, convert the values `A + sigma * I`.
    const CMat &convert(const double *Ax_data, double sigma, const double *B_data) {
        if (!m_legacy) throw std::runtime_error("convertWithShift() is for legacy mode only!");
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter.convert");
        catamari::Int nc = m_result.NumColumns();

        if (B_data != nullptr) {
            for (size_t i = 0; i < m_result.Entries().Size(); ++i) {
                SuiteSparse_long loc = m_sourceLocForCatamariInputEntry[i];
                m_result.Entries()[i].value = Ax_data[loc] + sigma * B_data[loc];
            }
        }
        else {
            for (size_t i = 0; i < m_result.Entries().Size(); ++i)
                m_result.Entries()[i].value = Ax_data[m_sourceLocForCatamariInputEntry[i]];
            if (sigma != 0) { // Add the shift to the diagonal entries
                // This is slow!!!
                for (catamari::Int j = 0; j < m_result.NumColumns(); ++j)
                    m_result.Entries()[m_result.EntryOffset(j, j)].value += sigma;
            }
        }

        return m_result;
    }

    // Get the most recently converted matrix.
    const CMat &get() const { return m_result; }

    void printDebugEntries(size_t maxEntries = 15) const {
        std::cout << "entries:";
        for (size_t i = 0; i < std::min<size_t>(m_result.NumEntries(), maxEntries); ++i) {
            const auto &e = m_result.Entry(i);
            std::cout << "\t" << e.row << ", " << e.column;
        }
        std::cout << std::endl;

        std::cout << "Row offsets";
        for (size_t i = 0; i < std::min<size_t>(m_result.NumRows(), maxEntries); ++i) {
            std::cout << "\t" << m_result.RowEntryOffset(i);
        }
        std::cout << std::endl;
    }

    void freeCatamariMatrix() { m_result.Empty(); m_sourceReducedEntryForFullMatrixEntry.clear(); m_sourceReducedEntryForFullMatrixEntry.shrink_to_fit(); }

    ConversionPlan conversionPlan;

private:
    CMat m_result;

    const bool m_legacy = false;
    SuiteSparseMatrix m_Asp; // For legacy mode only
    std::vector<SuiteSparse_long> m_sourceLocForCatamariInputEntry; // For legacy mode only
};

CatamariFactorizer::CatamariFactorizer(bool legacy) {
    m_ldl        = std::make_unique<catamari::SparseLDL<double>>();
    m_ldlControl = std::make_unique<catamari::SparseLDLControl<double>>();
    m_ldlControl->SetFactorizationType(catamari::kCholeskyFactorization);
    m_ldlControl->supernodal_strategy = catamari::kSupernodalFactorization;
    m_ldlControl->supernodal_control.algorithm = catamari::kRightLookingLDL; // catamari::kRightLookingLDL;
    m_ldlControl->supernodal_control.relaxation_control.relax_supernodes = true;
    m_ldlControl->supernodal_control.parallel_ratio_threshold = 0.02;
    m_ldlControl->supernodal_control.legacy = m_legacy = legacy;
    // m_ldlControl->supernodal_control.factor_tile_size = std::numeric_limits<catamari::Int>::max(); // Effectively disable node-level parallelism
}

void CatamariFactorizer::setUseLeftLooking(bool use_left) { m_ldlControl->supernodal_control.algorithm = use_left ? catamari::kLeftLookingLDL : catamari::kRightLookingLDL; }
bool CatamariFactorizer::getUseLeftLooking() const { return m_ldlControl->supernodal_control.algorithm == catamari::kLeftLookingLDL; }

size_t CatamariFactorizer::m_reduced() const { assertFactorization(FactorizationType::Symbolic); return m_ldl->NumRows(); }
size_t CatamariFactorizer::n_reduced() const { assertFactorization(FactorizationType::Symbolic); return m_ldl->NumRows(); }

void CatamariFactorizer::factorizeSymbolic(const BlockCSCHessianBase &mat, const std::vector<size_t> &pinnedVars) {
    recordSymbolic(mat, pinnedVars);
    // We only support uniform block sizes, and only up to
    // `MAX_INSTANTIATED_BLOCK_SIZE`; all others get converted to an ordinary scalar matrix.
    // TODO: convert to GCD block size instead? Do we have a use case for this?
    // TODO: try block reordering of nonuniform block sizes (then expand to scalar)?

    const bool blockFactorizationSupported = m_useBlockAccel && mat.uniformBlockSize() && (mat.maxBlockSize() <= MAX_INSTANTIATED_BLOCK_SIZE);
    if (blockFactorizationSupported) {
        m_blockSize = mat.maxBlockSize();
        m_factorizeSymbolic((const SuiteSparseMatrix &) mat, pinnedVars);
    }
    else {
        m_scalarHessian = mat.toScalar(/* sparsityOnly = */ true);
        m_dataOffsetForScalarHessianLoc = mat.dataOffsetsForScalarCSCDataOffsets(m_scalarHessian);
        m_blockSize = 1;
        m_factorizeSymbolic(m_scalarHessian, pinnedVars);
    }
}

void CatamariFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    m_blockSize = 1;
    m_factorizeSymbolic(mat, pinnedVars);
}

// When `m_blockSize > 1` then `mat` holds the block sparsity pattern with
// uniform block size `m_blockSize`.
// `pinnedVars` always holds scalar variables.
void CatamariFactorizer::m_factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    const SuiteSparseMatrix *A_reduced;
    std::vector<SuiteSparse_long> reducedRowForRow_block;
    std::vector<SuiteSparse_long> blockEntryForReducedBlockEntry; // the original block nz corresponding to each nz in the block row-col-removed matrix

    if (m_blockSize > 1 && pinnedVars.size() > 0) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("BlockCSC Pin Handling");
        // Check for partially pinned blocks, which currently require a scalar
        // factorization fallback.

        // Convert the scalar variable indices in `pinnedVars` to their
        // corresponding block variable indices.
        size_t numBlockVars = mat.n;
        std::vector<bool> scalarFixedVarMask(numBlockVars * m_blockSize, false);
        std::vector<size_t> pinnedBlockVars, scalarFixedVars;
        std::vector<size_t> numComponentsPinned(numBlockVars); // how many scalar variables within each block have been pinned?
        for (size_t i : pinnedVars) {
            if (scalarFixedVarMask[i]) continue;
            scalarFixedVarMask[i] = true;
            scalarFixedVars.push_back(i);
            size_t bi = i / m_blockSize;
            if (++numComponentsPinned[bi] == 1) pinnedBlockVars.push_back(bi);
        }
        // Detect entries of `pinnedVars` that do not respect the block
        // structure (i.e., that pin only part of a block); these will need to
        // be handled specially.
        for (size_t bi : pinnedBlockVars) {
            if (numComponentsPinned[bi] != m_blockSize) {
                std::cout << "WARNING: Partially-pinned block variables not yet implemented; falling back to scalar factorization" << std::endl;
                const BlockCSCHessianBase &bmat = static_cast<const BlockCSCHessianBase &>(mat);
                m_scalarHessian = bmat.toScalar(/* sparsityOnly = */ true);
                m_dataOffsetForScalarHessianLoc = bmat.dataOffsetsForScalarCSCDataOffsets(m_scalarHessian);
                m_blockSize = 1;
                return m_factorizeSymbolic(m_scalarHessian, pinnedVars);
            }
            // TODO: keep the partially pinned block in the sparsity pattern and
            // apply the scalar pin constraint during numeric factorization?
        }
        A_reduced = m_initRowColRemoval(mat, pinnedBlockVars);
        blockEntryForReducedBlockEntry.swap(m_entryForReducedEntry);
        reducedRowForRow_block.swap(m_reducedRowForRow);

        // `m_initRowColRemoval` has now stored the pinned **block** variable
        // indices, whereas `m_fixedVars` should store **scalar** variable indices.
        m_fixedVars.swap(scalarFixedVars);

        if (!reducedRowForRow_block.empty()) {
            // Upgrade `reducedRowForRow_block` to a scalar version as needed
            // for the `solve` phase.
            m_reducedRowForRow.resize(m_blockSize * mat.n);
            for (size_t i = 0; i < reducedRowForRow_block.size(); ++i) {
                SuiteSparse_long brr = reducedRowForRow_block[i];
                if (brr == SuiteSparseMatrix::INDEX_NONE) {
                    for (size_t c = 0; c < m_blockSize; ++c)
                        m_reducedRowForRow[m_blockSize * i + c] = SuiteSparseMatrix::INDEX_NONE;
                }
                else {
                    for (size_t c = 0; c < m_blockSize; ++c)
                        m_reducedRowForRow[m_blockSize * i + c] = m_blockSize * brr + c;
                }
            }
        }
    }
    else {
        A_reduced = m_initRowColRemoval(mat, pinnedVars);
        reducedRowForRow_block = m_reducedRowForRow;
    }

    m_permutedReducedRowForRow.clear(); // The upcoming symbolic factorization will change any existing permutation...

    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Symbolic Factorize");
    // Note: passing `block_size = 1` below prevents the converter from
    // expanding entries in the (block) sparsity pattern into
    // `block_size` x `block_size` blocks of scalars in the block case.
    // (I.e., we leave the pattern in its compressed form.)
    if (m_catamariConverter) {
        BENCHMARK_SCOPED_TIMER_SECTION t2("CatamariConverter_reset");
        m_catamariConverter.reset();
    }
    m_catamariConverter = std::make_unique<CatamariConverter>(*A_reduced, /* block_size = */ 1, m_legacy, m_entryForReducedEntry);

    m_ldlControl->supernodal_control.relaxation_control.block_size = m_blockSize;

    if (orderingMethod == OrderingMethod::Catamari)
        m_ldl->Factor(m_catamariConverter->get(), *m_ldlControl, /* symbolic_only = */ true);
    else if ((orderingMethod == OrderingMethod::CholmodNesdis) || (orderingMethod == OrderingMethod::Metis)
          || (orderingMethod == OrderingMethod::AMD) || (orderingMethod == OrderingMethod::Adaptive)) {
        if (!m_c) {
            m_c = std::make_unique<cholmod_common>();
            cholmod_l_start(m_c.get());
        }

        OrderingMethod actualOrderingMethod = orderingMethod;
        if (orderingMethod == OrderingMethod::Adaptive) {
            actualOrderingMethod = adaptiveOrdering.updateSelection();
            // std::cout << "Adaptive ordering method selected: "
            //           << ((actualOrderingMethod == OrderingMethod::CholmodNesdis) ? "Nesdis" : "AMD")
            //           << "  " << adaptiveOrdering.factorizationTimingDescription();
            // std::cout << std::endl;
        }

        // Time the ordering-dependent parts of the symbolic factorization
        auto sym_fact_start = std::chrono::high_resolution_clock::now();

        catamari::SymmetricOrdering ordering;
        {
            static_assert(sizeof(SuiteSparse_long) == sizeof(catamari::Int), "Mismatched integer type");
            ordering.inverse_permutation.Resize(A_reduced->m);
            auto cholmat = cholmod_sparse_view(*A_reduced);
            // Note: the array `cholmat.x` apparently must be valid or cholmod_l_nested_dissection fails
            // (even though the Nested dissection algorithm should not be
            // looking at its entries...)
            // Presumably this is because the first step of cholmod_l_nested_dissection
            // is to convert the matrix from upper-triangular to full format.
            // In the future, we should bypass this step since we already do the
            // conversion ourselves for Catamari.
            cholmat.x = const_cast<double *>((const double *) A_reduced->Ai.data());

            if (actualOrderingMethod == OrderingMethod::CholmodNesdis) {
#if 0 // Whether to downcast for ordering -- the difference in time seems negligible
                if (!m_c_int) {
                    m_c_int = std::make_unique<cholmod_common>();
                    cholmod_start(m_c_int.get());
                }

                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_nesdis");
                VecX_T<int> Ai_downcast, Ap_downcast, iperm_downcast;

                Ai_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A_reduced->Ai.data(), A_reduced->Ai.size()).template cast<int>();
                Ap_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A_reduced->Ap.data(), A_reduced->Ap.size()).template cast<int>();
                auto cholmat_downcast = cholmod_sparse_view(A_reduced->m, A_reduced->n, A_reduced->nz, cholmat.x,
                                                            Ai_downcast.data(), Ap_downcast.data());
                iperm_downcast.resize(A_reduced->m);
                cholmod_nested_dissection(&cholmat_downcast, /* fset = */ nullptr, /* fsize = */ 0,
                                            iperm_downcast.data(), (int *) CParent.Data(), (int *) CMember.Data(), m_c_int.get());
                Eigen::Map<VecX_T<catamari::Int>>(ordering.inverse_permutation.Data(), A_reduced->m) = iperm_downcast.template cast<catamari::Int>();
#else
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_nested_dissection");
                catamari::Buffer<SuiteSparse_long> CParent(A_reduced->m), CMember(A_reduced->m);
                cholmod_l_nested_dissection(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                            (SuiteSparse_long *) ordering.inverse_permutation.Data(),
                                            CParent.Data(), CMember.Data(), m_c.get());
                quotient::InvertPermutation(ordering.inverse_permutation, &ordering.permutation);
#endif
            }
            else if (actualOrderingMethod == OrderingMethod::Metis) {
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_metis");
                cholmod_l_metis(&cholmat, /* fset = */ nullptr, /* fsize = */ 0, /* postorder = */ true,
                                (SuiteSparse_long *) ordering.inverse_permutation.Data(), m_c.get());
                quotient::InvertPermutation(ordering.inverse_permutation, &ordering.permutation);
            }
            else if (actualOrderingMethod == OrderingMethod::AMD) {
                BENCHMARK_SCOPED_TIMER_SECTION t("AMD ordering");
                using ordering_index_type = int32_t;
                const ordering_index_type n = A_reduced->m;

                // AMD_2 is passed only the off-diagonal entries of the *full* matrix (i.e., both upper and lower triangles).
                // Furthermore, it needs some additional "elbow room" in the
                // the row index array (the `cholmod_amd` wrapper allocates around 50%).
                const auto &A = m_catamariConverter->get();
                ordering_index_type padded_input_matrix_size = (A.NumEntries() - n) * 1.5;

                VecX_T<ordering_index_type> Pe(n + 1), Nv(n), workspace;
                workspace.resize(7 * n + padded_input_matrix_size);

                ordering_index_type *Degree = workspace.data(),
                                    *Wi     = workspace.data() + n,
                                    *Len    = workspace.data() + 2 * n,
                                    *Elen   = workspace.data() + 3 * n,
                                    *Head   = workspace.data() + 4 * n,
                                    * perm  = workspace.data() + 5 * n,
                                    *iperm  = workspace.data() + 6 * n,
                                    *Iw     = workspace.data() + 7 * n; // length `padded_input_matrix_size`

                tbb::parallel_for(tbb::blocked_range<ordering_index_type>(0, n), [&](const tbb::blocked_range<ordering_index_type> &r) {
                    for (ordering_index_type j = r.begin(); j < r.end(); ++j) {
                        auto col_start = A.RowEntryOffset(j);
                        auto col_end   = A.RowEntryOffset(j + 1);
                        Len[j] = ordering_index_type(col_end - col_start - 1); // diagonal is excluded
                        Pe[j] = col_start - j; // diagonal is excluded
                        ordering_index_type back = Pe[j];
                        for (ordering_index_type ii = col_start; ii < col_end; ++ii) {
                            ordering_index_type i = A.Entry(ii).column; // Catamari matrix is transposed!
                            if (i != j) Iw[back++] = i;
                        }
                    }
                });
                Pe[n] = Pe[n - 1] + Len[n - 1];

                {
                    double *Control = nullptr; // Use AMD defaults.
                    double Info [AMD_INFO];
                    BENCHMARK_SCOPED_TIMER_SECTION t2("amd_2");
                    // amd_l2(n, Pe.data(), Iw, Len, padded_input_matrix_size, Pe[n], Nv.data(), iperm, perm, Head, Elen,
                    //        Degree, Wi, Control, Info);
                    amd_2(n, Pe.data(), Iw, Len, padded_input_matrix_size, Pe[n], Nv.data(), iperm, perm, Head, Elen,
                          Degree, Wi, Control, Info);

                }

                ordering.permutation.Resize(n);
                tbb::parallel_for(tbb::blocked_range<ordering_index_type>(0, n), [&](const tbb::blocked_range<ordering_index_type> &r) {
                    for (ordering_index_type j = r.begin(); j < r.end(); ++j) {
                        // Note that SuiteSparse and Catamari disagree on which
                        // permutation they call the "inverse" one.
                        // In Catamari, `permutation[j_orig]` gives the
                        // column index in the permuted matrix where column
                        // `j_orig` of the original matrix ends up.
                        ordering.inverse_permutation[j] =  perm[j];
                        ordering.permutation[j]         = iperm[j];
                    }
                });

#if 1
                // Extract preliminary supernode information and assembly tree from
                // the AMD output; this is needed to parallelize symbolic
                // factorization in Catamari.
                // (Note that the assembly tree created by AMD
                // is generally different from the supernodal assembly tree
                // consisting of fundamental supernodes).
                {
                    using catamari::Int;

                    // Record which supernode contains column `j` of L.
                    // Note that only the entries corresponding to the
                    // "representative column" of each supernode are populated.
                    // By this, we mean the root of the "subtree" within
                    // each node of the assembly tree.
                    // In terms of the AMD output, these are the indices for
                    // which `Nv` is nonzero, and are the *last* column indices
                    // of each supernode.
                    VecX_T<Int> supernode_index(n);

                    // Determine supernodes and sizes
                    Int num_supernodes = (Nv.array() > 0).count();
                    ordering.supernode_sizes.Resize(num_supernodes);
                    for (Int s = 0, j_perm = 0; j_perm < Int(n); ++j_perm) { // Loop through columns of L
                        Int size = Nv[ordering.inverse_permutation[j_perm]];
                        if (size > 0) {
                            supernode_index[j_perm] = s;
                            ordering.supernode_sizes[s++] = size;
                        }
                    }

                    OffsetScan(ordering.supernode_sizes, &ordering.supernode_offsets);

                    // Convert the assembly tree from AMD's `Pe` array into `ordering.assembly_forest.parents`.
                    // Note that, when `j` is the start of a supernode, `Pe[j]`
                    // holds the parent index of column `j` where all indices
                    // here are in the *original* matrix.
                    ordering.assembly_forest.parents.Resize(num_supernodes);
                    for (Int s = 0; s < num_supernodes; ++s) {
                        // Note: the "representative column" is the *last* column of the supernode.
                        Int representative_col = ordering.inverse_permutation[ordering.supernode_offsets[s + 1] - 1];
                        assert(Nv[representative_col] > 0 && "Failed to find supernode's 'representative'/'root' column");
                        Int parent_repcol_orig = Pe[representative_col];
                        if (parent_repcol_orig < 0) { ordering.assembly_forest.parents[s] = -1; continue; }
                        Int parent_representative_col = ordering.permutation[parent_repcol_orig];
                        assert(Nv[ordering.inverse_permutation[parent_representative_col]] > 0 && "Pe did not return a representative col.");
                        ordering.assembly_forest.parents[s] = supernode_index[parent_representative_col];
                    }

                    ordering.assembly_forest.FillFromParents();
                }
#endif
#if 0
                {
                    std::cout << "Nv: " << Nv.head(20).transpose() << std::endl;
                    std::cout << "Pe: " << Pe.head(20).transpose() << std::endl;

                    // Pe[j] and Nv[j] hold data for the original column `j`.
                    // We need arrays that are permuted to correspond to the
                    // lower factor L.
                    VecX_T<ordering_index_type> Pe_perm(n + 1), Nv_perm(n);
                    for (ordering_index_type j = 0; j < n; ++j) {
                        ordering_index_type j_orig = ordering.inverse_permutation[j];
                        Pe_perm[j] = ordering.permutation[Pe[j_orig]];
                        Nv_perm[j] = Nv[j_orig];
                    }

                    std::cout << "n: " << n << std::endl;
                    int argmin;
                    std::cout << "Nv.min(): " << Nv_perm.minCoeff(&argmin) << std::endl;
                    std::cout << "Nv.max(): " << Nv_perm.maxCoeff() << std::endl;
                    std::cout << "Nv.sum(): " << Nv_perm.sum() << std::endl;
                    std::cout << "Permuted Nv: " << Nv_perm.segment(0, 40).transpose() << std::endl;
                    std::cout << "Permuted Pe: " << Pe_perm.segment(0, 40).transpose() << std::endl;

                    std::ofstream("Nv_perm.txt") << Nv_perm << std::endl;
                    std::ofstream("Pe_perm.txt") << Pe_perm << std::endl;
                }
#endif
            }
            else throw std::runtime_error("Unknown orderingMethod");
        }
        m_ldl->Factor(m_catamariConverter->get(), ordering, *m_ldlControl, /* symbolic_only = */ true);

        double sym_fact_duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - sym_fact_start).count();
        if (orderingMethod == OrderingMethod::Adaptive)
            adaptiveOrdering.recordSymbolic(sym_fact_duration);
    }
    else if (orderingMethod == OrderingMethod::Scotch) {
#if MESHFEM_WITH_SCOTCH
        catamari::SymmetricOrdering ordering;
        ordering.permutation        .Resize(A_reduced->m);
        ordering.inverse_permutation.Resize(A_reduced->m);

        Eigen::Map<VecX_T<SuiteSparse_long>> perm(ordering.permutation.Data(), A_reduced->m);
        Eigen::Map<VecX_T<SuiteSparse_long>> iperm(ordering.inverse_permutation.Data(), A_reduced->m);

        scotch_ordering(*A_reduced, perm, iperm, scotchSettings.stratFlag, scotchSettings.imbalanceRatio);

        m_ldl->Factor(m_catamariConverter->get(), ordering, *m_ldlControl, /* symbolic_only = */ true);
#else
        throw std::runtime_error("Scotch support not compiled in");
#endif
    }
    else throw std::runtime_error("Unknown orderingMethod");

    std::unique_ptr<catamari::SparseLDL<double>> ldl_block;
    if (m_blockSize > 1) {
        // Currently we must expand the symbolic factorization to a scalar one.
        // TODO: once a full "block factorization type" is supported,
        // we can omit this conversion.
        ldl_block = std::move(m_ldl);
        m_ldl = ldl_block->ExpandSymbolicFactorizationToScalar(m_blockSize);
    }

    if (!m_legacy) {
        // Build a conversion plan to support direct injection of values
        // into the Cholesky factor. This must be done specially for non-unit
        // block sizes.
        BENCHMARK_SCOPED_TIMER_SECTION t2("ConversionPlan");
        if (m_blockSize > 1) {
            assert(ldl_block);
            const BlockCSCHessianBase &bmat = static_cast<const BlockCSCHessianBase &>(mat);
            if (bmat.hasContiguousBlocks())
                m_catamariConverter->conversionPlan = catamari_conversion_plan::constructBlockConversionPlan(m_catamariConverter->get(), m_blockSize, *m_ldl, *ldl_block, m_catamariConverter->m_sourceReducedEntryForFullMatrixEntry, blockEntryForReducedBlockEntry);
            else
                m_catamariConverter->conversionPlan = catamari_conversion_plan::constructScalarConversionPlan(m_catamariConverter->get(), mat, reducedRowForRow_block, m_blockSize, *m_ldl, *ldl_block, m_catamariConverter->m_sourceReducedEntryForFullMatrixEntry, blockEntryForReducedBlockEntry);
            // auto cp_compare = catamari_conversion_plan::constructConversionPlan(m_catamariConverter->get(), *ldl_block, m_catamariConverter->m_sourceReducedEntryForFullMatrixEntry, blockEntryForReducedBlockEntry);
        }
        else m_catamariConverter->conversionPlan = catamari_conversion_plan::constructConversionPlan(m_catamariConverter->get(), *m_ldl, m_catamariConverter->m_sourceReducedEntryForFullMatrixEntry, m_entryForReducedEntry, m_dataOffsetForScalarHessianLoc);

#if 0
        // Validation
        {
            BENCHMARK_SCOPED_TIMER_SECTION tv("Conversion plan validate");
            SuiteSparseMatrix A_scalar = expandSparsityPattern<>(mat, m_blockSize);
            std::vector<SuiteSparse_long> reducedRowForRow_scalar;
            std::vector<SuiteSparse_long> entryForReducedEntry_scalar;

            SuiteSparseMatrix A_scalar_reduced = A_scalar;
            std::vector<bool> scalarFixedVarMask(A_scalar.n, false);
            for (size_t i : pinnedVars) scalarFixedVarMask[i] = true;
            A_scalar_reduced.rowColRemoval([&](SuiteSparse_long i) { return scalarFixedVarMask[i]; }, &reducedRowForRow_scalar, &entryForReducedEntry_scalar);
            catamari_conversion_plan::validate(m_catamariConverter->conversionPlan, *m_ldl, A_scalar, reducedRowForRow_scalar, A_scalar_reduced.m);
        }
#endif

        {
            BENCHMARK_SCOPED_TIMER_SECTION t("Cleanup");
            ldl_block.reset();
            m_catamariConverter->freeCatamariMatrix();
        }
    }
    m_factorizationType = FactorizationType::Symbolic;
}

void CatamariFactorizer::factorizeNumeric(const SuiteSparseMatrix &A, bool /* isInTryCatch */) {
    m_numericFactorizationImpl(A);
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool /* isInTryCatch */) {
    m_numericFactorizationImpl(A, sigma, B.Ax.data());
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool /* isInTryCatch */) {
    m_numericFactorizationImpl(A, sigma, nullptr);
}

template<typename... Args>
void CatamariFactorizer::m_numericFactorizationImpl(const SuiteSparseMatrix &A, Args&&... args) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Numeric Factorize");
    assertFactorization(FactorizationType::Symbolic);

    auto num_fact_start = std::chrono::high_resolution_clock::now();

    catamari::SparseLDLResult<double> result;
    // TODO: account for m_dataOffsetForScalarHessianLoc in legacy mode.
    if (m_legacy) result = m_ldl->RefactorWithFixedSparsityPattern(m_catamariConverter->          convert(A.Ax.data(), std::forward<Args>(args)...));
    else          result = m_ldl->RefactorWithFixedSparsityPattern(m_catamariConverter->conversionPlan, m_useBlockAccel ? m_blockSize : 1, A.Ax.data(), std::forward<Args>(args)...);

    if constexpr (false) {
        static bool first = true;
        if (first) {
            using catamari::Int;
            auto &lf = m_ldl->supernodal_factorization->lower_factor_;
            const Int num_supernodes = m_ldl->supernodal_factorization->ordering_.supernode_sizes.Size();
            std::cout << "Lower factor structure size (total degree): " << lf->StructureEnd(num_supernodes - 1) - lf->StructureBeg(0) << std::endl;

            if (!m_legacy) {
                std::cout << "Factor data size: " << m_ldl->supernodal_factorization->factor_values_.Height() << std::endl;
                std::cout << "Catamari converter size: " << m_ldl->supernodal_factorization->m_inputData.cplan->size() << std::endl;
            }
            first = false;
        }
    }

#if CATAMARI_FINEGRAINED_TIMERS
    if (m_ldlControl->supernodal_control.algorithm == catamari::kRightLookingLDL) {
        static std::string directory = "catamari_timers";
        static size_t counter = 0;
        if (counter == 0) {
            // Get a unique directory name.
            size_t id = 0;
            while (std::filesystem::exists(directory)) directory = "catamari_timers_" + std::to_string(id++);
            std::filesystem::create_directory(directory);

            std::cout << "Writing Catamari timers to " << directory << std::endl;
            std::cout << "To disable, set CATAMARI_FINEGRAINED_TIMERS to 0" << std::endl;
        }
        std::string dirname = directory + "/" + std::to_string(counter++);
        std::filesystem::create_directory(dirname);
        m_ldl->supernodal_factorization->WriteFinegrainedTimerStats(dirname);
        m_ldl->supernodal_factorization->WriteSupernodeStats(dirname);
    }
#endif

    double num_fact_duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - num_fact_start).count();
    if (orderingMethod == OrderingMethod::Adaptive)
        adaptiveOrdering.recordNumeric(num_fact_duration);

    if (size_t(result.num_successful_pivots) != n_reduced()) {
        m_factorizationType = FactorizationType::Symbolic;
        throw std::runtime_error(std::to_string(result.num_successful_pivots) + "/" +
                                 std::to_string(n_reduced()) + "  pivots successful in Catamari numeric factorization (non-positive definite?)");
    }
    m_factorizationType = FactorizationType::Numeric;
}

size_t CatamariFactorizer::getFactorNNZ() const {
    assertFactorization(FactorizationType::Symbolic);
    return m_ldl->supernodal_factorization->GetFactorNNZ();
}

double CatamariFactorizer::getFlopEstimate() const {
    assertFactorization(FactorizationType::Symbolic);
    return m_ldl->supernodal_factorization->EstimateTotalWork();
}

void CatamariFactorizer::writeSolveTimers() const {
#if CATAMARI_FINEGRAINED_TIMERS
    static std::string directory = "catamari_solve_timers";
    static size_t counter = 0;
    if (counter == 0) {
        // Get a unique directory name.
        size_t id = 0;
        while (std::filesystem::exists(directory)) directory = "catamari_solve_timers_" + std::to_string(id++);
        std::filesystem::create_directory(directory);

        std::cout << "Writing Catamari solve timers to " << directory << std::endl;
        std::cout << "To disable, set CATAMARI_FINEGRAINED_TIMERS to 0" << std::endl;
    }
    std::string dirname = directory + "/" + std::to_string(counter++);
    std::filesystem::create_directory(dirname);
    m_ldl->supernodal_factorization->WriteFinegrainedSolveTimerStats(dirname);
    m_ldl->supernodal_factorization->WriteSupernodeStats(dirname);
    m_ldl->supernodal_factorization->ResetFinegrainedSolveTimerStats();
#endif
}

// Raw pointer version (Use with care! Caller must allocate/own both pointers)
void CatamariFactorizer::solveRawReduced(const Real *b, Real *x, CholeskySys sys, bool alreadyPermuted) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariFactorizer.solveRawReduced");
    const size_t s = m_reduced();
    if (alreadyPermuted) {
        BENCHMARK_SCOPED_TIMER_SECTION timer2("copy " + std::to_string(s) + " entries");
        // Eigen::Map<Eigen::VectorXd>(x, s) = Eigen::Map<const Eigen::VectorXd>(b, s);
        copyParallel(m_reduced(), b, x);

        solveRawReducedInPlace(x, sys, alreadyPermuted);
    }
    else {
        // Avoid extra copy step by permuting into the scratch RHS
        if (size_t(m_permuted_rhs_scratch.size()) < s)
            m_permuted_rhs_scratch.resize(s);

        catamari::BlasMatrixView<double> v_perm;
        v_perm.height = s;
        v_perm.width = 1;
        v_perm.leading_dim = s;
        v_perm.data = m_permuted_rhs_scratch.data();

        catamari::BlasMatrixView<double> v = v_perm;
        v.data = const_cast<Real *>(b);

        auto f = m_ldl->supernodal_factorization.get();
        if (f == nullptr) throw std::runtime_error("solveRawReduced: only supernodal factorizations are supported");
        InversePermute(f->ordering_.inverse_permutation, v, &v_perm); // Note: InversePermute is faster than Permute due to contiguous writes avoiding false sharing.

        {
            BENCHMARK_SCOPED_TIMER_SECTION solveTimer("Catamari Solve");
            m_ldl->Solve(&v_perm, /* alreadyPermuted = */ true);
        }

        catamari::BlasMatrixView<double> v_x = v_perm;
        v_x.data = x;

        InversePermute(f->ordering_.permutation, v_perm, &v_x);
    }
}

void CatamariFactorizer::solveRawReducedInPlace(Real *bx, CholeskySys sys, bool alreadyPermuted) const {
    assertFactorization(sys);
    if (sys != CholeskySys::A) {
        std::cout << "Alternative CholeskySys not yet wrapped for Catamari" << std::endl;
        throw std::runtime_error("Alternative CholeskySys not yet wrapped for Catamari");
    }

    catamari::BlasMatrixView<double> v;
    const size_t s = m_reduced();
    v.height = s;
    v.width = 1;
    v.leading_dim = s;
    v.data = bx;

    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Solve");
    m_ldl->Solve(&v, alreadyPermuted);
}

void CatamariFactorizer::solveMultiRHS(const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &B, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &X) const {
    BENCHMARK_SCOPED_TIMER_SECTION otimer("solveMultiRHS");
    if (size_t(B.rows()) != m()) throw std::runtime_error("Incorrect RHS size");
    const size_t nrhs = B.cols();
    if (nrhs < 1) throw std::runtime_error("Must specify at least one rhs.");

    catamari::BlasMatrixView<double> v;
    const size_t s = m_reduced();
    v.height = s;
    v.width = nrhs;
    v.leading_dim = s;

    if (hasFixedVars()) {
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> X_scratch;
        removeFixedEntries(B, X_scratch, /* permute = */ true);

#if 1
        v.data = X_scratch.data();

        {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Solve");
            m_ldl->Solve(&v, /* alreadyPermuted = */ true);
        }
#else
        v.width = 1;
        for (size_t i = 0; i < nrhs; ++i) {
            v.data = X_scratch.col(i).data();
            {
                BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Solve");
                m_ldl->Solve(&v, /* alreadyPermuted = */ true);
            }
        }
#endif

        extractFullSolution(X_scratch, X, /* permute = */ true);
    }
    else {
        X = B;
        v.data = X.data();
        BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Solve");
        m_ldl->Solve(&v, /* alreadyPermuted = */ false);
    }
}

void CatamariFactorizer::m_populatePermutedReducedRowForRow() const {
    const size_t n_full = n();
    if (m_reducedRowForRow.size() != n_full) throw std::runtime_error("Incorrect m_reducedRowForRow size");
    if (m_permutedReducedRowForRow.size() == n_full) return;
    auto f = m_ldl->supernodal_factorization.get();

    if (f == nullptr) throw std::runtime_error("Only supernodal factorizations are supported");
    const auto &o = f->ordering_;

    m_permutedReducedRowForRow.resize(n_full);
    for (size_t i = 0; i < n_full; ++i) {
        SuiteSparse_long row_orig = m_reducedRowForRow[i];
        m_permutedReducedRowForRow[i] = (row_orig != SuiteSparseMatrix::INDEX_NONE)
                                            ? o.permutation[row_orig] : row_orig;
    }
}

// Stashing support
void CatamariFactorizer::       stashFactorization()       { m_ldlStash = m_ldl->Clone(); }
bool CatamariFactorizer::  hasStashedFactorization() const { return bool(m_ldlStash); }
void CatamariFactorizer:: swapStashedFactorization()       { if (!hasStashedFactorization()) { throw std::runtime_error("No stashed factorization"); } std::swap(m_ldl, m_ldlStash); }
void CatamariFactorizer::clearStashedFactorization()       { m_ldlStash.reset(); }

CatamariFactorizer::~CatamariFactorizer() {
    if (m_c) cholmod_l_finish(m_c.get());
    if (m_c_int) cholmod_finish(m_c_int.get());
}

#endif
