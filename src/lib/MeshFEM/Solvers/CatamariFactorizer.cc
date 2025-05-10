#include "CatamariFactorizer.hh"
#include <limits>

#if MESHFEM_WITH_CATAMARI

#include "CholmodFactorizer.hh"

#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include <specify.hpp>

#if CATAMARI_FINEGRAINED_TIMERS
#include <filesystem>
#endif

// The largest block size for which we'll instantiate a BlockCatamari solver.
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
            // Get an integer-valued sparse matrix where each entry holds the
            // index of the source upper triangle entry that generated it.
            CSCMatrix<SuiteSparse_long, SuiteSparse_long> A_full = Asp_ptr->toSymmetryModeImpl<SuiteSparse_long>(SuiteSparseMatrix::SymmetryMode::NONE, [](size_t ii) { return ii; });

            catamari::Buffer<catamari::MatrixEntry<typename SuiteSparseMatrix::value_type>> new_entries(A_full.nz);
            for (SuiteSparse_long j = 0; j < A_full.n; ++j) {
                for (SuiteSparse_long ii = A_full.Ap[j]; ii < A_full.Ap[j + 1]; ++ii) {
                    SuiteSparse_long i = A_full.Ai[ii];
                    new_entries[ii].row = j; // transpose: Catamari uses CSR storage
                    new_entries[ii].column = i;
                    new_entries[ii].value = 1;
                }
            }

            m_result.SetSortedEntries(std::move(new_entries));
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

    const ConversionPlan &conversionPlan() const { return m_conversionPlan; }
    void constructConversionPlan(catamari::SparseLDL<double> &ldl, std::vector<SuiteSparse_long> &entryForReducedEntry) {
        BENCHMARK_SCOPED_TIMER_SECTION ctimer("Construct plan");
        auto f = ldl.supernodal_factorization.get();
        if (f == nullptr) throw std::runtime_error("Only supernodal factorizations are supported");

        const auto &df = f->diagonal_factor_;
        const auto &lf = f->lower_factor_;

        using Int = catamari::Int;
        auto &o  = f->ordering_;
        auto &sno = o.supernode_offsets;
        const double *f_vals = f->factor_values_.Data();
        const Int num_supernodes = o.supernode_sizes.Size();
        if (o.permutation.Empty()) throw std::runtime_error("Expected permutation");

        const Int nc = m_result.NumColumns();
        m_conversionPlan.columnOffsets.resize(nc + 1);

        // Count the lower-triangular entries *in the permuted matrix*.
        // We work with the *full* (non-triangular) matrix so that we can efficiently loop over all nonzeros in a given
        // column of the *permuted* lower factor.
        Int *columnSizes = m_conversionPlan.columnOffsets.data() + 1;
        static tbb::affinity_partitioner ap;
        tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, num_supernodes), [&](const tbb::blocked_range<catamari::Int> &r) {
            for (Int supernode = r.begin(); supernode < r.end(); ++supernode) {
                const Int supernode_end = sno[supernode + 1];
                for (Int j_perm = sno[supernode]; j_perm < supernode_end; ++j_perm) {
                    Int j_orig = o.inverse_permutation[j_perm];
                    const Int col_entries_end = m_result.RowEntryOffset(j_orig + 1);
                    Int colSize = 0;
                    for (Int ii = m_result.RowEntryOffset(j_orig); ii < col_entries_end; ++ii)
                        if (o.permutation[m_result.Entry(ii).column] >= j_perm) ++colSize; // entry in lower triangle?
                    columnSizes[j_perm] = colSize;
                }
            }
        }, ap);

        // Convert sizes to offsets and allocate conversion plan entries.
        m_conversionPlan.columnOffsets[0] = 0;
        Int *columnBacks = m_conversionPlan.columnOffsets.data() + 1; // Back indices of the (initially empty) column buckets
                                                                      // These will be incremented and eventually become the column end indices.
        {
            Int back = 0;
            for (Int i = 0; i < nc; ++i) {
                // Note: we are updating in-place (columnSizes == columnBacks)!
                Int s = columnSizes[i];
                columnBacks[i] = back;
                back += s;
            }

            m_conversionPlan.resize(back);
        }

        BENCHMARK_START_TIMER_SECTION("Build");

        // For each entry in the full (non-triangular) row-col-removed input matrix `m_result`,
        // determine whether/where its permuted instance goes in the *lower triangle* of the factorization
        // as well as which original matrix entry it originated from.
        tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, num_supernodes), [&](const tbb::blocked_range<catamari::Int> &r) {
            for (Int supernode = r.begin(); supernode < r.end(); ++supernode) {
                const Int supernode_start = sno[supernode    ];
                const Int supernode_end   = sno[supernode + 1];
                catamari::BlasMatrixView<double>& db = df->blocks[supernode];
                catamari::BlasMatrixView<double>& lb = lf->blocks[supernode];
                const Int *index_beg = lf->StructureBeg(supernode);
                const Int *index_end = lf->StructureEnd(supernode);

                for (Int j_perm = supernode_start; j_perm < supernode_end; ++j_perm) {
                    Int j_orig = o.inverse_permutation[j_perm];
                    Int columnBack = columnBacks[j_perm];

                    // Note: catamari::CoordinateMatrix is row major, hence the implicit transpose happening here...
                    const Int col_entries_begin = m_result.RowEntryOffset(j_orig);
                    const Int col_entries_end   = m_result.RowEntryOffset(j_orig + 1);
                    const Int *guess = nullptr;
                    for (Int ii = col_entries_begin; ii < col_entries_end; ++ii) {
                        const catamari::MatrixEntry<double> &e = m_result.Entry(ii);
                        Int i_perm = o.permutation[e.column];
                        if (i_perm < j_perm) continue; // Skip the strict upper triangle.

                        // Locate (i_perm, j_perm) in the supernode structure.
                        Int locForEntry; // destination location

                        const Int j_rel = j_perm - supernode_start;
                        if (i_perm < supernode_end) {
                            const Int i_rel = i_perm - supernode_start;
                            locForEntry = std::distance(f_vals, (const double *) db.Pointer(i_rel, j_rel));
                        }
                        else {
                            // Search [lf->structureBeg, lf->structureEnd) for value `i_perm`,
                            // first checking at `*guess` (which will be correct for consecutive
                            // strips of entries).
                            const Int *iter;
                            if ((guess >= index_beg) && (guess < index_end) && (*guess == i_perm)) iter = guess;
                            else {
                                iter = sb_lower_bound(index_beg, index_end, i_perm);
                                if ((iter == index_end) || (*iter != i_perm)) throw std::runtime_error("Couldn't locate row index " + std::to_string(i_perm) + " in supernode " + std::to_string(supernode) + " containing rows in [" + std::to_string(*index_beg) + ", " +  std::to_string(*index_end) + ")");
                            }
                            guess = iter + 1;

                            const Int i_rel = std::distance(index_beg, iter);
                            locForEntry = std::distance(f_vals, (const double *) lb.Pointer(i_rel, j_rel));
                        }

                        // Record which source entry should be read for `locForEntry`
                        SuiteSparse_long srcEntry = m_sourceReducedEntryForFullMatrixEntry[ii];
                        if (entryForReducedEntry.size()) srcEntry = entryForReducedEntry[srcEntry];
                        m_conversionPlan.entries()[columnBack++] = ConversionPlan::Entry{locForEntry, srcEntry};
                    }
                    columnBacks[j_perm] = columnBack;
                    // Sorting doesn't seem to help :(
                    // std::sort(m_conversionPlan.columnData(j_perm), m_conversionPlan.columnData(j_perm + 1),
                    //         [](const std::pair<Int, Int> &a, const std::pair<Int, Int> &b) { return a.dst < b.dst; });
                }
            }
        });
        BENCHMARK_STOP_TIMER_SECTION("Build");
    }

private:
    CMat m_result;
    ConversionPlan m_conversionPlan;

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
    m_recordSymbolic(mat, pinnedVars);
    // We only support uniform block sizes, and only up to
    // `MAX_INSTANTIATED_BLOCK_SIZE`; all others get converted to an ordinary scalar matrix.
    // TODO: convert to GCD block size instead? Do we have a use case for this?
    // TODO: try block reordering of nonuniform block sizes (then expand to scalar)?

    const bool blockFactorizationSupported = mat.uniformBlockSize() && (mat.maxBlockSize() <= MAX_INSTANTIATED_BLOCK_SIZE);
    if (blockFactorizationSupported) {
        m_blockSize = mat.maxBlockSize();
        m_factorizeSymbolic((const SuiteSparseMatrix &) mat, pinnedVars);
    }
    else {
        m_scalarHessian = mat.toScalar();
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
    if (m_blockSize > 1 && pinnedVars.size() > 0) {
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
                m_scalarHessian = ((const BlockCSCHessianBase &)(mat)).toScalar();
                m_blockSize = 1;
                return m_factorizeSymbolic(m_scalarHessian, pinnedVars);
            }
            // TODO: keep the partially pinned block in the sparsity pattern and
            // apply the scalar pin constraint during numeric factorization.
        }
        A_reduced = m_initRowColRemoval(mat, pinnedBlockVars);
        // `m_initRowColRemoval` has now stored the pinned block variable
        // indices, whereas `m_fixedVars` should store scalar variable indices.
        m_fixedVars.swap(scalarFixedVars);

        // Convert `m_reducedRowForRow` and `m_entryForReducedEntry` from block to scalar.
        // TODO: keep block versions around to accelerate fixed var removal
        // during solve? Also, once we implement the version where the entire
        // symbolic factorization is constructed from scalar sparsity
        // pattern, the block version of `m_entryForReducedEntry` will need to
        // be passed to `constructConversionPlan` below!!!
        // auto upgrade_block_indices = [&](std::vector<SuiteSparse_long> &indices) {
        //     std::vector<SuiteSparse_long> scalarIndices;
        //     scalarIndices.reserve(indices.size() * m_blockSize);
        //     for (size_t bi : indices) {
        //         for (size_t j = 0; j < m_blockSize; ++j)
        //             scalarIndices.push_back(bi * m_blockSize + j);
        //     }
        //     indices.swap(scalarIndices);
        // };
        // upgrade_block_indices(m_reducedRowForRow);

        // TODO: remove
        SuiteSparseMatrix A_scalar = expandSparsityPattern<>(mat, m_blockSize);
        m_reducedRowForRow.clear();
        A_scalar.rowColRemoval([&](SuiteSparse_long i) { return scalarFixedVarMask[i]; }, &m_reducedRowForRow, &m_entryForReducedEntry);
    }
    else {
        A_reduced = m_initRowColRemoval(mat, pinnedVars);
    }

    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Symbolic Factorize");
    m_catamariConverter = std::make_unique<CatamariConverter>(*A_reduced, m_blockSize, m_legacy, m_entryForReducedEntry);

    if (orderingMethod == OrderingMethod::Catamari)
        m_ldl->Factor(m_catamariConverter->get(), *m_ldlControl, /* symbolic_only = */ true);
    else if ((orderingMethod == OrderingMethod::CholmodNesdis) || (orderingMethod == OrderingMethod::Metis)) {
        if (!m_c) {
            m_c = std::make_unique<cholmod_common>();
            cholmod_l_start(m_c.get());
        }

        catamari::SymmetricOrdering ordering;
        {
            static_assert(sizeof(SuiteSparse_long) == sizeof(catamari::Int), "Mismatched integer type");
            ordering.inverse_permutation.Resize(A_reduced->m);
            catamari::Buffer<SuiteSparse_long> CParent(A_reduced->m), CMember(A_reduced->m);
            auto cholmat = cholmod_sparse_view(*A_reduced);
            // Note: the array `cholmat.x` apparently must be valid or cholmod_l_nested_dissection fails
            // (even though the Nested dissection algorithm should not be
            // looking at its entries...)
            // Presumably this is because the first step of cholmod_l_nested_dissection
            // is to convert the matrix from upper-triangular to full format.
            // In the future, we should bypass this step since we already do the
            // conversion ourselves for Catamari.
            cholmat.x = const_cast<double *>((const double *) A_reduced->Ai.data());

            if (orderingMethod == OrderingMethod::CholmodNesdis) {
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_nested_dissection");
                cholmod_l_nested_dissection(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                            (SuiteSparse_long *) ordering.inverse_permutation.Data(),
                                            CParent.Data(), CMember.Data(), m_c.get());
            }
            else {
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_metis");
                cholmod_l_metis(&cholmat, /* fset = */ nullptr, /* fsize = */ 0, /* postorder = */ true,
                                (SuiteSparse_long *) ordering.inverse_permutation.Data(), m_c.get());
            }
            quotient::InvertPermutation(ordering.inverse_permutation, &ordering.permutation);

            if (m_blockSize > 1) {
                // "Upgrade" the block permutation to a scalar permutation.
                auto upgrade_permutation = [&](auto &perm) {
                    std::decay_t<decltype(perm)> scalarPermutation(m_blockSize * A_reduced->n);
                    for (size_t i = 0; i < size_t(A_reduced->m); ++i) {
                        for (size_t j = 0; j < m_blockSize; ++j)
                            scalarPermutation[m_blockSize * i + j] = perm[i] * m_blockSize + j;
                    }
                    perm = std::move(scalarPermutation);
                };
                upgrade_permutation(ordering.permutation);
                upgrade_permutation(ordering.inverse_permutation);
            }
        }
        m_ldl->Factor(m_catamariConverter->get(), ordering, *m_ldlControl, /* symbolic_only = */ true);
    }
    else throw std::runtime_error("Unknown orderingMethod");

    if (!m_legacy) {
        // Build a conversion plan to support direct injection of entries.
        m_catamariConverter->constructConversionPlan(*m_ldl, m_entryForReducedEntry);
        m_catamariConverter->freeCatamariMatrix();
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
    catamari::SparseLDLResult<double> result;
    if (m_legacy) result = m_ldl->RefactorWithFixedSparsityPattern(m_catamariConverter->          convert(A.Ax.data(), std::forward<Args>(args)...));
    else          result = m_ldl->RefactorWithFixedSparsityPattern(m_catamariConverter->conversionPlan(), m_useBlockAccel ? m_blockSize : 1, A.Ax.data(), std::forward<Args>(args)...);

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

    if (size_t(result.num_successful_pivots) != n_reduced()) {
        m_factorizationType = FactorizationType::Symbolic;
        throw std::runtime_error(std::to_string(result.num_successful_pivots) + "/" +
                                 std::to_string(n_reduced()) + "  pivots successful in Catamari numeric factorization (non-positive definite?)");
    }
    m_factorizationType = FactorizationType::Numeric;
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
}

#endif
