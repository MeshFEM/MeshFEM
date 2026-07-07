#ifndef CATAMARICONVERTER_HH
#define CATAMARICONVERTER_HH

#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include <MeshFEMSparse/SparseMatrices.hh>

namespace MeshFEM {

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

    // Note `Asp_in` is the rowcol-reduced *block* sparsity pattern.
    // If `blockSize > 1`, then `Asp_in` will be expanded from a "block sparsity
    // pattern" to a "scalar sparsity pattern" before converting.
    // We retain this to support legacy-Catamari mode, but for best efficiency,
    // the caller should pass `blockSize = 1` and interpret the converter's
    // entries as representing blocks of the appropriate size.
    CatamariConverter(const SuiteSparseMatrix &Asp_in, const size_t blockSize, bool legacy, const std::vector<SuiteSparse_long> &entryForReducedEntry)
        : m_legacy(legacy)
    {
#ifdef MESHFEM_USE_LEGACY_CATAMARI
        if (!m_legacy) throw std::runtime_error("legacy must be true if MESHFEM_USE_LEGACY_CATAMARI is set");
#endif
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter");
        if (Asp_in.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");
        if (Asp_in.m != Asp_in.n) throw std::runtime_error("Only square matrices are supported");

        const SuiteSparseMatrix *Asp_ptr = &Asp_in;
        SuiteSparseMatrix A_scalar;
        if (blockSize > 1) {
            BENCHMARK_SCOPED_TIMER_SECTION t2("Expand Sparsity Pattern");
            A_scalar = expandSparsityPattern<>(Asp_in, blockSize); Asp_ptr = &A_scalar;
        }
        const SuiteSparseMatrix &Asp = *Asp_ptr;

        // Convert upper-triangle sparsity pattern to a full symmetric sparsity
        // pattern in Catamari format.
        m_result.Resize(Asp_ptr->n, Asp_ptr->n);

        {
            // TODO: generalized version of `CSCMat::InOrderBuilder` that
            // enables us to build the catamari::CoordinateMatrix in-place...

            // Get an integer-valued sparse matrix where each entry holds the
            // index of the source upper triangle entry that generated it.
            CSCMatrix<SuiteSparse_long, SuiteSparse_long> A_full;

            {
                BENCHMARK_SCOPED_TIMER_SECTION t2("toSymmetryMode");
                A_full = Asp_ptr->toSymmetryModeImpl<SuiteSparse_long>(SuiteSparseMatrix::SymmetryMode::NONE, [](size_t ii) { return ii; });
            }

            catamari::Buffer<catamari::MatrixEntry<typename SuiteSparseMatrix::value_type>> new_entries;
            {
                BENCHMARK_SCOPED_TIMER_SECTION t2("Entry Generation");
                new_entries.Resize(A_full.nz);
                parallel_for_range(A_full.n, [&](size_t j) {
                    for (SuiteSparse_long ii = A_full.Ap[j]; ii < A_full.Ap[j + 1]; ++ii) {
                        SuiteSparse_long i = A_full.Ai[ii];
                        new_entries[ii].row = j; // transpose: Catamari uses CSR storage
                        new_entries[ii].column = i;
                        // new_entries[ii].value = 1; // Value won't be referenced...
                    }
                }, /* grain_size = */ 64, /* parallelism_threshold = */ 128);
            }


#ifdef MESHFEM_USE_LEGACY_CATAMARI
            {
                BENCHMARK_SCOPED_TIMER_SECTION t2("Legacy AddEntries");
                m_result.AddEntries(std::move(new_entries));
            }
#else
            catamari::Buffer<catamari::Int> row_entry_offsets(A_full.Ap.size());
            for (size_t i = 0; i < A_full.Ap.size(); ++i)
                row_entry_offsets[i] = A_full.Ap[i];
            m_result.SetSortedEntries(std::move(new_entries), std::move(row_entry_offsets));
#endif

            m_sourceReducedEntryForFullMatrixEntry = std::move(A_full.Ax);
        }

        if (legacy) {
            BENCHMARK_SCOPED_TIMER_SECTION t2("Legacy source lookup");
            // // Determine where to find each entry of the Catamari matrix `m_result`
            // // within the scalar values array of the original input matrix (pre row-col removal).
            // m_sourceLocForCatamariInputEntry.assign(m_result.NumEntries(), -1);
            // for (catamari::Int j = 0; j < Asp.n; ++j) {
            //     for (auto ii = Asp.Ap[j]; ii < Asp.Ap[j + 1]; ++ii) {
            //         catamari::Int i = Asp.Ai[ii];
            //         SuiteSparse_long loc = entryForReducedEntry.empty() ? ii : entryForReducedEntry[ii];
            //         m_sourceLocForCatamariInputEntry[m_result.EntryOffset(i, j)] = loc;
            //         if (i != j)
            //             m_sourceLocForCatamariInputEntry[m_result.EntryOffset(j, i)] = loc;
            //     }
            // }
            // for (size_t ii = 0; ii < m_sourceLocForCatamariInputEntry.size(); ++ii) {
            //     if (m_sourceLocForCatamariInputEntry[ii] < 0)
            //         throw std::runtime_error("Missing source entry for full matrix entry");
            //     auto loc = m_sourceReducedEntryForFullMatrixEntry[ii];
            //     if (!entryForReducedEntry.empty()) loc = entryForReducedEntry[loc];
            //     if (m_sourceLocForCatamariInputEntry[ii] != loc)
            //         throw std::runtime_error("Source entry lookup mismatch");
            // }
            m_sourceLocForCatamariInputEntry.resize(m_result.NumEntries());
            for (int i = 0; i < m_sourceLocForCatamariInputEntry.size(); ++i) {
                auto loc = m_sourceReducedEntryForFullMatrixEntry[i];
                if (!entryForReducedEntry.empty()) loc = entryForReducedEntry[loc];
                m_sourceLocForCatamariInputEntry[i] = loc;
            }
        }
    }
    std::vector<SuiteSparse_long> m_sourceReducedEntryForFullMatrixEntry;

    // Note Ax_data is the scalar data before rowcol removal.
    // Legacy: convert and cache the numerical values of matrix `A` (assuming `A` has
    // an identical sparsity pattern to `m_Asp`).
    const CMat &convert(const double *Ax_data, const VecX_T<SuiteSparse_long> &dataOffsetForScalarHessianLoc) {
        if (!m_legacy) throw std::runtime_error("convert() is for legacy mode only!");
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter.convert");
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_result.Entries().Size()), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                SuiteSparse_long loc = m_sourceLocForCatamariInputEntry[i];
                if (dataOffsetForScalarHessianLoc.size()) loc = dataOffsetForScalarHessianLoc[loc];
                m_result.Entries()[i].value = Ax_data[loc];
            }
        });
        return m_result;
    }

    // Legacy: convert and cache the numerical values of matrix `A + sigma B` (assuming
    // `A` and `B` have identical sparsity patterns to `m_Asp`).
    // If `B_data == nullptr`, convert the values `A + sigma * I`.
    const CMat &convert(const double *Ax_data, const VecX_T<SuiteSparse_long> &dataOffsetForScalarHessianLoc, double sigma, const double *B_data) {
        if (!m_legacy) throw std::runtime_error("convertWithShift() is for legacy mode only!");
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter.convert");

        if (B_data != nullptr) {
            for (size_t i = 0; i < m_result.Entries().Size(); ++i) {
                SuiteSparse_long loc = m_sourceLocForCatamariInputEntry[i];
                if (dataOffsetForScalarHessianLoc.size()) loc = dataOffsetForScalarHessianLoc[loc];
                m_result.Entries()[i].value = Ax_data[loc] + sigma * B_data[loc];
            }
        }
        else {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_result.Entries().Size()), [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    SuiteSparse_long loc = m_sourceLocForCatamariInputEntry[i];
                    if (dataOffsetForScalarHessianLoc.size()) loc = dataOffsetForScalarHessianLoc[loc];
                    m_result.Entries()[i].value = Ax_data[loc];
                }
            });
            if (sigma != 0) { // Add the shift to the diagonal entries
                tbb::parallel_for(tbb::blocked_range<size_t>(0, m_result.NumColumns()), [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t j = r.begin(); j < r.end(); ++j)
                        m_result.Entries()[m_result.EntryOffset(j, j)].value += sigma;
                });
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

#ifndef MESHFEM_USE_LEGACY_CATAMARI
    catamari::ConversionPlan conversionPlan;
#endif

private:
    CMat m_result;

    const bool m_legacy = false;
    SuiteSparseMatrix m_Asp; // For legacy mode only
    VecX_T<catamari::Int> m_sourceLocForCatamariInputEntry; // For legacy mode only
};

} // namespace MeshFEM

#endif /* end of include guard: CATAMARICONVERTER_HH */
