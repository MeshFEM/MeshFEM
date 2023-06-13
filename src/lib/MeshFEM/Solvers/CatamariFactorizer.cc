#include <MeshFEM/SparseMatrices.hh>
#include "MeshFEM/Solvers/CholeskyFactorizerBase.hh"

#if MESHFEM_WITH_CATAMARI

extern "C" {
#include <cholmod.h>
}

#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include <specify.hpp>
#include <atomic>

#define DUMP_MATRICES 0

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

    CatamariConverter(const SuiteSparseMatrix &Asp) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter");
        if (Asp.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");
        if (Asp.m != Asp.n) throw std::runtime_error("Only square matrices are supported");

        // Convert upper triangle sparsity pattern to a full symmetric sparsity
        // pattern in Catamari format.
        m_result.Resize(Asp.m, Asp.n);

        {
            // Get an integer-valued sparse matrix where each entry holds the
            // index of the source upper triangle entry that generated it.
            CSCMatrix<SuiteSparse_long, SuiteSparse_long> A_full = Asp.toSymmetryModeImpl<SuiteSparse_long>(SuiteSparseMatrix::SymmetryMode::NONE, [](size_t ii) { return ii; });

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
    }
    std::vector<SuiteSparse_long> m_sourceReducedEntryForFullMatrixEntry;

    // Achieve the same result as
    // `catamari::supernodal_ldl::InitializeBlockColumn` for sparse matrix `A`
    // or `A + sigma B` (with B of identical sparsity pattern to `A`) after
    // possibly converting `A` and `B` into "reduced" versions by removing rows
    // and columns corresponding to pinned vars.
    void injectEntries(const catamari::SparseLDL<double> &ldl, const SuiteSparseMatrix &A, double sigma = 0.0, const SuiteSparseMatrix *B_optional = nullptr) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Inject entries");
        auto f = ldl.supernodal_factorization.get();
        if (f == nullptr) throw std::runtime_error("Only supernodal factorizations are supported");
        if (A.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");

        auto &df = f->diagonal_factor_;
        auto &lf = f->lower_factor_;
        double *f_vals = f->factor_values_.Data();

        using Int = catamari::Int;
        auto &o   = f->ordering_;
        auto &sno = o.supernode_offsets;
        const Int num_supernodes = o.supernode_sizes.Size();

        static tbb::affinity_partitioner ap;

        if (m_conversionPlan.empty()) throw std::runtime_error("Conversion plan was not constructed");

        size_t nthreads = get_max_num_tbb_threads();
        {
            BENCHMARK_SCOPED_TIMER_SECTION ztimer("zero and copy");
            if (B_optional == nullptr || sigma == 0) {
                const double *Ax = A.Ax.data();
                tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, num_supernodes),
                      [&](const tbb::blocked_range<catamari::Int> &r) {
                           for (Int supernode = r.begin(); supernode < r.end(); ++supernode) {
                                auto &db = df->blocks[supernode];
                                auto &lb = lf->blocks[supernode];
                                catamari::BlasMatrixView<double> front;
                                front.data = db.Data();
                                front.height = front.leading_dim = db.leading_dim;
                                front.width = db.width;
                                const Int supernode_start = sno[supernode];
                                const Int supernode_end   = sno[supernode + 1];
                                for (Int j_perm = supernode_start; j_perm < supernode_end; ++j_perm) {
                                    Int j_rel = j_perm - supernode_start;
                                    const Int outSize = front.height - j_rel;
                                    double *outPtr = front.Pointer(j_rel, j_rel);

                                    const auto cpBegin = m_conversionPlan.columnData(j_perm);
                                    const auto cpEnd   = m_conversionPlan.columnData(j_perm + 1);

#if 0 // Single-pass version (apparently slower than using Eigen, and requires sorting)
                                    double *endOutPtr = outPtr + outSize;
                                    for (auto it = cpBegin; it != cpEnd; ++it) {
                                        double *dst = f_vals + it->first;
                                        while (outPtr < dst) *outPtr++ = 0.0;
                                        *outPtr++ = Ax[it->second];
                                    }
                                    while (outPtr < endOutPtr) *outPtr++ = 0.0;
#else
                                    catamari::eigenMap(outPtr, outSize).setZero();
                                    for (auto it = cpBegin; it != cpEnd; ++it)
                                        f_vals[it->dst] = Ax[it->src];
#endif
                                }
                           }
                        }, ap);
            }
            else {
                // Factorize with shift.
                const auto &B = *B_optional;
                SuiteSparse_long nc = A.m;
                if ((B.m != nc) || (B.n != nc)) throw std::runtime_error("Unexpected input shape(s)");
                if (B.Ai.size() != A.Ai.size()) throw std::runtime_error("B must have the same sparsity pattern as A");
                const double *Ax = A.Ax.data();
                tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, num_supernodes),
                      [&](const tbb::blocked_range<catamari::Int> &r) {
                           for (Int supernode = r.begin(); supernode < r.end(); ++supernode) {
                                auto &db = df->blocks[supernode];
                                auto &lb = lf->blocks[supernode];
                                catamari::BlasMatrixView<double> front;
                                front.data = db.Data();
                                front.height = front.leading_dim = db.leading_dim;
                                front.width = db.width;
                                const Int supernode_start = sno[supernode];
                                const Int supernode_end   = sno[supernode + 1];
                                for (Int j_perm = supernode_start; j_perm < supernode_end; ++j_perm) {
                                    Int j_rel = j_perm - supernode_start;
                                    catamari::eigenMap(front.Pointer(j_rel, j_rel), front.height - j_rel).setZero();

                                    const auto cpBegin = m_conversionPlan.columnData(j_perm);
                                    const auto cpEnd   = m_conversionPlan.columnData(j_perm + 1);
                                    for (auto it = cpBegin; it != cpEnd; ++it)
                                        f_vals[it->dst] = Ax[it->src];
                                }
                           }
                        }, ap);
            }
        }
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
                                iter = std::lower_bound(index_beg, index_end, i_perm);
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
};

CatamariFactorizer::CatamariFactorizer() {
    m_ldl        = std::make_unique<catamari::SparseLDL<double>>();
    m_ldlControl = std::make_unique<catamari::SparseLDLControl<double>>();
    m_ldlControl->SetFactorizationType(catamari::kCholeskyFactorization);
    m_ldlControl->supernodal_strategy = catamari::kSupernodalFactorization;
    m_ldlControl->supernodal_control.algorithm = catamari::kRightLookingLDL;
    m_ldlControl->supernodal_control.relaxation_control.relax_supernodes = false; // Setting this to true seems faster on 5950X, slower on Apple Silicon
}

size_t CatamariFactorizer::m_reduced() const { assertFactorization(FactorizationType::Symbolic); return m_ldl->NumRows(); }
size_t CatamariFactorizer::n_reduced() const { assertFactorization(FactorizationType::Symbolic); return m_ldl->NumRows(); }

void CatamariFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    const SuiteSparseMatrix *A_reduced = m_initRowColRemoval(mat, pinnedVars);
#if DUMP_MATRICES
    {
        static size_t i = 0;
        size_t n_zero = 4;
        std::string padded_num = std::to_string(i++);
        padded_num = std::string(n_zero - std::min(n_zero, padded_num.length()), '0') + padded_num;
        A_reduced->dumpBinary("symbolic_mat_" + padded_num + ".bin");
    }
#endif
    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Symbolic Factorize");
    m_catamariConverter = std::make_unique<CatamariConverter>(*A_reduced);

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
        }
        m_ldl->Factor(m_catamariConverter->get(), ordering, *m_ldlControl, /* symbolic_only = */ true);
    }
    else throw std::runtime_error("Unknown orderingMethod");

    m_catamariConverter->constructConversionPlan(*m_ldl, m_entryForReducedEntry);
    m_catamariConverter->freeCatamariMatrix();
    m_factorizationType = FactorizationType::Symbolic;
}

void CatamariFactorizer::factorizeNumeric(const SuiteSparseMatrix &mat, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
#if DUMP_MATRICES
    {
        const SuiteSparseMatrix *A = m_rowColRemoval(mat);
        static size_t i = 0;
        size_t n_zero = 4;
        std::string padded_num = std::to_string(i++);
        padded_num = std::string(n_zero - std::min(n_zero, padded_num.length()), '0') + padded_num;
        A->dumpBinary("numeric_mat_" + padded_num + ".bin");
    }
#endif

    auto result = m_ldl->RefactorWithFixedSparsityPattern(m_catamariConverter->conversionPlan(), mat.Ax.data());
    if (size_t(result.num_successful_pivots) != n_reduced()) {
        m_factorizationType = FactorizationType::Symbolic;
        throw std::runtime_error(std::to_string(result.num_successful_pivots) + "/" +
                                 std::to_string(n_reduced()) + "  pivots successful in Catamari numeric factorization (non-positive definite?)");
    }
    m_factorizationType = FactorizationType::Numeric;
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    auto result = m_ldl->RefactorWithFixedSparsityPattern(m_catamariConverter->conversionPlan(), A.Ax.data(), sigma, B.Ax.data());
    if (size_t(result.num_successful_pivots) != n_reduced()) {
        m_factorizationType = FactorizationType::Symbolic;
        throw std::runtime_error(std::to_string(result.num_successful_pivots) + "/" +
                                 std::to_string(n_reduced()) + "  pivots successful in Catamari numeric factorization (non-positive definite?)");
    }
    m_factorizationType = FactorizationType::Numeric;
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    auto result = m_ldl->RefactorWithFixedSparsityPattern(m_catamariConverter->conversionPlan(), A.Ax.data(), sigma, nullptr);
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
    // Brute-force reference implementation for solvers that don't support solving for multiple RHS at once.
    if (size_t(B.rows()) != m()) throw std::runtime_error("Incorrect RHS size");
    const size_t nrhs = B.cols();
    if (nrhs < 1) throw std::runtime_error("Must specify at least one rhs.");

    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> X_scratch;

    if (hasFixedVars()) {
        removeFixedEntries(B, X_scratch, /* permute = */ true);

#if 1
        catamari::BlasMatrixView<double> v;
        const size_t s = n_reduced();
        v.height = s;
        v.width = nrhs;
        v.leading_dim = s;
        v.data = X_scratch.data();

        {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Solve");
            m_ldl->Solve(&v, /* alreadyPermuted = */ true);
        }
#else
        for (size_t i = 0; i < nrhs; ++i) {
            catamari::BlasMatrixView<double> v;
            const size_t s = n_reduced();
            v.height = s;
            v.width = 1;
            v.leading_dim = s;
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
        throw std::runtime_error("Unimplemented");
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
