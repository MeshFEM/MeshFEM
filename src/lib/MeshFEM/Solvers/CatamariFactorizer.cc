#include <MeshFEM/SparseMatrices.hh>
#include "CatamariFactorizer.hh"
#include "MeshFEM/Solvers/CholeskyFactorizerBase.hh"

extern "C" {
#include <cholmod.h>
}

#if MESHFEM_WITH_CATAMARI

#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include <specify.hpp>

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

    CatamariConverter(const SuiteSparseMatrix &Asp) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter");
        if (Asp.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");
        if (Asp.m != Asp.n) throw std::runtime_error("Only square matrices are supported");

        // Convert upper triangle sparsity pattern to a full symmetric sparsity
        // pattern in Catamari format.
        m_result.Resize(Asp.m, Asp.n);
#if 0
        m_result.ReserveEntryAdditions(Asp.Ax.size() * 2 - Asp.Ap.size());
        for (auto t : Asp) {
            m_result.                QueueEntryAddition(t.i, t.j, t.v);
            if (t.i != t.j) m_result.QueueEntryAddition(t.j, t.i, t.v);
        }
        m_result.FlushEntryQueues();
#else
        {
            SuiteSparseMatrix A_full = Asp.toSymmetryModeSparsityOnly(SuiteSparseMatrix::SymmetryMode::NONE);

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
        }
#endif
    }

    // Achieve the same result as
    // `catamari::supernodal_ldl::InitializeBlockColumn` for sparse matrix `A`
    // or `A + sigma B` (with B of identical sparsity pattern to `A`) after
    // possibly converting `A` and `B` into "reduced" versions by removing rows
    // and columns corresponding to pinned vars.
    // This row/column removal is effectively implemented by the
    // `reducedRowForRow` and `reducedEntryForEntry` arguments.
    void injectEntries(catamari::SparseLDL<double> &ldl, const SuiteSparseMatrix &A, std::vector<SuiteSparse_long> &reducedRowForRow, std::vector<SuiteSparse_long> &reducedEntryForEntry, double sigma = 0.0, const SuiteSparseMatrix *B_optional = nullptr) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Inject entries");
        auto f = ldl.supernodal_factorization.get();
        if (f == nullptr) throw std::runtime_error("Only supernodal factorizations are supported");
        if (A.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");

        auto &df = f->diagonal_factor_;
        auto &lf = f->lower_factor_;

        using Int = catamari::Int;
        auto &o  = f->ordering_;
        auto &sno = o.supernode_offsets;
        const Int num_supernodes = o.supernode_sizes.Size();
        const SuiteSparse_long lowerBlockOffset = df->values_.Size();

        if (o.permutation.Empty()) throw std::runtime_error("Expected permutation");

        size_t nthreads = get_max_num_tbb_threads();
        if (nthreads >= 2) {
            setZeroParallel(catamari::eigenMap(df->values_));
            setZeroParallel(catamari::eigenMap(lf->values_));
        }
        else {
            catamari::eigenMap(df->values_).setZero();
            catamari::eigenMap(lf->values_).setZero();
        }

        if (m_locForEntry.empty()) {
            BENCHMARK_SCOPED_TIMER_SECTION ctimer("Construct plan");

            m_locForEntry.resize(A.nz, SuiteSparseMatrix::INDEX_NONE);
            if (size_t(A.nz) != A.Ai.size()) throw std::runtime_error("Incorrect nonzero count");

            auto reducedVarIndex = [&](SuiteSparse_long  i) { return reducedRowForRow.empty() ? i : reducedRowForRow[i]; };
            auto nonzeroRemoved  = [&](SuiteSparse_long ii) { return reducedEntryForEntry.size() && (reducedEntryForEntry[ii] == SuiteSparseMatrix::INDEX_NONE); };

            // First, locate the supernode corresponding to each column of the reduced matrix.
            const Int nReducedVars = ldl.NumRows();
            Eigen::Array<SuiteSparse_long, Eigen::Dynamic, 1> supernodeForReducedCol(nReducedVars);
            if (sno[num_supernodes] != nReducedVars) throw std::runtime_error("Columns missing from supernodes");
            for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
                for (Int col = sno[supernode]; col < sno[supernode + 1]; ++col) {
                    if (col >= A.n) throw std::runtime_error("Supernode column index out of bounds");
                    supernodeForReducedCol[col] = supernode;
                }
            }

            // For each entry in the (upper triangle) input matrix, figure out where it goes
            // in the *lower triangle* of the factorization structure...
            parallel_for_range(A.n, [&](SuiteSparse_long j_orig) {
                const Int *guess = nullptr; // guess for index search performed inside...
                for (SuiteSparse_long ii = A.Ap[j_orig]; ii < A.Ap[j_orig + 1]; ++ii) {
                    if (nonzeroRemoved(ii)) continue;

                    Int i_perm = o.permutation[reducedVarIndex(A.Ai[ii])];
                    Int j_perm = o.permutation[reducedVarIndex(j_orig)];
                    if (i_perm < j_perm) std::swap(i_perm, j_perm); // write lower triangle entry!
                    // Locate (i_perm, j_perm) in the supernode structure

                    // Find the supernode
                    const Int supernode = supernodeForReducedCol[j_perm];
                    catamari::BlasMatrixView<double>& diagonal_block = df->blocks[supernode];
                    catamari::BlasMatrixView<double>& lower_block = lf->blocks[supernode];

                    const Int supernode_start = sno[supernode    ];
                    const Int supernode_end   = sno[supernode + 1];

                    const Int j_rel = j_perm - supernode_start;
                    if (i_perm < supernode_start) throw std::runtime_error("i_perm before start");

                    if (i_perm < supernode_end) {
                        const Int i_rel = i_perm - supernode_start;
                        size_t dbIndex = std::distance(df->values_.Data(), diagonal_block.Pointer(i_rel, j_rel));
                        m_locForEntry[ii] = dbIndex;
                    }
                    else {
                        const Int *index_beg = lf->StructureBeg(supernode);
                        const Int *index_end = lf->StructureEnd(supernode);

                        // Search [lf->structureBeg, lf->structureEnd) for value `i_perm`,
                        // first checking at `*guess` (which will be correct for consecutive
                        // strips of entries).
                        const Int *iter;
                        if ((guess >= index_beg) && (guess < index_end) && (*guess == i_perm)) iter = guess;
                        else {
                            iter = std::lower_bound(index_beg, index_end, i_perm);
                            if ((iter == index_end) || (*iter != i_perm)) throw std::runtime_error("Couldn't locate row index in supernode");
                        }
                        guess = iter + 1;

                        const Int i_rel = std::distance(index_beg, iter);
                        size_t lbIndex = std::distance(lf->values_.Data(), lower_block.Pointer(i_rel, j_rel));
                        m_locForEntry[ii] = lowerBlockOffset + lbIndex;
                    }
                }
            });
        }

        {
            if (B_optional == nullptr || sigma == 0) {
                if (nthreads > 1) {
                    parallel_for_range(A.nz, [&](size_t ii) {
                            SuiteSparse_long loc = m_locForEntry[ii];
                            if (loc == SuiteSparseMatrix::INDEX_NONE) return; // skip removed entries
                            if (loc < lowerBlockOffset) df->values_[loc                   ] = A.Ax[ii];
                            else                        lf->values_[loc - lowerBlockOffset] = A.Ax[ii];
                        });
                }
                else {
                    for (SuiteSparse_long ii = 0; ii < A.nz; ++ii) {
                        SuiteSparse_long loc = m_locForEntry[ii];
                        if (loc == SuiteSparseMatrix::INDEX_NONE) continue; // skip removed entries
                        if (loc < lowerBlockOffset) df->values_[loc                   ] = A.Ax[ii];
                        else                        lf->values_[loc - lowerBlockOffset] = A.Ax[ii];
                    }
                }
            }
            else {
                // Factorize with shift.
                const auto &B = *B_optional;
                SuiteSparse_long nc = A.m;
                if ((B.m != nc) || (B.n != nc)) throw std::runtime_error("Unexpected input shape(s)");
                if (B.Ai.size() != A.Ai.size()) throw std::runtime_error("B must have the same sparsity pattern as A");
                parallel_for_range(A.nz, [&](size_t ii) {
                        SuiteSparse_long loc = m_locForEntry[ii];
                        if (loc == SuiteSparseMatrix::INDEX_NONE) return; // skip removed entries
                        double value = A.Ax[ii] + sigma * B.Ax[ii];
                        if (loc < lowerBlockOffset) df->values_[loc                   ] = value;
                        else                        lf->values_[loc - lowerBlockOffset] = value;
                    });
            }
        }
    }

    // Get the most recently converted matrix.
    const CMat &get() const { return m_result; }

private:
    CMat m_result;
    std::vector<SuiteSparse_long> m_locForEntry;
};

CatamariFactorizer::CatamariFactorizer() {
    m_ldl        = std::make_unique<catamari::SparseLDL<double>>();
    m_ldlControl = std::make_unique<catamari::SparseLDLControl<double>>();
    m_ldlControl->SetFactorizationType(catamari::kCholeskyFactorization);
    m_ldlControl->supernodal_strategy = catamari::kSupernodalFactorization;
    m_ldlControl->supernodal_control.algorithm = catamari::kRightLookingLDL;
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
    // Catamari doesn't currently support computing only the symbolic factorization...
    // It looks like it may be possible to hack a true `factorizeSymbolic` implementation
    // by creating a copy of the `Factor` routine that omits this part:
    //      https://gitlab.com/hodge_star/catamari/-/blob/master/include/catamari/sparse_ldl/supernodal/factorization/common-impl.hpp#L328
    // However, the overhead of doing a numeric factorization is quite low, especially since
    // we often run the symbolic factorization on a singular matrix with the sparsity pattern
    // filled with ones (in which case, the numeric factorization fails quickly).
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
    m_factorizationType = FactorizationType::Symbolic;
}

void CatamariFactorizer::factorizeNumeric(const SuiteSparseMatrix &mat, bool /* isInTryCatch */) {
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
    m_catamariConverter->injectEntries(*m_ldl, mat, m_reducedRowForRow, m_reducedEntryForEntry);
    m_factorizeInjectedEntries();
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, const SuiteSparseMatrix &B, Real sigma, bool /* isInTryCatch */) {
    m_catamariConverter->injectEntries(*m_ldl, A, m_reducedRowForRow, m_reducedEntryForEntry, sigma, &B);
    m_factorizeInjectedEntries();
}

void CatamariFactorizer::m_factorizeInjectedEntries() {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Numeric Factorize");
    const auto &cmat = m_catamariConverter->get(); // TODO: remove matrix argument from RefactorWithFixedSparsityPattern.
    auto result = m_ldl->RefactorWithFixedSparsityPattern(cmat);
    if (result.num_successful_pivots != cmat.NumColumns()) {
        m_factorizationType = FactorizationType::Symbolic;
        throw std::runtime_error(std::to_string(result.num_successful_pivots) + "/" +
                                 std::to_string(cmat.NumColumns()) + "  pivots successful in Catamari numeric factorization (non-positive definite?)");
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

CatamariFactorizer::~CatamariFactorizer() {
    if (m_c) cholmod_l_finish(m_c.get());
}

#endif
