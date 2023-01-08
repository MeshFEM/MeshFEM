#include <MeshFEM/SparseMatrices.hh>
#include "CatamariFactorizer.hh"
#include "MeshFEM/Solvers/CholeskyFactorizerBase.hh"

extern "C" {
#include <cholmod.h>
}

#if MESHFEM_WITH_CATAMARI

#define DUMP_MATRICES 0

void CatamariFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    const SuiteSparseMatrix *A_reduced = m_initRowColRemoval(mat, pinnedVars);
#if DUMP_MATRICES
    {
        static size_t i = 0;
        size_t n_zero = 4;
        std::string padded_num = std::to_string(i++);
        padded_num = std::string(n_zero - std::min(n_zero, padded_num.length()), '0') + padded_num;
        A->dumpBinary("symbolic_mat_" + padded_num + ".bin");
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
        m_ldl.Factor(m_catamariConverter->get(), m_ldlControl);
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
        m_ldl.Factor(m_catamariConverter->get(), ordering, m_ldlControl);
    }
    else throw std::runtime_error("Unknown orderingMethod");
    m_factorizationType = FactorizationType::Symbolic;
}

void CatamariFactorizer::factorizeNumeric(const SuiteSparseMatrix &mat, bool /* isInTryCatch */) {
#if DUMP_MATRICES
    {
        const SuiteSparse *A = m_rowColRemoval(mat);
        static size_t i = 0;
        size_t n_zero = 4;
        std::string padded_num = std::to_string(i++);
        padded_num = std::string(n_zero - std::min(n_zero, padded_num.length()), '0') + padded_num;
        A->dumpBinary("numeric_mat_" + padded_num + ".bin");
    }
#endif
    m_catamariConverter->injectEntries(m_ldl, mat, m_reducedRowForRow, m_reducedEntryForEntry);
    m_factorizeInjectedEntries();
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, const SuiteSparseMatrix &B, Real sigma, bool /* isInTryCatch */) {
    m_catamariConverter->injectEntries(m_ldl, A, m_reducedRowForRow, m_reducedEntryForEntry, sigma, &B);
    m_factorizeInjectedEntries();
}

void CatamariFactorizer::m_factorizeInjectedEntries() {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Numeric Factorize");
    const auto &cmat = m_catamariConverter->get(); // TODO: remove matrix argument from RefactorWithFixedSparsityPattern.
    auto result = m_ldl.RefactorWithFixedSparsityPattern(cmat);
    if (result.num_successful_pivots != cmat.NumColumns()) {
        m_factorizationType = FactorizationType::Symbolic;
        throw std::runtime_error(std::to_string(result.num_successful_pivots) + "/" +
                                 std::to_string(cmat.NumColumns()) + "  pivots successful in Catamari numeric factorization (non-positive definite?)");
    }
    m_factorizationType = FactorizationType::Numeric;
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
            m_ldl.Solve(&v, /* alreadyPermuted = */ true);
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
                m_ldl.Solve(&v, /* alreadyPermuted = */ true);
            }
        }
#endif

        extractFullSolution(X_scratch, X, /* permute = */ true);
    }
    else {
        throw std::runtime_error("Unimplemented");
    }
}

#endif
