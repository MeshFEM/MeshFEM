#include <MeshFEM/SparseMatrices.hh>

#if MESHFEM_WITH_CATAMARI

#define DUMP_MATRICES 0

void CatamariFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat) {
#if DUMP_MATRICES
    {
        static size_t i = 0;
        size_t n_zero = 4;
        std::string padded_num = std::to_string(i++);
        padded_num = std::string(n_zero - std::min(n_zero, padded_num.length()), '0') + padded_num;
        mat.dumpBinary("symbolic_mat_" + padded_num + ".bin");
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
    m_catamariConverter = std::make_unique<CatamariConverter>(mat);

    if (orderingMethod == OrderingMethod::Catamari)
        m_ldl.Factor(m_catamariConverter->get(), m_ldlControl);
    else if ((orderingMethod == OrderingMethod::CholmodNesdis) || (orderingMethod == OrderingMethod::Metis)) {
        if (!m_c) {
            m_c = std::make_unique<cholmod_common>();
            cholmod_l_start(m_c.get());
        }

        catamari::SymmetricOrdering ordering;
        {
            ordering.inverse_permutation.Resize(mat.m);
            catamari::Buffer<catamari::Int> CParent(mat.m), CMember(mat.m);
            auto cholmat = cholmod_sparse_view(mat);
            if (orderingMethod == OrderingMethod::CholmodNesdis) {
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_nested_dissection");
                cholmod_l_nested_dissection(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                            ordering.inverse_permutation.Data(),
                                            CParent.Data(), CMember.Data(), m_c.get());
            }
            else {
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_metis");
                cholmod_l_metis(&cholmat, /* fset = */ nullptr, /* fsize = */ 0, /* postorder = */ true,
                                ordering.inverse_permutation.Data(), m_c.get());
            }
            quotient::InvertPermutation(ordering.inverse_permutation, &ordering.permutation);
        }
        m_ldl.Factor(m_catamariConverter->get(), ordering, m_ldlControl);
    }
    else throw std::runtime_error("Unknown orderingMethod");
    m_factorizationType = FactorizationType::Symbolic;
}

void CatamariFactorizer::factorizeNumeric(const CMat &cmat) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Numeric Factorize");
    auto result = m_ldl.RefactorWithFixedSparsityPattern(cmat);
    if (result.num_successful_pivots != cmat.NumColumns()) {
        m_factorizationType = FactorizationType::Symbolic;
        throw std::runtime_error(std::to_string(result.num_successful_pivots) + "/" +
                                 std::to_string(cmat.NumColumns()) + "  pivots successful in Catamari numeric factorization (non-positive definite?)");
    }
    m_factorizationType = FactorizationType::Numeric;
}

void CatamariFactorizer::factorizeNumeric(const SuiteSparseMatrix &mat, bool /* isInTryCatch */) {
#if DUMP_MATRICES
    {
        static size_t i = 0;
        size_t n_zero = 4;
        std::string padded_num = std::to_string(i++);
        padded_num = std::string(n_zero - std::min(n_zero, padded_num.length()), '0') + padded_num;
        mat.dumpBinary("numeric_mat_" + padded_num + ".bin");
    }
#endif
    factorizeNumeric(m_catamariConverter->convert(mat));
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, const SuiteSparseMatrix &B, Real sigma, bool /* isInTryCatch */) {
    factorizeNumeric(m_catamariConverter->convertWithShift(A, sigma, B));
}

#endif
