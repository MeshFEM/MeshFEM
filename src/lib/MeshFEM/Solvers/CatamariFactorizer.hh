#ifndef CATAMARIFACTORIZER_HH
#define CATAMARIFACTORIZER_HH

#include "CholeskyFactorizerBase.hh"

#if MESHFEM_WITH_CATAMARI

#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include <specify.hpp>

struct CatamariFactorizer final : public CholeskyFactorizerBase {
    // Size of the factorized matrix.
    size_t m() const override { return m_AStorage.m; }
    size_t n() const override { return m_AStorage.n; }

    CatamariFactorizer(const SuiteSparseMatrix &mat) {
    }

    // TODO: Parallelized conversion from SuiteSparseMatrix to CoordinateMatrix<double>
    // Figure out what to do with the updateFactorization method to avoid extraneous sequential copies
    // even with Cholmod, e.g., in the Newton solver...
    //
    // Can we point m_A in CholmodFactorizer at the passed matrix's data?
    // Try this and test the existing use cases...

    void factorize() override { throw std::runtime_error("Unimplemented"); }

    // Perform only the symbolic factorization with the current system matrix
    // (useful this matrix holds the sparsity pattern that will be used for
    // many numeric factorizations).
    void factorizeSymbolic() override { throw std::runtime_error("Unimplemented"); }

    // Update the symbolic factorization for with a different sparsity pattern.
    void updateSymbolicFactorization(SuiteSparseMatrix mat) override { throw std::runtime_error("Unimplemented"); }

    void updateFactorization(SuiteSparseMatrix mat, bool isInTryCatch=false) override { throw std::runtime_error("Unimplemented"); }

    void solveRawExistingFactorization(const Real *b, Real *x, CholeskySys sys = CholeskySys::A) const override { throw std::runtime_error("Unimplemented"); }

    bool hasFactorization() const override { throw std::runtime_error("Unimplemented"); }

    void clearFactors() override { throw std::runtime_error("Unimplemented"); }

    void        stashFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    bool   hasStashedFactorization() const override { throw std::runtime_error("Stashing unimplemented"); }
    void  swapStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    void clearStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }

    virtual ~CatamariFactorizer() { }

    // Check if the matrix for which factor "L" was computed is positive definite.
    bool checkPosDef() const override { throw std::runtime_error("Unimplemented"); }
    virtual CholeskyProvider provider() const override { return CholeskyProvider::Catamari; }

private:
    void m_init() { throw std::runtime_error("Unimplemented"); }

    catamari::CoordinateMatrix<double> m_A;
};
#endif

#endif /* end of include guard: CATAMARIFACTORIZER_HH */
