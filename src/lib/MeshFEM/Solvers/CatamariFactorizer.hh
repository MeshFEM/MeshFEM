#ifndef CATAMARIFACTORIZER_HH
#define CATAMARIFACTORIZER_HH

#include "CholeskyFactorizerBase.hh"

#if MESHFEM_WITH_CATAMARI

#include <MeshFEM/Parallelism.hh>
#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include "CholmodFactorizer.hh"
#include <specify.hpp>
#include <MeshFEM_export.h>

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
        if (Asp.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");
        if (Asp.m != Asp.n) throw std::runtime_error("Only square matrices are supported");

        // Convert upper triangle sparsity pattern to a full symmetric sparsity
        // pattern in Catamari format.
        m_result.Resize(Asp.m, Asp.n);
        m_result.ReserveEntryAdditions(Asp.Ax.size() * 2 - Asp.Ax.size());
        for (auto t : Asp) {
            m_result.                QueueEntryAddition(t.i, t.j, t.v);
            if (t.i != t.j) m_result.QueueEntryAddition(t.j, t.i, t.v);
        }
        m_result.FlushEntryQueues();

        // Note: although we technically need to access only the strict upper
        // triangle through `m_upperTriPtr`, we store the full upper triangle
        // pointers to avoid a branch in the conversion routine.
        // (This means diagonal entries will be written twice).
        m_upperTriPtr.resize(Asp.nnz());
        size_t loc = 0;
        for (auto t : Asp)
            m_upperTriPtr[loc++] = m_result.Entries().Data() + m_result.EntryOffset(t.i, t.j);
    }

    // Convert and cache the numerical values of matrix `A` (assuming `A` has
    // an identical sparsity pattern to `Asp`).
    const CMat &convert(const SuiteSparseMatrix &A) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter.convert");
        catamari::Int nc = m_result.NumColumns();
        if ((A.m != nc) || (A.n != nc)) throw std::runtime_error("Unexpected input shape");

        parallel_for_range(nc, [&](int j) {
            auto outPtr = m_result.Entries().Data() + m_result.RowEntryOffsets()[j];
            size_t end = A.Ap[j + 1];
            for (size_t loc = A.Ap[j]; loc < end; ++loc)
                (outPtr++)->value = m_upperTriPtr[loc]->value = A.Ax[loc];
        });

        return m_result;
    }

    // Convert and cache the numerical values of matrix `A + sigma B` (assuming
    // `A` and `B` have identical sparsity patterns to `Asp`).
    const CMat &convertWithShift(const SuiteSparseMatrix &A, double sigma, const SuiteSparseMatrix &B) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter.convert");
        catamari::Int nc = m_result.NumColumns();
        if ((A.m != nc) || (A.n != nc) || (B.m != nc) || (B.n != nc)) throw std::runtime_error("Unexpected input shape(s)");

        parallel_for_range(nc, [&](int j) {
            auto outPtr = m_result.Entries().Data() + m_result.RowEntryOffsets()[j];
            size_t end = A.Ap[j + 1];
            for (size_t loc = A.Ap[j]; loc < end; ++loc)
                (outPtr++)->value = m_upperTriPtr[loc]->value = A.Ax[loc] + sigma * B.Ax[loc];
        });

        return m_result;
    }

    // Get the most recently converted matrix.
    const CMat &get() const { return m_result; }

private:
    std::vector<quotient::MatrixEntry<double> *> m_upperTriPtr;
    CMat m_result;
};

struct MESHFEM_EXPORT CatamariFactorizer final : public CholeskyFactorizerBase {
    using CMat = catamari::CoordinateMatrix<double>;
    enum class OrderingMethod {
        Catamari, CholmodNesdis, Metis
    };

    CatamariFactorizer() {
        m_ldlControl.SetFactorizationType(catamari::kCholeskyFactorization);
        m_ldlControl.supernodal_control.algorithm = catamari::kRightLookingLDL;
    }

    size_t m() const override { assertFactorization(FactorizationType::Symbolic); return m_ldl.NumRows(); }
    size_t n() const override { assertFactorization(FactorizationType::Symbolic); return m_ldl.NumRows(); }

    void factorizeSymbolic(const SuiteSparseMatrix &mat) override;

    void factorizeNumeric(const CMat &mat);
    void factorizeNumeric(const SuiteSparseMatrix &mat, bool /* isInTryCatch */ = false) override;
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, const SuiteSparseMatrix &B, Real sigma, bool isInTryCatch=false) override;

    // (Re)compute both symbolic and numeric factorizations
    void factorize(const SuiteSparseMatrix &mat, bool /* isInTryCatch */ = false) {
        factorizeSymbolic(mat);
        m_factorizationType = FactorizationType::Numeric;
    }

    void clearFactors() override {
        m_factorizationType = FactorizationType::None;
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRaw(const Real *b, Real *x, CholeskySys sys = CholeskySys::A) const override {
        assertFactorization(sys);
        if (sys != CholeskySys::A) {
            std::cout << "Alternative CholeskySys not yet wrapped for Catamari" << std::endl;
            throw std::runtime_error("Alternative CholeskySys not yet wrapped for Catamari");
        }
        const size_t s = m();

        // Catamari does the solve in-place! Copy `b` into `x` and wrap it in a
        // catamari::BlasMatrixView.
        Eigen::Map<Eigen::VectorXd>(x, s) = Eigen::Map<const Eigen::VectorXd>(b, s);

        catamari::BlasMatrixView<double> v;
        v.height = s;
        v.width = 1;
        v.leading_dim = s;
        v.data = x;

        BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Solve");
        m_ldl.Solve(&v);
    }

    void        stashFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    bool   hasStashedFactorization() const override { throw std::runtime_error("Stashing unimplemented"); }
    void  swapStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    void clearStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }

    bool checkPosDef() const override { return m_factorizationType == FactorizationType::Numeric; }
    CholeskyProvider provider() const override { return CholeskyProvider::Catamari; }

    virtual ~CatamariFactorizer() { }

    OrderingMethod orderingMethod = OrderingMethod::CholmodNesdis;
    // OrderingMethod orderingMethod = OrderingMethod::Catamari;

private:
    catamari::SparseLDL<double> m_ldl;
    catamari::SparseLDLControl<double> m_ldlControl;

    std::unique_ptr<CatamariConverter> m_catamariConverter;

    std::unique_ptr<cholmod_common> m_c; // Used for Nesdis and Metis ordering
};
#endif

#endif /* end of include guard: CATAMARIFACTORIZER_HH */
