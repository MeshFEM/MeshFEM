#ifndef CHOLESKYFACTORIZERBASE_HH
#define CHOLESKYFACTORIZERBASE_HH

#include <stdexcept>
#include <cassert>
#include <memory>
#include "../Types.hh"
#include "../GlobalBenchmark.hh"
#include "MeshFEM/SparseMatrices.hh"
#include "MeshFEM/unused.hh"

enum class CholeskyProvider {
    CHOLMOD, Catamari, CatamariNesdis, PARDISO
};

// Eigen provides a `swap` method rather than overloading `std::swap`
// (because the swapped matrices could be different expression types...)
namespace myswap {
    template< class T>
    void swap(T &a, T &b) noexcept {
        return std::swap(a, b);
    }

    template<class Derived1, class Derived2>
    void swap(Eigen::MatrixBase<Derived1> &A, Eigen::MatrixBase<Derived2> &B) {
        A.swap(B);
    }
}

// Solve Ax =     b when sys = A,
//       Lx =     b when sys = L,
//    L^T x =     b when sys = Lt,
//        x = P   b when sys = P
//        x = P^T b when sys = Pt
enum class CholeskySys { A, L, Lt, P, Pt };

// Interface to a Cholesky factorization class supporting the enforcement of
// pin constraints of the form:
//      x[fixedVars] = 0
struct CholeskyFactorizerBase {
    enum class FactorizationType : int {
        None = 0, Symbolic = 1, Numeric = 2
    };

    size_t m() const { return m_reduced() + m_fixedVars.size(); }
    size_t n() const { return n_reduced() + m_fixedVars.size(); }

    virtual size_t m_reduced() const = 0;
    virtual size_t n_reduced() const = 0;

    bool hasFixedVars() const { return !m_fixedVars.empty(); }
    const std::vector<size_t> &getFixedVars() const { return m_fixedVars; }
    bool varIsFixed(size_t i) const { return hasFixedVars() && m_fixedVars[i]; }

    // Perform only the symbolic factorization for the given matrix `mat` after removing the
    // rows and columns indicated by the indices in `pinnedVars`.
    virtual void factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) = 0;
    void factorizeSymbolic(const SuiteSparseMatrix &mat) { factorizeSymbolic(mat, std::vector<size_t>()); }

    // (Re)compute the numeric factorization, reusing the symbolic factorization
    // if it exists; otherwise a symbolic factorization is computed.
    // For symbolic factorization reuse to work, `mat` must have the same
    // sparsity pattern as the matrix for which the symbolic factorization was computed.
    virtual void factorizeNumeric(const SuiteSparseMatrix &mat, bool isInTryCatch=false) = 0;

    // Compute the numeric factorization of `A + sigma * B`, reusing the
    // symbolic factorization if it exists.
    virtual void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) = 0;

    // Compute the numeric factorization of `A + sigma * I`, reusing the
    // symbolic factorization if it exists.
    virtual void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool isInTryCatch=false) = 0;

    // (Re)compute both symbolic and numeric factorizations
    virtual void factorize(const SuiteSparseMatrix &mat, const std::vector<size_t> &fixedVars = std::vector<size_t>(), bool /* isInTryCatch */ = false) = 0;
    virtual void clearFactors() = 0;

    virtual void stashFactorization() = 0;
    virtual bool hasStashedFactorization() const = 0;
    virtual void swapStashedFactorization() = 0;
    virtual void clearStashedFactorization() = 0;

    virtual void setSuppressWarnings(bool /* suppressWarnings */) { }
    virtual bool checkPosDef() const = 0;

    // Check whether the factorization needed to solve `sys` exists;
    // this is generally a numeric factorization, but only a symbolic
    // factorization if `sys`is `P` or `Pt`.
    bool hasFactorization(FactorizationType type) const {
        return m_factorizationType >= type;
    }

    bool hasFactorization(CholeskySys sys = CholeskySys::A) const {
        if ((sys == CholeskySys::A) ||
            (sys == CholeskySys::L) ||
            (sys == CholeskySys::Lt)) return hasFactorization(FactorizationType::Numeric);
        return hasFactorization(FactorizationType::Symbolic);
    }

    void assertFactorization(FactorizationType type)           const { if (!hasFactorization(type)) throw std::runtime_error(((type == FactorizationType::Numeric) ? "Numeric" : "Symbolic") + std::string(" factorization does not exist")); }
    void assertFactorization(CholeskySys sys = CholeskySys::A) const { if (!hasFactorization( sys)) throw std::runtime_error("Factorization does not exist"); }

    virtual void solveMultiRHS(const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &B, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &X) const {
        // Brute-force reference implementation for solvers that don't support solving for multiple RHS at once.
        if (size_t(B.rows()) != m()) throw std::runtime_error("Incorrect RHS size");
        const size_t nrhs = B.cols();
        if (nrhs < 1) throw std::runtime_error("Must specify at least one rhs.");

        X.resize(B.rows(), B.cols());
        for (size_t i = 0; i < nrhs; ++i)
            X.col(i) = solve(B.col(i).eval());
    }

    template<typename _Vec1, typename _Vec2>
    void solve(const _Vec1 &b, _Vec2 &x, CholeskySys sys = CholeskySys::A) const {
        assertFactorization(sys);
        BENCHMARK_SCOPED_TIMER_SECTION timer("CholeskyFactorizerBase.solve");
        assert(size_t(b.size()) == m());
        x.resize(n());
        m_solveScratch.resize(n()); // always the full unreduced size to support swapping with `x`

        if (hasFixedVars()) {
            if (preferInPlaceSolve()) {
                const bool permute = supportsPrePermutation();
                removeFixedEntries(b, m_solveScratch.head(n_reduced()), permute);
                solveRawReducedInPlace(m_solveScratch.data(), sys, /* alreadyPermuted = */ permute);
                extractFullSolution(m_solveScratch.head(n_reduced()), x, permute);
            }
            else {
                removeFixedEntries(b, m_solveScratch.head(n_reduced()));
                solveRawReduced(m_solveScratch.data(), x.data(), sys);
                extractFullSolution(x.head(n_reduced()), m_solveScratch);
                myswap::swap(m_solveScratch, x);
            }
        }
        else {
            solveRawReduced(b.data(), x.data(), sys);
        }
    }

    template<typename _Vec>
    _Vec solve(const _Vec &b, CholeskySys sys = CholeskySys::A) const {
        assert(size_t(b.size()) == m());
        _Vec x;
        solve(b, x, sys);
        return x;
    }

    // `permute`: whether to also apply the permutation/inverse permutation in a fused operation.
    template<class VecIn, class VecOut>
    void extractFullSolution(const VecIn &xReduced, VecOut &&x, bool permute = false) const {
        if (!hasFixedVars()) {
            if (permute) throw std::runtime_error("Unimplemented");
            x = xReduced;
            return;
        }

        if (m_reducedRowForRow.size() != n()) throw std::logic_error("Variables were not fixed");
        if (size_t(xReduced.rows()) != n_reduced()) throw std::runtime_error("Invalid xReduced size");
        x.resize(n(), xReduced.cols());

        const SuiteSparse_long *reducedRowForRow = m_reducedRowForRow.data();
        if (permute) {
            m_populatePermutedReducedRowForRow();
            reducedRowForRow = m_permutedReducedRowForRow.data();
        }

        if (xReduced.cols() > 1) {
            parallel_for_range(x.rows(), [&](size_t i) {
                SuiteSparse_long row = reducedRowForRow[i];
                if (row != SuiteSparseMatrix::INDEX_NONE) x.row(i) = xReduced.row(row);
                else                                      x.row(i).setZero();
            });
        }
        else {
            const Real *xR_ptr = xReduced.data();
            Real *x_ptr  = x.data();
            for (decltype(x.size()) i = 0; i < x.size(); ++i) {
                SuiteSparse_long row = reducedRowForRow[i];
                x_ptr[i] = (row != SuiteSparseMatrix::INDEX_NONE) ? xR_ptr[row] : 0.0;
            }
        }
    }

    template<class VecIn, class VecOut>
    void removeFixedEntries(const VecIn &x, VecOut &&xReduced, bool permute = false) const {
        if (!hasFixedVars()) {
            if (permute) throw std::runtime_error("Unimplemented");
            xReduced = x;
            return;
        }

        if (m_reducedRowForRow.size() != n()) throw std::logic_error("Variables were not fixed");
        if (size_t(x.rows()) != n()) throw std::runtime_error("Invalid x size");

        const SuiteSparse_long *reducedRowForRow = m_reducedRowForRow.data();
        if (permute) {
            m_populatePermutedReducedRowForRow();
            reducedRowForRow = m_permutedReducedRowForRow.data();
        }

        xReduced.resize(n_reduced(), x.cols());
        if (x.cols() > 1)  {
            parallel_for_range(x.rows(), [&](size_t i) {
                SuiteSparse_long row = reducedRowForRow[i];
                if (row != SuiteSparseMatrix::INDEX_NONE) xReduced.row(row) = x.row(i);
            });
        }
        else {
            Real *xR_ptr = xReduced.data();
            const Real *x_ptr  = x.data();
            for (decltype(x.size()) i = 0; i < x.size(); ++i) {
                SuiteSparse_long row = reducedRowForRow[i];
                if (row != SuiteSparseMatrix::INDEX_NONE) xR_ptr[row] = x_ptr[i];
            }
        }
    }

    // Raw pointer versions (Use with care! Caller must allocate/own both pointers)
    // These calls are for the *reduced* system obtained after row/col reduction.
    // (b and x should be the reduced right-hand-side and x, respectively).
    virtual void solveRawReduced(const Real *b, Real *x, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const = 0;
    virtual void solveRawReducedInPlace(Real *bx, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const { UNUSED(bx); UNUSED(sys); UNUSED(alreadyPermuted); throw std::runtime_error("Unimplemented"); }
    // Which version of solve is "native"?
    virtual bool preferInPlaceSolve() const = 0;
    // Whether the solver allows us to apply the permutation (accepts `alreadyPermuted = true`).
    // This enables an optimization where the permutation is fused with the row removal operation.
    virtual bool supportsPrePermutation() const = 0;

    virtual CholeskyProvider provider() const = 0;

    virtual ~CholeskyFactorizerBase() { }
protected:
    FactorizationType m_factorizationType = FactorizationType::None;

    // Functionality for efficient solves under variable pins
    std::vector<size_t> m_fixedVars;
    std::vector<SuiteSparse_long> m_entryForReducedEntry;
    std::vector<SuiteSparse_long> m_reducedRowForRow;
    mutable std::vector<SuiteSparse_long> m_permutedReducedRowForRow;
    std::unique_ptr<SuiteSparseMatrix> m_Areduced;
    using VXd = VecX_T<Real>;
    mutable VXd m_solveScratch;

    // This is meant to be called only once upon symbolic factorization, and
    // the resulting reduced matrix is re-used for factorization
    const SuiteSparseMatrix *m_initRowColRemoval(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
        if (pinnedVars.empty()) return &mat;

        m_Areduced = std::make_unique<SuiteSparseMatrix>(mat);

        // Deduplicate fixed vars and construct mask needed for efficient row/col removal.
        m_fixedVars.clear();
        std::vector<bool> varIsFixed;
        varIsFixed.assign(mat.n, false);
        for (size_t var : pinnedVars) {
            if (!varIsFixed[var]) m_fixedVars.push_back(var);
            varIsFixed[var] = true;
        }

        m_Areduced->rowColRemoval([&](SuiteSparse_long i) { return varIsFixed[i]; }, &m_reducedRowForRow, &m_entryForReducedEntry);
        m_permutedReducedRowForRow.clear();
        return m_Areduced.get();
    }

    const SuiteSparseMatrix *m_rowColRemoval(const SuiteSparseMatrix &mat) {
        // Inject values of `A` into row-col-removed matrix
        if (!hasFixedVars()) return &mat;
        if (!m_Areduced)  throw std::logic_error("Variables were not fixed");

        auto &A_reduced = *m_Areduced;
        for (SuiteSparse_long ii = 0; ii < A_reduced.nz; ++ii)
            A_reduced.Ax[ii] = mat.Ax[m_entryForReducedEntry[ii]];
        return m_Areduced.get();
    }

    virtual void m_populatePermutedReducedRowForRow() const { throw std::runtime_error("Unimplemented"); }
};

#endif /* end of include guard: CHOLESKYFACTORIZERBASE_HH */
