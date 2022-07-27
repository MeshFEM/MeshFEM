#ifndef CHOLESKYFACTORIZERBASE_HH
#define CHOLESKYFACTORIZERBASE_HH

#include <stdexcept>
#include <cassert>
#include <memory>
#include "../Types.hh"
#include "../GlobalBenchmark.hh"

enum class CholeskyProvider {
    CHOLMOD, Catamari
};

// Solve Ax =     b when sys = A,
//       Lx =     b when sys = L,
//    L^T x =     b when sys = Lt,
//        x = P   b when sys = P
//        x = P^T b when sys = Pt
enum class CholeskySys { A, L, Lt, P, Pt };

// Interface to a CHOLMOD-like Cholesky factorization class.
struct CholeskyFactorizerBase {
    enum class FactorizationType : int {
        None = 0, Symbolic = 1, Numeric = 2
    };

    virtual size_t m() const = 0;
    virtual size_t n() const = 0;

    // Perform only the symbolic factorization for the given matrix `mat`.
    virtual void factorizeSymbolic(const SuiteSparseMatrix &mat) = 0;

    // (Re)compute the numeric factorization, reusing the symbolic factorization
    // if it exists; otherwise a symbolic factorization is computed.
    // For symbolic factorization reuse to work, `mat` must have the same
    // sparsity pattern as the matrix for which the symbolic factorization was computed.
    virtual void  factorizeNumeric(const SuiteSparseMatrix &mat, bool isInTryCatch=false) = 0;

    // Compute the numeric factorization of `A + sigma * B`, reusing the
    // symbolic factorization if it exists.
    virtual void  factorizeNumericWithShift(const SuiteSparseMatrix &A, const SuiteSparseMatrix &B, Real sigma, bool isInTryCatch=false) = 0;

    // (Re)compute both symbolic and numeric factorizations
    virtual void  factorize       (const SuiteSparseMatrix &mat, bool isInTryCatch=false) = 0;
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

    void assertFactorization(FactorizationType type)           const { if (!hasFactorization(type)) throw std::runtime_error("Factorization does not exist"); }
    void assertFactorization(CholeskySys sys = CholeskySys::A) const { if (!hasFactorization(sys)) throw std::runtime_error("Factorization does not exist"); }

    template<typename _Vec1, typename _Vec2>
    void solve(const _Vec1 &b, _Vec2 &x, CholeskySys sys = CholeskySys::A) const {
        assert(size_t(b.size()) == m());
        x.resize(m());
        solveRaw(&b[0], &x[0], sys);
    }

    template<typename _Vec>
    _Vec solve(const _Vec &b, CholeskySys sys = CholeskySys::A) const {
        assert(size_t(b.size()) == m());
        _Vec x(m());
        solveRaw(&b[0], &x[0], sys);
        return x;
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    virtual void solveRaw(const Real *b, Real *x, CholeskySys sys = CholeskySys::A) const = 0;

    virtual CholeskyProvider provider() const = 0;

    virtual ~CholeskyFactorizerBase() { }
protected:
    FactorizationType m_factorizationType = FactorizationType::None;
};

#endif /* end of include guard: CHOLESKYFACTORIZERBASE_HH */
