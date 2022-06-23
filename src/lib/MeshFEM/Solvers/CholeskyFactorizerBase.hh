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
    virtual size_t m() const = 0;
    virtual size_t n() const = 0;
    virtual void factorize() = 0;
    virtual void factorizeSymbolic() = 0;
    virtual void updateSymbolicFactorization(SuiteSparseMatrix mat) = 0;
    virtual void updateFactorization(SuiteSparseMatrix mat, bool isInTryCatch=false) = 0;
    virtual void solveRawExistingFactorization(const Real *b, Real *x, CholeskySys sys = CholeskySys::A) const = 0;
    virtual void stashFactorization() = 0;
    virtual bool hasStashedFactorization() const = 0;
    virtual void swapStashedFactorization() = 0;
    virtual void clearStashedFactorization() = 0;
    virtual bool hasFactorization() const = 0;
    virtual void clearFactors() = 0;
    virtual void setSuppressWarnings(bool /* suppressWarnings */) { }
    virtual bool checkPosDef() const = 0;

    template<typename _Vec1, typename _Vec2>
    void solve(const _Vec1 &b, _Vec2 &x, CholeskySys sys = CholeskySys::A) {
        assert(size_t(b.size()) == m());
        x.resize(m());
        solveRaw(&b[0], &x[0], sys);
    }

    template<typename _Vec>
    _Vec solve(const _Vec &b, CholeskySys sys = CholeskySys::A) {
        assert(size_t(b.size()) == m());
        _Vec x(m());
        solveRaw(&b[0], &x[0], sys);
        return x;
    }

    template<typename _Vec1, typename _Vec2>
    void solveExistingFactorization(const _Vec1 &b, _Vec2 &x, CholeskySys sys = CholeskySys::A) const {
        assert(size_t(b.size()) == m());
        x.resize(m());
        solveRawExistingFactorization(&b[0], &x[0], sys);
    }

    template<typename _Vec>
    _Vec solveExistingFactorization(const _Vec &b, CholeskySys sys = CholeskySys::A) const {
        assert(size_t(b.size()) == m());
        _Vec x(m());
        solveRawExistingFactorization(&b[0], &x[0], sys);
        return x;
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRaw(const Real *b, Real *x, CholeskySys sys = CholeskySys::A) {
        if (!hasFactorization()) factorize();
        solveRawExistingFactorization(b, x, sys);
    }

    virtual CholeskyProvider provider() const = 0;

    virtual ~CholeskyFactorizerBase() { }
};

#endif /* end of include guard: CHOLESKYFACTORIZERBASE_HH */
