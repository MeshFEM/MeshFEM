#ifndef UMFPACKFACTORIZER_HH
#define UMFPACKFACTORIZER_HH

extern "C" {
#include <umfpack.h>

namespace MeshFEM {
}

#ifdef MESHFEM_WITH_UMFPACK
class UmfpackFactorizer {
public:
    template<typename _Triplet>
    UmfpackFactorizer(TripletMatrix<_Triplet> &tmat)
        : m_mat(tmat), symbolic(NULL), numeric(NULL),
          m_factorizationMemoryBytes(0) { }

    void factorize() {
        clear();

        umfpack_dl_defaults(Control);
        BENCHMARK_START_TIMER("UMFPACK Symbolic Factorize");
        int status = umfpack_dl_symbolic(m_mat.m, m_mat.n, Ap(), Ai(), Ax(),
                                         &symbolic, Control, Info);
        BENCHMARK_STOP_TIMER("UMFPACK Symbolic Factorize");
        if (status != UMFPACK_OK) {
            // Symbolic object isn't created when there is a failure, so there
            // is nothing to free.
            throw std::runtime_error("Umfpack symbolic factorization failed: "
                    + std::to_string(status));
        }

        BENCHMARK_START_TIMER("UMFPACK Numeric Factorize");
        status = umfpack_dl_numeric(Ap(), Ai(), Ax(), symbolic, &numeric,
                                    Control, Info);
        BENCHMARK_STOP_TIMER("UMFPACK Numeric Factorize");
        if (status != UMFPACK_OK) {
            umfpack_dl_free_symbolic(&symbolic);
            // A numeric object is allocated if we just got the singular matrix
            // warning, so we better free it. In all other cases, no object is
            // created.
            if (status == UMFPACK_WARNING_singular_matrix)
                umfpack_dl_free_numeric(&numeric);
            umfpack_dl_report_status(Control, status);
            throw std::runtime_error("Umfpack numeric factorization failed: "
                    + std::to_string(status));
        }

        m_factorizationMemoryBytes = Info[UMFPACK_PEAK_MEMORY] *
                                     Info[UMFPACK_SIZE_OF_UNIT];
        BENCHMARK_ADD_MESSAGE("Peak factorization memory (MB):\t" +
                              std::to_string(m_factorizationMemoryBytes / (1 << 20)));
    }

    // Perform only the symbolic factorization with the current system matrix
    // (useful this matrix holds the sparsity pattern that will be used for
    // many numeric factorizations).
    [[ noreturn ]] void factorizeSymbolic(int /* nmethods */) {
        throw std::runtime_error("Unimplemented");
    }

    // Recompute the numeric factorization using the new system matrix "tmat",
    // reusing the symbolic factorization. For this to work, it must have the same
    // sparsity pattern as the matrix for which the symbolic factorization was computed.
    template<typename _Triplet>
    [[ noreturn ]] void updateFactorization(const TripletMatrix<_Triplet> &/* tmat */) {
        throw std::runtime_error("Unimplemented");
    }

    template<typename _Vec1, typename _Vec2>
    void solve(const _Vec1 &b, _Vec2 &x) {
        if (numeric == NULL) factorize();

        assert(b.size() == (size_t) m_mat.m);
        x.resize(m_mat.n);
        int status = umfpack_dl_solve(UMFPACK_A, Ap(), Ai(), Ax(), &x[0], &b[0],
                                      numeric, Control, Info);
        if (status != UMFPACK_OK) {
            throw std::runtime_error("Umfpack solve failed: "
                    + std::to_string(status));
        }
    }

    double peakMemoryMB() const {
        return m_factorizationMemoryBytes / (1 << 20);
    }

    void clear() {
        if (symbolic) umfpack_dl_free_symbolic(&symbolic);
        if (numeric)  umfpack_dl_free_numeric(&numeric);
    }

    ~UmfpackFactorizer() {
        clear();
    }

    // Size of the factorized matrix.
    size_t m() const { return m_mat.m; }
    size_t n() const { return m_mat.m; }

private:
    const SuiteSparse_long *Ap() const { return &m_mat.Ap[0]; }
    const SuiteSparse_long *Ai() const { return &m_mat.Ai[0]; }
    const double *Ax()           const { return &m_mat.Ax[0]; }

    // Note: SuiteSparse version of A  must be kept around because UmfPackLU's
    // solve accesses the original matrix for iterative refinement.
    SuiteSparseMatrix m_mat;
    void *symbolic;
    void *numeric;
    double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
    double m_factorizationMemoryBytes;
};

using DefaultLUFactorizer = UmfpackFactorizer;
#else
class LUFactorizerStub {
public:
    const std::runtime_error no_lu_support = std::runtime_error("MeshFEM was compiled without LU factorization support");
    template<typename _Triplet>
    LUFactorizerStub(TripletMatrix<_Triplet> &/* tmat */) { }

    [[ noreturn ]] void factorize() { throw no_lu_support; }
    [[ noreturn ]] void factorizeSymbolic(int /* nmethods */) { throw no_lu_support; }
    template<typename _Triplet>
    [[ noreturn ]] void updateFactorization(const TripletMatrix<_Triplet> &/* tmat */) { throw no_lu_support; }

    template<typename _Vec1, typename _Vec2>
    void solve(const _Vec1 &, _Vec2 &) { throw no_lu_support; }

    double peakMemoryMB() const { throw no_lu_support; }

    void clear() { throw no_lu_support; }

    // Size of the factorized matrix.
    size_t m() const { return 0; }
    size_t n() const { return 0; }
};

using DefaultLUFactorizer = LUFactorizerStub;

#endif

} // namespace MeshFEM

#endif /* end of include guard: UMFPACKFACTORIZER_HH */
