////////////////////////////////////////////////////////////////////////////////
// SparseMatrices.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Provides a simple triplet-based sparse marix class "TripletMatrix" that
//		supports conversion to umfpack/cholmod format.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/22/2014 16:40:42
////////////////////////////////////////////////////////////////////////////////
#ifndef SPARSEMATRICES_HH
#define SPARSEMATRICES_HH

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cassert>
#include <memory>
#include <cstdint>

#ifndef GLOBALBENCHMARK_HH
#include "BenchmarkStub.hh"
#endif

extern "C" {
#include <umfpack.h>
#include <cholmod.h>
}

#ifndef SuiteSparse_long
#define SuiteSparse_long UF_long
#endif

template<typename Real>
struct Triplet
{
    typedef Real value_type;
    size_t i, j;
    Real v;

    Triplet(size_t ii, size_t jj, Real vv)
        : i(ii), j(jj), v(vv) { }

    size_t &row() { return i; }
    size_t &col() { return j; }
    Real &value() { return v; }

    size_t row() const { return i; }
    size_t col() const { return j; }
    Real value() const { return v; }

    // (col, row) lexical ordering
    bool operator<(const Triplet &b) const {
        if (j != b.j)
            return j < b.j;
        return i < b.i;
    }
};

template<typename _Triplet = Triplet<Real>>
struct TripletMatrix {
    typedef enum {APPEND_ABOVE, APPEND_BELOW,
                  APPEND_LEFT , APPEND_RIGHT} AppendPos;

    // Rudimentary support for tagging symmetric/nonsymmetric matrices. This
    // effects, e.g., the interpretation of matrix multiplication.
    enum class SymmetryMode { NONE, UPPER_TRIANGLE };
    SymmetryMode symmetry_mode = SymmetryMode::NONE;

    TripletMatrix(size_t m = 0, size_t n = 0) : m(m), n(n) { }
    typedef TripletMatrix<_Triplet>         TMatrix;
    typedef _Triplet                        Triplet;
    typedef typename _Triplet::value_type   Real;
    typedef Real                            value_type;
    size_t m, n;
    std::vector<Triplet> nz;

    void init(size_t mm = 0, size_t nn = 0) {
        m = mm, n = nn;
        clear();
    }

    void clear() { nz.clear(); }
    void reserve(size_t n) { nz.reserve(n); }
    size_t nnz() const { return nz.size(); }
    void addNZ(size_t i, size_t j, Real v) {
        assert((i < m) && (j < n));
        if (std::abs(v) > 0) // Possibly give this a tolerance...
            nz.push_back(Triplet(i, j, v));
    }

    // Sort and sum of repeated entries
    void sumRepeated() {
        BENCHMARK_START_TIMER("Compress Matrix");

        // Organize columns into buckets all stored contiguously in a vector.
        // First compute the start/end of each bucket.
        // (bucketStart[j] is the start of bucket j and end of bucket j - 1)
        std::vector<size_t> bucketStart(n + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t ti = 0; ti < nz.size(); ++ti) {
            size_t &bucketEnd = bucketStart[nz[ti].j + 1];
#ifdef _OPENMP
#pragma omp atomic
#endif
            ++bucketEnd;
        }

        for (size_t j = 1; j <= n; ++j) // get bucket offsets
            bucketStart[j] += bucketStart[j - 1];
        assert(bucketStart[n] == nz.size());

        // Index of current end of bucket (initially at the start since buckets
        // are empty).
        std::vector<size_t> bucketEndIndex(bucketStart);

        // Fill the buckets.
        // NOTE: the order of entries within each bucket is undefined when
        // multiple processors are used. This means that there will be a
        // nondeterministic roundoff error in both the matrix and the solution.
        // The roundoff error can be made deterministic by sorting the buckets
        // by value as well as row index (in fact, there's probably an order
        // that minimizes roundoff error).
        typedef std::pair<size_t, Real> CEntry;
        std::vector<CEntry> columnBuckets(nz.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t ti = 0; ti < nz.size(); ++ti) {
            auto &t = nz[ti];
            size_t &end = bucketEndIndex[t.j];
            size_t newEntry;
#ifdef _OPENMP
 #pragma omp atomic capture
#endif
            newEntry = end++;
            auto &entry = columnBuckets[newEntry];
            entry.first = t.i;
            entry.second = t.v;
        }
        for (size_t j = 0; j < n; ++j) // make sure we filled each bucket.
            assert(bucketEndIndex[j] == bucketStart[j + 1]);

        int backIndex = -1;

        // Sort each bucket in parallel. 
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t j = 0; j < n; ++j) {
            std::sort(columnBuckets.begin() + bucketStart[j],
                      columnBuckets.begin() + bucketStart[j + 1],
                      [](const CEntry &a, const CEntry &b) { return a.first < b.first; });
        }

        // sum each bucket's repeated entries into nz array
        auto bucketBegin = columnBuckets.begin();
        for (size_t j = 0; j < n; ++j) {
            auto bucketEnd = columnBuckets.begin() + bucketStart[j + 1];
            if (bucketBegin == bucketEnd) continue;
            ++backIndex; // new column
            nz[backIndex].j = j;
            nz[backIndex].i = bucketBegin->first;
            nz[backIndex].v = bucketBegin->second;
            for (auto it = bucketBegin + 1; it != bucketEnd; ++it) {
                auto &backEntry = nz[backIndex];
                if (backEntry.i == it->first)
                    backEntry.v += it->second;
                else {
                    ++backIndex;
                    auto &newEntry = nz[backIndex];
                    newEntry.j = j;
                    newEntry.i = it->first;
                    newEntry.v = it->second;
                }
            }
            bucketBegin = bucketEnd; // move to next bucket
        }
        assert(size_t(backIndex) < nz.size());
        nz.erase(nz.begin() + backIndex + 1, nz.end());

        // // in-place sum of repeated entries
        // assert(nz.size() > 0);
        // auto back = nz.begin();
        // const size_t num_nz = nnz();
        // for (size_t i = 1; i < num_nz; ++i) {
        //     if (nz[i].row() == back->row() && nz[i].col() == back->col())
        //         back->value() += nz[i].value();
        //     else
        //         *(++back) = nz[i];
        // }
        // // back points to the last entry... delete the ones after it
        // nz.erase(++back, nz.end());

        // size_t oldSize = nz.size();
        // // remove small entries
        // back = std::remove_if(nz.begin(), nz.end(),
        //         [](const Triplet &t) -> bool { return std::abs(t.v) < 1e-14; });
        // nz.erase(back, nz.end());
        // std::cout << "removed " << oldSize - nz.size() << " small entries" << std::endl;

        BENCHMARK_STOP_TIMER("Compress Matrix");
    }

    // Clear the current matrix and copy over only the upper triangle (including
    // diagonal) of B.
    void setUpperTriangle(const TMatrix &B) {
        clear();
        m = B.m;
        n = B.n;
        size_t numUpper = std::count_if(B.nz.begin(), B.nz.end(),
                [](const Triplet &t) -> bool { return t.i <= t.j; });
        reserve(numUpper);
        for (const Triplet &t : B.nz) {
            if (t.i <= t.j)
                nz.push_back(t);
        }
    }

    void removeLowerTriangle() {
        auto back = std::remove_if(nz.begin(), nz.end(),
                [](const Triplet &t) -> bool { return t.i > t.j; });
        nz.erase(back, nz.end());
    }

    // Number of triplets in the strict upper triangle
    size_t strictUpperTriangleNNZ() const {
        return std::count_if(nz.begin(), nz.end(),
                [](const Triplet &t) -> bool { return t.i < t.j; });
    }

    // Replace the (strict) lower triangle with a copy of the upper triangle
    void reflectUpperTriangle() {
        removeLowerTriangle();
        size_t numStrictUpper = strictUpperTriangleNNZ();
        size_t oldSize = nnz();
        reserve(oldSize + numStrictUpper);
        for (size_t i = 0; i < oldSize; ++i) {
            const auto &t = nz[i];
            if (t.i < t.j)
                nz.push_back(Triplet(t.j, t.i, t.v));
        }
    }

    // WARNING: Assumes sumRepeated() has already been called.
    void getCompressedColumn(SuiteSparse_long *Ap, SuiteSparse_long *Ai,
                             double *Ax) const {
        const size_t num_nz = nnz();
        for (size_t i = 0; i < num_nz; ++i) {
            Ai[i] = nz[i].row();
            Ax[i] = nz[i].value();
        }

        // Compute column pointers
        Ap[0] = 0;
        size_t i = 0;
        for (size_t j = 0; j < n; ++j) {
            assert(i <= num_nz);
            assert((i == num_nz) || (j <= nz[i].col()));
            // Advance past this column's nonzeros
            while ((i < num_nz) && (nz[i].col() == j)) {
                ++i;
            }
            assert((i == num_nz) || (j < nz[i].col()));
            // Write column end index (next column's begin index)
            Ap[j + 1] = i;
        }

        assert(size_t(Ap[n]) == num_nz);
    }

    void setIdentity(size_t I_n) {
        m = n = I_n;
        nz.clear();
        nz.reserve(I_n);
        for (size_t i = 0; i < I_n; ++i)
            addNZ(i, i, 1);
    }

    TMatrix &operator*=(Real s) {
        for (Triplet &t: nz)
            t.v *= s;
        return *this;
    }

    TMatrix operator*(Real s) const {
        TMatrix result(*this);
        result *= s;
        return result;
    }

    // Re-index the variables in this symmetric matrix, A, by applying a
    // permutation-like matrix, S:
    //      x = S new_x
    //      newA = S^T A S ==> new_x^T newA new_x = new_x^T S^T A S new_x
    //           = x^T A x
    // where newA is a newNumVars x newNumVars matrix.
    // S is encoded in newVarIndexForVar (analogous to compressed row format)
    //      S_ij = 1 if j == newVarIndexForVar[i]
    //             0 otherwise
    // S could be a true permutation matrix, in which case the number of
    // variables is unchanged (newNumVars == A.m == A.n) and newVarIndexForVar
    // is a permutation of 0..(A.m - 1).
    //
    // Instead, S could represent a projection into a subspace whose basis
    // vectors (columns of S^T) have ones in at least one variable location
    // (and all other entries zero).
    // In this case, (newNumVars < A.m == A.n) and newVarIndexForVar will have
    // repeated values covering 0..(newNumVars - 1).
    void reindexVariables(size_t newNumVars,
                          const std::vector<size_t> &newVarIndexForVar) {
        if (m != n) throw std::runtime_error("reindexVariables on non-square (nonsymmetric) matrix.");
        if (newVarIndexForVar.size() != m) throw std::runtime_error("Invalid newVarIndexForVar size.");
        if (symmetry_mode == SymmetryMode::UPPER_TRIANGLE) {
            for (auto &t : nz) {
                // Validate that the matrix is upper-triangle-only
                if (t.i > t.j) throw std::runtime_error("Symmetry mode violated.");
                t.i = newVarIndexForVar.at(t.i);
                t.j = newVarIndexForVar.at(t.j);
                // We must maintain the upper-triangle storage in the
                // reduce/permuted variables: if a value was permuted into the
                // lower triangle, switch to storing its upper-triangle pair.
                if (t.i > t.j) std::swap(t.i, t.j);

                if ((t.i >= newNumVars) || (t.j >= newNumVars))
                    throw std::runtime_error("New variable index out of bounds.");
            }
        }
        else {
            // Symmetry properties are more expensive to validate--let's just
            // trust the user.
            for (auto &t : nz) {
                t.i = newVarIndexForVar.at(t.i);
                t.j = newVarIndexForVar.at(t.j);
                if ((t.i >= newNumVars) || (t.j >= newNumVars))
                    throw std::runtime_error("New variable index out of bounds.");
            }
        }

        m = n = newNumVars;
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Append another matrix above, below, to the left, or to the right of this
    //  one.
    //  @param[in]  B           Matrix with which to aument this matrix.
    //  @param[in]  pos         Where in this matrix to place B.
    //  @param[in]  pad         Whether to allow padding
    //  @param[in]  transpose   Whether to transpose B before appending.
    *///////////////////////////////////////////////////////////////////////////
    void append(const TMatrix &B, AppendPos pos, bool pad = false,
                bool transpose = false) {
        size_t Bm = transpose ? B.n : B.m, Bn = transpose ? B.m : B.n;

        switch (pos) {
            case APPEND_ABOVE: {
                assert((n == Bn) || (pad && (n >= Bn)));

                nz.reserve(nnz() + B.nnz());
                for (Triplet &t: nz)
                    t.i += Bm;
                if (transpose) {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.col(), t.row(), t.value()));
                }
                else {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.row(), t.col(), t.value()));
                }

                m += Bm;
                break;
            }
            case APPEND_BELOW:
                assert((n == Bn) || (pad && (n >= Bn)));

                reserve(nnz() + B.nnz());

                if (transpose) {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.col() + m, t.row(), t.value()));
                }
                else {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.row() + m, t.col(), t.value()));
                }

                m += Bm;
                break;
            case APPEND_LEFT: {
                assert((m == Bm) || (pad && (m >= Bm)));

                nz.reserve(nnz() + B.nnz());
                for (Triplet &t: nz)
                    t.j += Bn;

                if (transpose) {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.col(), t.row(), t.value()));
                }
                else {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.row(), t.col(), t.value()));
                }

                n += Bn;
                break;
            }
            case APPEND_RIGHT:
                assert((m == Bm) || (pad && (m >= Bm)));

                reserve(nnz() + B.nnz());

                if (transpose) {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.col(), t.row() + n, t.value()));
                }
                else {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.row(), t.col() + n, t.value()));
                }

                n += Bn;
                break;
            default:
                assert(false);
        }
    }

    void dump(const std::string &path) const {
        std::ofstream outFile(path);
        outFile << std::setprecision(20);
        if (!outFile.is_open()) {
            std::cout << "Failed to open output file '"
                      << path << '\'' << std::endl;
        }
        else{
            for (size_t i = 0; i < nnz(); ++i) {
                outFile << nz[i].i << '\t' << nz[i].j << '\t'
                        << nz[i].v << std::endl;
            }  
        }
    }


    // Much more efficient matrix dumping--output in a binary format:
    // number of nonzeros (uint64)
    // Row indices...     (each uint64)
    // Col indices...     (each uint64)
    // Values...          (each double)
    // Note, this won't necessarily be portable across architectures...
    void dumpBinary(const std::string &path) const {
        std::ofstream os(path);
        if (!os.is_open()) throw std::runtime_error("Failed to open output file " + path);
        uint64_t N = nnz();
        os.write((char *) &N, sizeof(uint64_t));

        std::vector<uint64_t> indices(N);
        for (size_t i = 0; i < N; ++i) indices[i] = nz[i].i;
        os.write((char *) &indices[0], N * sizeof(uint64_t));

        for (size_t i = 0; i < N; ++i) indices[i] = nz[i].j;
        os.write((char *) &indices[0], N * sizeof(uint64_t));

        std::vector<double> values(N);
        for (size_t i = 0; i < N; ++i) values[i] = nz[i].v;
        os.write((char *) &values[0], N * sizeof(double));
    }

    void read(std::ifstream &is) {
        std::string line;
        nz.clear();
        size_t maxi = 0, maxj = 0;
        while (std::getline(is, line)) {
            size_t i, j;
            double v;
            std::stringstream ss(line);
            ss >> i >> j >> v;
            if (ss)
                nz.push_back(Triplet(i, j, v));
            else
                std::cout << "WARNING: couldn't parse line '" << line << "'"
                          << std::endl;
            maxi = std::max(maxi, i);
            maxj = std::max(maxj, j);
        }

        // Deduce matrix size from the triplets.
        m = maxi + 1; n = maxj + 1;
    }

    // Matrix-vector multiply (not so efficient).
    template<typename _Vector>
    _Vector apply(const _Vector &x) const {
        _Vector result(x.size());
        // Some _Vector types don't zero-initialize.
        for (size_t i = 0; i < result.size(); ++i) result[i] = 0.0;
        if (symmetry_mode == SymmetryMode::NONE) {
            for (const Triplet &t: nz)
                result[t.i] += t.v * x[t.j];
        }
        else if (symmetry_mode == SymmetryMode::UPPER_TRIANGLE) {
            for (const Triplet &t: nz) {
                if (t.i < t.j) {
                    result[t.i] += t.v * x[t.j];
                    result[t.j] += t.v * x[t.i];
                }
                else if (t.i == t.j)
                    result[t.i] += t.v * x[t.j];
                else throw std::runtime_error("Symmetry mode violated.");
            }
        }
        else throw std::runtime_error("Unsupported matrix symmetry mode");
        return result;
    }
};

struct SuiteSparseMatrix {
    std::vector<SuiteSparse_long>  Ap, Ai;
    std::vector<double>            Ax;
    SuiteSparse_long m, n, nz;

    SuiteSparseMatrix()
        : m(0), n(0), nz(0) { }

    template<typename TMatrix>
    SuiteSparseMatrix(TMatrix &mat) { setFromTMatrix(mat); }

    // Set from a triplet matrix
    // Side effect: mat's triplets are sorted and compressed.
    template<typename TMatrix>
    void setFromTMatrix(TMatrix &mat) {
        mat.sumRepeated();

        m = mat.m, n = mat.n;
        nz = mat.nnz();
        Ap.resize(n + 1);
        Ai.resize(nz);
        Ax.resize(nz);

        mat.getCompressedColumn(&Ap[0], &Ai[0], &Ax[0]);
    }
};

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

class CholmodFactorizer {
public:
    template<typename _Triplet>
    CholmodFactorizer(const TripletMatrix<_Triplet> &tmat)
        : m_A(NULL), m_L(NULL) {
        TripletMatrix<_Triplet> mat(tmat);
        mat.removeLowerTriangle();
        mat.sumRepeated();

        cholmod_l_start(&m_c);
#ifdef TOO_LARGE_FOR_METIS
         // Use NESDIS since plain Metis is failing on large matrices.
         // This can be slower for some matrices, so we make this an option.
        m_c.default_nesdis = 1.0;
#endif

        // Completely bypass Metis/NESDIS (for large matrices, this fails...)
        // Note: this shouldn't be done for smaller matrices because it results in slower solves.
        //// This version avoids Metis, but fails for even more matrices due to fill-in.
        //// m_c.nmethods = 1;
        //// m_c.method[0].ordering = CHOLMOD_AMD;
        //// m_c.postorder = 1; // TRUE
        //// m_c.error_handler = error_handler;
        //
        // // This puts us in LDL' mode
        // // "To factorize a large indefinite matrix, set Common->supernodal to
        // // CHOLMOD_SIMPLICIAL, and the simplicial LDL' method will always be
        // // used. This will be significantly slower than a supernodal LL'
        // // factorization, however.
        // m_c.supernodal = CHOLMOD_SIMPLICIAL;
        // m_c.grow2 = 0; // We don't plan to use the modify routines

        m_A = cholmod_l_allocate_sparse(mat.m, mat.n, mat.nnz(),
                true,           // Row indices in each column are sorted
                true,           // packed
                1,              // Symmetry type (0: full matrix stored,
                                //                1: upper triangle stored
                                //                2: lower triangle stored)
                CHOLMOD_REAL,   // Keep it real
                &m_c);

        mat.getCompressedColumn((SuiteSparse_long *) m_A->p,
                (SuiteSparse_long *) m_A->i, (double *) m_A->x);
    }

    void factorize() {
        clearFactors();
        BENCHMARK_START_TIMER("CHOLMOD Symbolic Factorize");
        m_L = cholmod_l_analyze(m_A, &m_c);
        BENCHMARK_STOP_TIMER("CHOLMOD Symbolic Factorize");
        BENCHMARK_START_TIMER("CHOLMOD Numeric Factorize");
        int success = cholmod_l_factorize(m_A, m_L, &m_c);
        BENCHMARK_STOP_TIMER("CHOLMOD Numeric Factorize");
        if (!success)
            throw std::runtime_error("Factorize failed.");
        if (m_c.status == CHOLMOD_NOT_POSDEF)
            throw std::runtime_error("CHOLMOD detected non-positive definite matrix!");
        BENCHMARK_ADD_MESSAGE("Peak factorization memory (MB):\t" +
                              std::to_string(peakMemoryMB()));
    }

    template<typename _Vec1, typename _Vec2>
    void solve(const _Vec1 &b, _Vec2 &x) {
        if (m_L == NULL) factorize();

        size_t m = m_A->nrow, n = m_A->ncol;
        assert(b.size() == m);
        cholmod_dense *chol_b = cholmod_l_allocate_dense(n, 1,
                n,            // Leading dimension
                CHOLMOD_REAL, // Keep it real
                &m_c);

        for (size_t i = 0; i < m; ++i)
            ((double *) chol_b->x)[i] = b[i];

        BENCHMARK_START_TIMER("CHOLMOD Backsub");
        cholmod_dense *chol_x = cholmod_l_solve(CHOLMOD_A, m_L, chol_b, &m_c);
        BENCHMARK_STOP_TIMER("CHOLMOD Backsub");

        x.resize(n);
        for (size_t i = 0; i < n; ++i)
            x[i] = ((double *) chol_x->x)[i];

        cholmod_l_free_dense(&chol_x, &m_c);
        cholmod_l_free_dense (&chol_b, &m_c);
    }

    double peakMemoryMB() const {
        return ((double) m_c.memory_usage) / (1 << 20);
    }

    void clearFactors() {
        if (m_L) cholmod_l_free_factor(&m_L, &m_c);
    }

    ~CholmodFactorizer() {
        clearFactors();
        if (m_A) cholmod_l_free_sparse(&m_A, &m_c);
        cholmod_l_finish(&m_c);
    }

    static void error_handler(int status, const char *file, int line,
            const char *message) {
        std::cout << "Caught error." << std::endl;
        if (status < 0)
            throw std::runtime_error("Cholmod error in " + std::string(file) + ", line " +
                    std::to_string(line) + ": " + message + "( status " +
                    std::to_string(status) + ")");
        if (status > 0)
            std::cout << "Cholmod warning in " << file << ", line " << line
                      << ": " << message << "( status "
                      << std::to_string(status) << ")" << std::endl;
    }

    // Size of the factorized matrix.
    size_t m() const { return m_A->nrow; }
    size_t n() const { return m_A->ncol; }

private:
    cholmod_common m_c;
    cholmod_sparse *m_A;
    cholmod_factor *m_L;
};

////////////////////////////////////////////////////////////////////////////////
/*! Wraps a (constrained) SPSD system that can be solved for several
//  different righthand sides. The constraint RHS is specified at system setup
//  time, so only the unconstrained RHS is specified for each solve. Lagrange
//  multipliers are used for general linear constraints. For example, for system
//  "K u = f" with constraints C, we have the following terminology:
//
//  [ K C'] [u     ]   [ f     ]
//  [ C   ] [lambda] = [ C_rhs ]
//  -- A -- - u_l -    --  b  --
//  ONLY THE UPPER TRIANGLE OF K IS REFERENCED.
//
//  When Lagrange multipliers are used, the full system matrix is indefinite.
//  This means a Cholesky factorization can only be used on unconstrained
//  systems.
//
//  However, single variable constraints can be implemented with the
//  fixVariables() call that removes DoFs, giving a smaller, SPD system. If all
//  constraints are in this form then a Cholesky factorization can be used.
//
//  Calls to fixVariables() result in a smaller system for "reduced variables."
//  However, solve() takes and returns the full, unreduced RHS and solution.
*///////////////////////////////////////////////////////////////////////////////
template<typename _Real, class _LUFactorizer = UmfpackFactorizer,
                         class _LLTFactorizer = CholmodFactorizer>
class SPSDSystem {
public:
    typedef TripletMatrix<Triplet<_Real>> TMatrix;
    SPSDSystem() { }

    SPSDSystem(const TMatrix &K, const TMatrix &C, const std::vector<_Real> &C_rhs)
    { setConstrained(K, C, C_rhs); }
    SPSDSystem(const TMatrix &K) { set(K); }

    void setConstrained(const TMatrix &K, const TMatrix &C, const std::vector<_Real> &C_rhs) {
        clear();

        // Build the upper triangle of the system matrix.
        assert(C.m == C_rhs.size());
        m_AUpper.setUpperTriangle(K);
        m_AUpper.m += C.m;
        // Append's boolean arguments:             pad    transpose
        m_AUpper.append(C, TMatrix::APPEND_RIGHT,  true,  true);

        m_constraintRHS = C_rhs;
        // If no constraint rows were specified, the system is still SPD/SPSD.
        m_isSPD = (C.m == 0);
        m_numVars = m_AUpper.m;

        m_initReducedVariables();
    }

    // set a SPSD system
    void set(const TMatrix &K) {
        clear();
        m_AUpper.setUpperTriangle(K);
        m_isSPD = true;
        m_numVars = m_AUpper.m;

        m_initReducedVariables();
    }

    // The constraint RHS can be updated without refactoring.
    void setConstraintRHS(const std::vector<_Real> &constraintRHS) {
        if (m_constraintRHS.size() != constraintRHS.size())
            throw std::runtime_error("Invalid constraint RHS");
        m_constraintRHS = constraintRHS;
    }

    // Note: in economy mode, we could have cleared m_AUpper's triplets before
    // factorizing.
    bool isSet() const { return factorized() || (m_AUpper.nnz() != 0); }

    // Eliminate DoFs in fixedVars from the system. The system matrix is shrunk,
    // and variables are re-indexed in a way that the original system's solution
    // can be returned from the solve() call.
    void fixVariables(const std::vector<size_t> &fixedVars,
                      const std::vector<_Real>  &fixedVarValues) {
        assert(fixedVars.size() == fixedVarValues.size());
        if (fixedVars.size() == 0) return;
        clearFactorization();
        if (m_AUpper.nnz() == 0)
            throw std::runtime_error("Empty triplets--attempted to modify system post-solve in economy mode?");

        // replacementIndex tracks what the current reduced variable indices are
        // remapped to. Initially it is used to flag (reduced) variables for
        // elimination (with -1), but afterward the full array is filled in.
        std::vector<int> replacementIndex(m_AUpper.m, 0);

        // The value to which each (reduced) variable will be fixed, or zero if
        // the variable will not be fixed. Needed for efficiently computing RHS
        // contribution of fixedVarValues
        std::vector<_Real> rvNewlyFixedValue(m_AUpper.m, 0.0);
        for (size_t i = 0; i < fixedVars.size(); ++i) {
            int rv = m_reducedVarForVar[fixedVars[i]];
            if (rv < 0) continue;
            assert(size_t(rv) < rvNewlyFixedValue.size());
            rvNewlyFixedValue[rv] = fixedVarValues[i];
        }

        // Mark fixed variables for elimination and store their values
        // m_fixedVarValues for post-solve recovery.
        // Also move fixedVarValues[i] over to m_reducedVarForVar
        for (size_t i = 0; i < fixedVars.size(); ++i) {
            size_t toFix = fixedVars[i];
            assert(toFix < m_reducedVarForVar.size());

            // Get the current reduced index of the variable.
            int curr = m_reducedVarForVar[toFix];
            if (curr < 0) throw std::runtime_error("Variable already fixed.");
            assert(size_t(curr) < replacementIndex.size());

            replacementIndex[curr] = -1;
            m_reducedVarForVar[toFix] = -1 - int(m_fixedVarValues.size());
            _Real val = fixedVarValues[i];
            m_fixedVarValues.push_back(val);
        }

        // Move fixedVarValues[i]'s terms over to m_fixedVarRHSContribution
        // (essentially "elimination", but triplets are left in m_AUpper for now)
        for (const auto &t : m_AUpper.nz) {
            // Move over the upper triangle term...
            _Real val = rvNewlyFixedValue[t.j];
            if (val != 0.0) m_fixedVarRHSContribution[t.i] -= t.v * val;
            // and the strict lower triangle term.
            if (t.i < t.j) {
                val = rvNewlyFixedValue[t.i];
                if (val != 0.0) m_fixedVarRHSContribution[t.j] -= t.v * val;
            }
        }

        // Reindex all the current reduced variables.
        size_t newIdx = 0;
        for (size_t i = 0; i < m_AUpper.m; ++i) {
            if (replacementIndex[i] >= 0)
                replacementIndex[i] = newIdx++;
        }

        // Apply replacement to m_reducedVarForVar.
        for (size_t i = 0; i < m_numVars; ++i) {
            int curr = m_reducedVarForVar[i];
            if (curr < 0) continue;
            assert(size_t(curr) < replacementIndex.size());
            m_reducedVarForVar[i] = replacementIndex[curr];
        }

        // Remove entries (rows, cols) of A
        auto newEnd = std::remove_if(m_AUpper.nz.begin(), m_AUpper.nz.end(),
            [&](const Triplet<_Real> &t) -> bool {
                return (replacementIndex[t.i] < 0) ||
                       (replacementIndex[t.j] < 0); });
        m_AUpper.nz.erase(newEnd, m_AUpper.nz.end());

        // Apply replacement to A matrix.
        for (Triplet<_Real> &t : m_AUpper.nz) {
            t.i = replacementIndex[t.i];
            t.j = replacementIndex[t.j];
        }

        // Shrink A matrix to account for removed rows/cols.
        m_AUpper.m -= fixedVars.size();
        m_AUpper.n -= fixedVars.size();

        // Remove rows of m_fixedVarRHSContribution
        // (It will be added to the RHS of the **reduced** system.)
        auto back = m_fixedVarRHSContribution.begin();
        for (size_t i = 0; i < m_fixedVarRHSContribution.size(); ++i) {
            if (replacementIndex[i] >= 0)
                *back++ = m_fixedVarRHSContribution[i];
        }
        m_fixedVarRHSContribution.erase(back, m_fixedVarRHSContribution.end());
        assert(m_fixedVarRHSContribution.size() == m_AUpper.m);
    }

    // Solve K u = f under any existing constraints/fixed variables.
    template<class _Vec>
    void solve(const _Vec &f, std::vector<_Real> &u) {
        // number of non-Lagrange multiplier variables
        size_t nPrimaryVars = f.size();

        if (!isSet()) throw std::runtime_error("No system to solve");
        if (nPrimaryVars + m_constraintRHS.size() != m_numVars) throw std::runtime_error("Bad RHS");

        // Reduced system rhs (reduced f and  Lagrange multipliers)
        // Exploits symmetry of system (identical indexing of variables and
        // equations).
        std::vector<_Real> bReduced(m_AUpper.m, 0);
        for (size_t v = 0; v < m_reducedVarForVar.size(); ++v) {
            int r = m_reducedVarForVar[v];
            if (r < 0) continue;
            assert(size_t(r) < bReduced.size());
            bReduced[r] =
                ((v < nPrimaryVars) ? f[v] : m_constraintRHS[v - nPrimaryVars])
                    + m_fixedVarRHSContribution[r];
        }

        // Allocate space for solution + Lagrange multipliers
        std::vector<_Real> uReduced(m_AUpper.m);

        // { 
        //     m_AUpper.dump("A.txt");
        //     static int solve = 0;
        //     std::ofstream rhsOut("rhs_" + std::to_string(solve));
        //     rhsOut << std::scientific << std::setprecision(16);
        //     for (_Real val : bReduced) {
        //         rhsOut << val << std::endl;
        //     }
        //     ++solve;
        //     // exit(-1);
        // }

        if (m_isSPD) {
            if (!m_LLT) {
                m_LLT = std::unique_ptr<_LLTFactorizer>(new _LLTFactorizer(m_AUpper));
                if (m_economyMode) m_clearAUpperTriplets();
            }

            m_LLT->solve(bReduced, uReduced);
        }
        else {
            // Expand m_AUpper into a full matrix.
            if (!m_LU) {
                TMatrix A;
                A.reserve(m_AUpper.nnz() + m_AUpper.strictUpperTriangleNNZ());
                A = m_AUpper;
                if (m_economyMode) m_clearAUpperTriplets();
                A.reflectUpperTriangle();
                m_LU = std::unique_ptr<_LUFactorizer>(new _LUFactorizer(A));
            }
            m_LU->solve(bReduced, uReduced);
        }

        // Read off solution (but not the Lagrange multipliers)
        u.resize(nPrimaryVars);
        for (size_t v = 0; v < nPrimaryVars; ++v) {
            int r = m_reducedVarForVar[v];
            if (r < 0) {
                size_t fixedVar = -1 - r;
                assert(fixedVar < m_fixedVarValues.size());
                u[v] = m_fixedVarValues[fixedVar];
            }
            else {
                assert(size_t(r) < uReduced.size());
                u[v] = uReduced[r];
            }
        }
    }

    template<class _Vec>
    std::vector<_Real> solve(const _Vec &f) {
        std::vector<_Real> u;
        solve(f, u);
        return u;
    }

    bool factorized() const {
        return (m_isSPD && m_LLT) || (!m_isSPD && m_LU);
    }

    void clearFactorization() {
        m_LU = NULL;
        m_LLT = NULL;
    }

    void clear() {
        clearFactorization();
        m_AUpper.init(0, 0);
        m_numVars = 0;
        m_initReducedVariables();
    }

    void setEconomyMode(bool emode) { m_economyMode = emode; }
    bool economyMode() const { return m_economyMode; }

    void dumpUpper(const std::string &path) const {
        if (economyMode())
            std::cerr << "WARNING: attempting to dump system triplet matrix in "
                      << "economy mode--may be empty." << std::endl;
        m_AUpper.dumpBinary(path);
    }

    void sumAndDumpUpper(const std::string &path) {
        if (economyMode())
            std::cerr << "WARNING: attempting to dump system triplet matrix in "
                      << "economy mode--may be empty." << std::endl;
        m_AUpper.sumRepeated();
        m_AUpper.dumpBinary(path);
    }

    ~SPSDSystem() { clear(); }
private:
    // Initialize the reduced variables arrays, clearing any fixed variables.
    // Must be called every time the system changes!
    void m_initReducedVariables() {
        assert(m_AUpper.m == m_numVars);
        m_reducedVarForVar.resize(m_numVars);
        // Identity mapping of variables to reduced variables.
        for (size_t i = 0; i < m_numVars; ++i)
            m_reducedVarForVar[i] = i;
        m_fixedVarRHSContribution.assign(m_numVars, 0.0);
        m_fixedVarValues.clear();
    }

    // Keep matrix size information, but clear out contents.
    void m_clearAUpperTriplets() {
        m_AUpper.nz.clear();
        m_AUpper.nz.shrink_to_fit();
    }

    bool m_isSPD = false;
    std::vector<_Real> m_constraintRHS;

    // Whether we're in "economy mode." In economy mode, the triplet
    // form of the system is zero-ed out the moment a factorization object has
    // been built from it to avoid the storage of redundant copies. However,
    // the system cannot be modified (e.g. fixing variables) after a
    // factorization call in this mode.
    bool m_economyMode = false;

    // Track fixed variables after fixVariables have been called.
    // >=  0: index of reduced variable corresponding to a variable
    // <= -1: encoded index of value for a fixed (eliminated) variable
    std::vector<int> m_reducedVarForVar;
    std::vector<_Real> m_fixedVarValues;
    // Store the RHS contribution caused by fixing variables to nonzero values.
    // (i.e. by moving the variable's term in each equation to the RHS).
    // This is stored as vector contribution to the **reduced** system RHS.
    std::vector<_Real> m_fixedVarRHSContribution;

    // (Reduced) system matrix's upper triangle in triplet form.
    TMatrix m_AUpper;

    // Number of full system variables (including Lagrange multipliers).
    size_t m_numVars;
    std::unique_ptr<_LUFactorizer>  m_LU;
    std::unique_ptr<_LLTFactorizer> m_LLT;
};

#endif /* end of include guard: SPARSEMATRICES_HH */
