////////////////////////////////////////////////////////////////////////////////
// SparseMatrices.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Provides a simple triplet-based sparse marix class "TripletMatrix" that
//		supports conversion to umfpack/choldmod format.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/22/2014 16:40:42
////////////////////////////////////////////////////////////////////////////////
#ifndef SPARSEMATRICES_HH
#define SPARSEMATRICES_HH

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cassert>

template<typename Real>
struct Triplet
{
    typedef Real value_type;
    size_t i, j;
    Real v;

    Triplet(size_t ii, size_t jj, Real vv)
        : i(ii), j(jj), v(vv) { }

    size_t row() const { return i; }
    size_t col() const { return j; }
    Real value() const { return v; }

    // (col, row) lexical ordering
    bool operator<(const Triplet &b) const {
        if (col() != b.col())
            return col() < b.col();
        return row() < b.row();
    }
};

template<typename _Triplet>
struct TripletMatrix {
    typedef enum {APPEND_ABOVE, APPEND_BELOW,
                  APPEND_LEFT , APPEND_RIGHT} AppendPos;
    TripletMatrix(size_t m = 0, size_t n = 0) : m(m), n(n) { }
    typedef TripletMatrix<_Triplet>         TMatrix;
    typedef _Triplet                        Triplet;
    typedef typename _Triplet::value_type   Real;
    typedef Real                            value_type;
    size_t m, n;
    std::vector<Triplet> nz;

    void clear() { nz.clear(); }
    void reserve(size_t n) { nz.reserve(n); }
    size_t nnz() const { return nz.size(); }
    void addNZ(size_t i, size_t j, Real v) {
        assert((i < m) && (j < n));
        nz.push_back(Triplet(i, j, v));
    }

    void sort() {
        std::sort(nz.begin(), nz.end());
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

    void dump(const char *path) const {
        std::ofstream outFile(path);
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
};

extern "C" {
#include <umfpack.h>
}

#ifndef SuiteSparse_long
#define SuiteSparse_long UF_long
#endif

struct SuiteSparseMatrix {
    std::vector<SuiteSparse_long>  Ap, Ai;
    std::vector<double>            Ax;
    SuiteSparse_long m, n, nz;

    SuiteSparseMatrix()
        : m(0), n(0), nz(0) { }

    template<typename TMatrix>
    SuiteSparseMatrix(TMatrix &mat) { setFromTMatrix(mat); }

    // Set from a triplet matrix
    // Side effect: mat's triplets are sorted in order of increasing columns
    template<typename TMatrix>
    void setFromTMatrix(TMatrix &mat) {
        mat.sort();

        m = mat.m;
        n = mat.n;

        // Sum up repeated entries
        std::vector<size_t> rows, cols;
        std::vector<double> vals;
        rows.reserve(mat.nnz()); cols.reserve(mat.nnz());
        vals.reserve(mat.nnz());

        assert(mat.nz.size() > 0);
        rows.push_back(mat.nz[0].row());
        cols.push_back(mat.nz[0].col());
        vals.push_back(mat.nz[0].value());
        for (size_t i = 1; i < mat.nnz(); ++i) {
            if (mat.nz[i].row() == rows.back() &&
                mat.nz[i].col() == cols.back()) {
                vals.back() += mat.nz[i].value();
            }
            else {
                rows.push_back(mat.nz[i].row());
                cols.push_back(mat.nz[i].col());
                vals.push_back(mat.nz[i].value());
            }
        }

        nz = rows.size();

        // Copy row indices and values
        Ai.resize(nz);
        Ax.resize(nz);
        for (size_t i = 0; i < nz; ++i) {
            Ai[i] = rows[i];
            Ax[i] = vals[i];
        }

        // Compute column pointers
        Ap.resize(n + 1);
        Ap[0] = 0;
        size_t i = 0;
        for (size_t j = 0; j < n; ++j) {
            assert(i <= nz);
            assert((i == nz) || (j <= cols[i]));
            // Advance past this column's nonzeros
            while ((i < nz) && (cols[i] == j)) {
                ++i;
            }
            assert((i == nz) || (j < cols[i]));
            // Write column end index (next column's begin index)
            Ap[j + 1] = i;
        }

        if (Ap[n] != nz) {
            std::cout << "Ap[n]: " << Ap[n] << ", nz: " << nz << std::endl;
            assert(false);
        }
    }
};

class UmfpackFactorizer {
public:
    UmfpackFactorizer(SuiteSparseMatrix &mat)
        : m_mat(mat), symbolic(NULL), numeric(NULL) {
        umfpack_dl_defaults(Control);
        int status = umfpack_dl_symbolic(mat.m, mat.n, Ap(), Ai(), Ax(),
                                         &symbolic, Control, Info);
        if (status != UMFPACK_OK) {
            // Symbolic object isn't created when there is a failure, so there
            // is nothing to free.
            throw std::runtime_error("Umfpack symbolic factorization failed: "
                    + std::to_string(status));
        }

        status = umfpack_dl_numeric(Ap(), Ai(), Ax(), symbolic, &numeric,
                                    Control, Info);
        if (status != UMFPACK_OK) {
            umfpack_dl_free_symbolic(&symbolic);
            // A numeric object is allocated if we just got the singular matrix
            // warning, so we better free it. In all other cases, no object is
            // created.
            if (status == UMFPACK_WARNING_singular_matrix)
                umfpack_dl_free_numeric(&numeric);
            throw std::runtime_error("Umfpack numeric factorization failed: "
                    + std::to_string(status));
        }
    }

    void solve(const std::vector<double> &b, std::vector<double> &x) {
        assert(b.size() == m_mat.m);
        x.resize(m_mat.n);
        int status = umfpack_dl_solve(UMFPACK_A, Ap(), Ai(), Ax(), &x[0], &b[0],
                                      numeric, Control, Info);
        if (status != UMFPACK_OK) {
            throw std::runtime_error("Umfpack solve failed: "
                    + std::to_string(status));
        }
    }

    ~UmfpackFactorizer() {
        if (symbolic)
            umfpack_dl_free_symbolic(&symbolic);
        if (numeric)
            umfpack_dl_free_numeric(&numeric);
    }

private:
    size_t m() const { return m_mat.m; }
    size_t n() const { return m_mat.m; }
    const SuiteSparse_long *Ap() const { return &m_mat.Ap[0]; }
    const SuiteSparse_long *Ai() const { return &m_mat.Ai[0]; }
    const double *Ax()           const { return &m_mat.Ax[0]; }

    void *symbolic;
    void *numeric;
    double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
    SuiteSparseMatrix &m_mat;
};

extern "C" {
#include <cholmod.h>
}

class CholmodFactorizer {
public:
    CholmodFactorizer(SuiteSparseMatrix &mat)
        : m_mat(mat), m_A(NULL), m_L(NULL), m_b(NULL) {
        cholmod_l_start(&m_c);
        m_c.error_handler = error_handler;

        m_A = cholmod_l_allocate_sparse(mat.m, mat.n, mat.nz,
                true,           // Row indices in each column are sorted
                true,           // packed
                1,              // Symmetry type (0: full matrix stored,
                                //                1: upper triangle stored
                                //                2: lower triangle stored)
                CHOLMOD_REAL,   // Keep it real
                &m_c);

        // Create an extra copy of the matrix data--in the future we should
        // probably pass a TMatrix directly to CholmodFactorizer to avoid the
        // copy in SuiteSparseMatrix mat.
        for (size_t i = 0; i <= mat.n; ++i)
            ((SuiteSparse_long *) m_A->p)[i] = mat.Ap[i];
        for (size_t i = 0; i < mat.nz; ++i) {
            ((SuiteSparse_long *) m_A->x)[i] = mat.Ax[i];
            ((double *) m_A->x)[i] = mat.Ax[i];
            ((SuiteSparse_long *) m_A->i)[i] = mat.Ai[i];
        }

        m_L = cholmod_l_analyze(m_A, &m_c);
        int success = cholmod_l_factorize(m_A, m_L, &m_c);
        if (!success)
            throw std::runtime_error("Factorize failed.");
    }

    void solve(const std::vector<double> &b, std::vector<double> &x) {
        size_t m = m_A->nrow, n = m_A->ncol;
        assert(b.size() == m);
        m_b = cholmod_l_allocate_dense(n, 1,
                n,            // Leading dimension
                CHOLMOD_REAL, // Keep it real
                &m_c);

        for (size_t i = 0; i < m; ++i)
            ((double *) m_b->x)[i] = b[i];

        cholmod_dense *chol_x = cholmod_l_solve(CHOLMOD_A, m_L, m_b, &m_c);

        x.resize(n);
        for (size_t i = 0; i < n; ++i)
            x[i] = ((double *) chol_x->x)[i];

        cholmod_l_free_dense(&chol_x, &m_c);
    }

    ~CholmodFactorizer() {
        if (m_A) cholmod_l_free_sparse(&m_A, &m_c);
        if (m_L) cholmod_l_free_factor(&m_L, &m_c);
        if (m_b) cholmod_l_free_dense (&m_b, &m_c);
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
private:
    cholmod_common m_c;
    cholmod_sparse *m_A;
    cholmod_factor *m_L;
    cholmod_dense  *m_b;
    SuiteSparseMatrix &m_mat;
};

#endif /* end of include guard: SPARSEMATRICES_HH */
