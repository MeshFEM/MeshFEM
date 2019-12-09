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
#include <numeric>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <memory>
#include <cstdint>
#include <cmath>
#include "Parallelism.hh"

#include <MeshFEM/Types.hh>
#include <MeshFEM/GlobalBenchmark.hh>

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

    // Needed for triplet matrix binary read...
    Triplet() : i(0), j(0), v(0) { }

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

template<class TMat>
bool tripletsSortedAndUnique(const TMat &mat) {
    const auto  end = mat.end();
          auto prev = mat.begin();
          auto curr = mat.begin();
    for (++curr; curr != end; ++curr, ++prev) {
        if  ((*curr).j > (*prev).j) continue;
        if (((*curr).j < (*prev).j) || ((*curr).i <= (*prev).i)) return false;
    }
    return true;
}

template<typename _Triplet = Triplet<Real>>
struct TripletMatrix {
    typedef enum {APPEND_ABOVE, APPEND_BELOW,
                  APPEND_LEFT , APPEND_RIGHT} AppendPos;

    // Rudimentary support for tagging symmetric/nonsymmetric matrices. This
    // effects, e.g., the interpretation of matrix multiplication.
    enum class SymmetryMode { NONE, UPPER_TRIANGLE };
    SymmetryMode symmetry_mode = SymmetryMode::NONE;

    TripletMatrix(size_t mm = 0, size_t nn = 0) : m(mm), n(nn) { }

    typedef TripletMatrix<_Triplet>         TMatrix;
    typedef _Triplet                        Triplet;
    typedef typename _Triplet::value_type   Real;
    typedef Real                            value_type;
    size_t m, n;
    std::vector<Triplet> nz;
    // Set this to false for minor speed gains if you know that your matrix is
    // already properly sorted and has its repeated entries summed.
    // Warning: it is not automatically set back to true if the matrix is modified!
    // Use at your own risk.
    bool needs_sum_repeated = true;

    void init(size_t mm = 0, size_t nn = 0) {
        m = mm, n = nn;
        clear();
    }

    void clear() { nz.clear(); }
    void reserve(size_t nn) { nz.reserve(nn); }
    size_t nnz() const { return nz.size(); }
    void addNZ(size_t i, size_t j, Real v) {
        assert((i < m) && (j < n));
        if (v == Real(0.0)) return; // Possibly give this a tolerance...
        nz.push_back(Triplet(i, j, v));
    }

    // Sort and sum of repeated entries
    bool needsSumRepated() const { return needs_sum_repeated; }
    void sumRepeated() {
        if (!needsSumRepated()) { return; }

        BENCHMARK_SCOPED_TIMER_SECTION timer("Compress Matrix");
        if (tripletsSortedAndUnique(*this)) return;

        const size_t origNNZ = nz.size();

#define PARALLEL_BIN 0 // Parallel binning seems to actually slow things down...

        // Organize columns into buckets all stored contiguously in a vector.
        // First compute sizes and then the start/end of each bucket.
        // (bucketStart[j] is the start of bucket j and end of bucket j - 1)
#if PARALLEL_BIN
        auto bucketStart = std::unique_ptr<std::atomic<size_t>[]>(new std::atomic<size_t>[n + 1]);
        for (size_t i = 0; i < n + 1; ++i) bucketStart[i] = 0;
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, origNNZ),
            [&bucketStart, this](const tbb::blocked_range<size_t> &r) {
                for (size_t ti = r.begin(); ti < r.end(); ++ti)
                    ++bucketStart[nz[ti].j + 1];
        });
#else
        std::vector<size_t> bucketStart(n + 1, 0);
        for (size_t ti = 0; ti < origNNZ; ++ti)
            ++bucketStart[nz[ti].j + 1];
#endif
        // compute bucket offsets
        for (size_t j = 1; j <= n; ++j) // get bucket offsets
            bucketStart[j] += bucketStart[j - 1];
        assert(bucketStart[n] == nz.size());

        // Index of current end of bucket (initially at the start since buckets
        // are empty).
#if PARALLEL_BIN
        auto bucketEndIndex = std::unique_ptr<std::atomic<size_t>[]>(new std::atomic<size_t>[n + 1]);
        for (size_t i = 0; i < n + 1; ++i) bucketEndIndex[i] = bucketStart[i].load(); // atomic has no copy constructor
#else
        std::vector<size_t> bucketEndIndex(bucketStart);
#endif

        // Fill the buckets.
        // NOTE: the order of entries within each bucket is undefined when
        // multiple processors are used. This means that there will be a
        // nondeterministic roundoff error in both the matrix and the solution.
        // The roundoff error can be made deterministic by sorting the buckets
        // by value as well as row index (in fact, there's probably an order
        // that minimizes roundoff error).
        using CEntry = std::pair<size_t, Real>;
        std::vector<CEntry> columnBuckets(nz.size());
        auto placeInBucket = [&columnBuckets, &bucketEndIndex, this](size_t ti) {
            const auto &t = nz[ti];
            size_t newEntry = bucketEndIndex[t.j]++; // atomic!
            columnBuckets[newEntry].first  = t.i;
            columnBuckets[newEntry].second = t.v;
        };
#if PARALLEL_BIN
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, origNNZ),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t ti = r.begin(); ti < r.end(); ++ti) placeInBucket(ti);
        });
#else
        for (size_t ti = 0; ti < origNNZ; ++ti) placeInBucket(ti);
#endif

        for (size_t j = 0; j < n; ++j) // make sure we filled each bucket.
            assert(bucketEndIndex[j] == bucketStart[j + 1]);

        // Sort each bucket in parallel and sum repeated entries into the
        // nonzeros corresponding to the first few bucket entries.
        // Can be called in parallel for each bucket.
        auto sortAndSumBucket = [&columnBuckets, &bucketStart, this](size_t j) {
            size_t si = bucketStart[j],
                   ei = bucketStart[j + 1];
            size_t len = ei - si;
            if (len == 0) { return; }
            if (len == 1) { nz[si] = { columnBuckets[si].first, j, columnBuckets[si].second }; return; }

            std::sort(columnBuckets.begin() + si, columnBuckets.begin() + ei,
                      [](const CEntry &a, const CEntry &b) { return a.first < b.first; });

            size_t backIndex = si;
            nz[backIndex] = { columnBuckets[si].first, j, columnBuckets[si].second };
            for (size_t k = si + 1; k < ei; ++k) {
                if (nz[backIndex].i == columnBuckets[k].first)
                    nz[backIndex].v += columnBuckets[k].second;
                else nz[++backIndex] = { columnBuckets[k].first, j, columnBuckets[k].second };
            }
            // Mark the unused entries for deletion
            for (size_t k = backIndex + 1; k < ei; ++k)
                nz[k].v = 0;
        };

#if MESHFEM_WITH_TBB
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t j = r.begin(); j < r.end(); ++j) sortAndSumBucket(j);
        });
#else
        for (size_t j = 0; j < n; ++j) sortAndSumBucket(j);
#endif

        // remove identically zero entries (could use a tolerance)
        auto back = std::remove_if(nz.begin(), nz.end(),
                [](const Triplet &t) -> bool { return t.v == 0.0; });
        // std::cout << "removed " << std::distance(back, nz.end()) << " small entries" << std::endl;
        nz.erase(back, nz.end());
    }

    // Clear the current matrix and copy over only the upper triangle (including
    // diagonal) of B.
    template<class TMat> // note: TMat can be a CSCMatrix
    void setUpperTriangle(const TMat &B) {
        clear();
        m = B.m;
        n = B.n;
        // size_t numUpper = std::count_if(B.nz.begin(), B.nz.end(),
        //         [](const Triplet &t) -> bool { return t.i <= t.j; });
        // reserve(numUpper);
        reserve(B.nnz()); // faster and not too wasteful...
        for (const Triplet &t : B) {
            if (t.i <= t.j)
                nz.push_back(t);
        }

        symmetry_mode = SymmetryMode::UPPER_TRIANGLE;
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
        symmetry_mode = SymmetryMode::NONE;
    }

    // WARNING: Assumes sumRepeated() has already been called.
    template<typename _Index, typename _Real>
    void getCompressedColumn(_Index *Ap, _Index *Ai,
                             _Real *Ax) const {
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
        (void) (pad);
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

    void readBinary(const std::string &path) {
        std::ifstream is(path);
        if (!is.is_open()) throw std::runtime_error("Failed to open input file " + path);
        uint64_t N;
        is.read((char *) &N, sizeof(uint64_t));
        nz.resize(N);

        std::vector<uint64_t> indices(N);
        std::vector<double>   values(N);

        is.read((char *) &indices[0], N * sizeof(uint64_t));
        for (size_t i = 0; i < N; ++i) nz[i].i = indices[i];

        // Infer number of rows
        m = *max_element(indices.begin(), indices.end()) + 1;

        is.read((char *) &indices[0], N * sizeof(uint64_t));
        for (size_t i = 0; i < N; ++i) nz[i].j = indices[i];

        // Infer number of cols
        n = *max_element(indices.begin(), indices.end()) + 1;

        is.read((char *) &values[0], N * sizeof(double));
        for (size_t i = 0; i < N; ++i) nz[i].v = values[i];
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
        if (size_t(x.size()) != n) throw std::runtime_error("Sparse matvec size mismatch.");
        _Vector result(m);
        // Some _Vector types don't zero-initialize.
        for (size_t i = 0; i < size_t(result.size()); ++i) result[i] = 0.0;
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

    // Remove the rows and columns at particular indices. This is intended to
    // be called on a symmetric matrix, in which case solving the resulting
    // linear system (with corresponding entries deleted in the RHS vector as
    // well) effectively minimizing energy while fixing the deleted variables at 0.
    void rowColRemoval(const std::vector<size_t> &indices) {
        if (m != n) throw std::runtime_error("rowColRemoval supported for square matrices only");

        const size_t nvars = n;
        std::vector<bool> shouldRemove(nvars, false);
        for (size_t i : indices) shouldRemove[i] = true;
        rowColRemoval([&shouldRemove](size_t i) { return shouldRemove[i]; });
    }

    template<class Predicate>
    void rowColRemoval(const Predicate &shouldRemove) {
        // Remove the triplets for deleted variables
        auto back = std::remove_if(nz.begin(), nz.end(),
                [&shouldRemove](const Triplet &t) -> bool { return (shouldRemove(t.i) || shouldRemove(t.j)); });
        nz.erase(back, nz.end());

        // Calculate the new index of each kept variable
        const size_t nvars = n;
        std::vector<size_t> newIndex(nvars);
        size_t idx = 0;
        for (size_t i = 0; i < nvars; ++i)
            if (!shouldRemove(i)) newIndex[i] = idx++;

        // Update matrix size.
        m = n = idx;

        // Update the row/col indices for the kept triplets.
        for (Triplet &t : nz) {
            t.i = newIndex[t.i];
            t.j = newIndex[t.j];
        }
    }

    using VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    // Extract the diagonal (summing repeated entries)
    VXd diag() const {
        VXd result = VXd::Zero(m);
        for (const Triplet &t : nz)
            if (t.i == t.j) result[t.i] += t.v;
        return result;
    }

    // Gets the maximum entry on the matrix's diagonal.
    Real maxDiagEntry() const { return diag().max(); }

    // Permit simpler range-for syntax for over triplets
    auto begin() const -> decltype(nz.begin()) { return nz.begin(); }
    auto   end() const -> decltype(nz.  end()) { return nz.  end(); }
};

// Search for "i" in "Ai" at indices in the range "[lb, ub)"
template<typename _Index>
_Index binary_search(_Index i, _Index *Ai, _Index lb, _Index ub) {
    return std::distance(Ai, std::lower_bound(Ai + lb, Ai + ub, i));
#if 0
    while (ub - lb > 6) {
        _Index mid = (ub + lb) / 2;
        _Index row = Ai[mid];
        if (row == i) { return mid; }
        if (row <  i) { lb = mid; }
        if (row >  i) { ub = mid; }
    }
    for (; (Ai[lb] != i) && (lb != ub); ++lb);
    return lb;
#endif
}

// Free-standing implementation for insertion/accumulation of triplet (i, j, v)
// into a compressed column matrix. We assume that the entries
// within each column are sorted by row index and that an entry
// at (i, j) already exists in the matrix.
// Pointers are used so that we can directly modify matrices
// stored in Cholmod's internal arrays.
template<typename _Index, typename _Real>
size_t csc_add_nz(size_t /* nz */, _Index *Ai, _Index *Ap, _Real *Ax, _Index i, _Index j, _Real v) {
#if 1
    const _Index colend = Ap[j + 1];
    _Index idx = binary_search(i, Ai, Ap[j], colend);
    assert((idx != colend) && (Ai[idx] == i) && "Entry absent from sparsity pattern");

    // Accumulate value
    Ax[idx] += v;
    return idx + 1;
#else
    _Index idx, idxend = Ap[j + 1];
    for (idx = Ap[j]; idx < idxend; ++idx)
        if (Ai[idx] == i) { Ax[idx] += v; break; }
    assert(idx < idxend);
    return idx + 1;
#endif
}

template<typename _Index, typename _Real>
size_t csc_add_nz(size_t nz, _Index *Ai, _Index *Ap, _Real *Ax, _Index i, _Index j, _Real v, size_t hint) {
    if ((hint < nz) && (Ai[hint] == i) && (hint < size_t(Ap[j + 1])) && (hint >= size_t(Ap[j]))) {
        Ax[hint] += v;
        return hint + 1;
    }
    return csc_add_nz(nz, Ai, Ap, Ax, i, j, v);
}

// Matrix in Compressed Sparse Column format
template<typename _Index, typename _Real>
struct CSCMatrix {
    using index_type = _Index;
    using value_type = _Real;
    std::vector<_Index>  Ap, Ai; // Column pointer and row index arrays
                                 // Note: the row index array must be sorted!
    std::vector<_Real>   Ax;     // Value array
    _Index m, n, nz;             // Number of rows, columns, and nonzeros

    // Rudimentary support for tagging symmetric/nonsymmetric matrices (used by CSCMatrix::apply). This
    // effects, e.g., the interpretation of matrix multiplication.
    enum class SymmetryMode { NONE, UPPER_TRIANGLE };
    SymmetryMode symmetry_mode = SymmetryMode::NONE;
    static constexpr _Index INDEX_NONE = std::numeric_limits<_Index>::max();

    size_t nnz() const { return nz; }

    CSCMatrix(_Index mm = 0, _Index nn = 0)
        : m(mm), n(nn), nz(0) { }

    CSCMatrix(const CSCMatrix  &b) : Ap(b.Ap), Ai(b.Ai), Ax(b.Ax), m(b.m), n(b.n), nz(b.nz), symmetry_mode(b.symmetry_mode) { }
    CSCMatrix(      CSCMatrix &&b) : Ap(std::move(b.Ap)), Ai(std::move(b.Ai)), Ax(std::move(b.Ax)), m(b.m), n(b.n), nz(b.nz), symmetry_mode(b.symmetry_mode) { }

    template<typename T> CSCMatrix(TripletMatrix<T>  &mat) { setFromTMatrix(mat); }
    template<typename T> CSCMatrix(TripletMatrix<T> &&mat) { setFromTMatrix(std::move(mat)); }

    // Set each nonzero entry to a particular value, preserving the sparsity pattern.
    void fill(_Real val) { Ax.assign(nz, val); }
    void setZero() { fill(0.0); }

    void setIdentity(bool preserveSparsity = false) {
        if (m != n) throw std::runtime_error("Only square matrices are supported");
        if (preserveSparsity) {
            setZero();
            for (_Index i = 0; i < m; ++i)
                Ax[findDiagEntry(i)] = 1.0;
        }
        else {
            nz = m;
            Ap.resize(n + 1);
            Ai.resize(nz);
            Ax.assign(nz, 1.0);
            std::iota(Ap.begin(), Ap.end(), 0);
            std::iota(Ai.begin(), Ai.end(), 0);
        }
    }

    _Real trace() const {
        if (m != n) throw std::runtime_error("Trace called on non-square matrix!");
        _Real result = 0.0;
        for (_Index i = 0; i < m; ++i) {
            _Index idx = findDiagEntry<true>(i);
            if (idx == INDEX_NONE) continue;
            result += Ax[idx];
        }
        return result;
    }

    // Set this matrix to have the same sparsity pattern as b, but with zeros
    void zeros_like(const CSCMatrix &b) {
        m = b.m; n = b.n; nz = b.nz;
        Ap = b.Ap; Ai = b.Ai;
        Ax.assign(Ai.size(), 0.0);
    }

    template<bool _detectMissing = false>
    _Index findDiagEntry(_Index i) const {
        if (symmetry_mode == SymmetryMode::UPPER_TRIANGLE) {
            _Index idx = Ap[i + 1] - 1; // Diagonal element is the last entry in the column "i"
            if (_detectMissing && ((idx < Ap[i]) || (Ai[idx] != i))) return INDEX_NONE;
            assert((idx >= Ap[i]) && (Ai[idx] == i));
            return idx;
        }
        return findEntry<_detectMissing>(i, i);
    }

    template<bool _detectMissing = false>
    _Index findEntry(_Index i, _Index j) const {
        // Find the entry in the sparsity pattern.
        // Row indices are sorted, so we can use a binary search.
        auto beginIt = &Ai[0] + Ap[j],
               endIt = &Ai[0] + Ap[j + 1];
        auto it = std::lower_bound(beginIt, endIt, i);
        if (_detectMissing && (it == endIt)) return INDEX_NONE;
        assert((it != endIt) && "Entry absent from sparsity pattern");
        _Index idx = std::distance(&Ai[0], it);
        if (_detectMissing && (Ai[idx] != i)) return INDEX_NONE;
        assert((Ai[idx] == i) && "Entry absent from sparsity pattern");
        return idx;
    }

    // Accumulate a value to (i, j)
    // Note: (i, j) must exist in the sparsity pattern!
    // Complexity: O(log(n_j)) where "n_j" is the number of nonzeros in column j
    size_t addNZ(_Index i, _Index j, _Real v) {
        assert((i < m) && (j < n) && "Index out of bounds");
        return csc_add_nz(nz, Ai.data(), Ap.data(), Ax.data(), i, j, v);
    }

    // Insert (i, j, v), with a guess that it should go at location "hint"
    size_t addNZ(const _Index i, const _Index j, const _Real v, _Index hint) {
        if ((hint < Ap[j + 1]) && (Ai[hint] == i) && (hint >= Ap[j])) {
            Ax[hint] += v;
            return hint + 1;
        }
#if 0
        const _Index lb = Ap[j], ub = Ap[j + 1];
        if ((hint < ub) && (hint >= lb)) {
            _Index row = Ai[hint];
            if (row <  i) { hint = binary_search(i, Ai.data(), hint, ub); }
            if (row >  i) { hint = binary_search(i, Ai.data(), lb, hint); }
            Ax[hint] += v;
            return hint + 1;
        }
#endif
#if 0
        const _Index lb = Ap[j], ub = Ap[j + 1];
        if ((hint >= lb) && (hint < ub)) {
            long dist = long(i) - long(Ai[hint]);
            if (dist == 0) { Ax[hint] += v; return hint + 1; }
            // Still use the hint if it gets us close to row i.
            if (std::abs(dist) <= 5) {
                _Index step = std::copysign(1, dist);
                for (hint = hint + step; hint < ub; hint += step)
                    if (Ai[hint] == i) { Ax[hint] += v; return hint + 1; }
                throw std::runtime_error("fail");
            }
            return addNZ(i, j, v);
        }
#endif
        return addNZ(i, j, v);
    }

    // Add a vertical strip of contiguous nonzero values starting at (i, j),
    // return the index of the next nonzero entry after the written strip.
    // (so that the adjacent strip below can be written by directly calling addNZ(idx, values))
    template<class Derived>
    _Index addNZ(_Index i, _Index j, const Eigen::EigenBase<Derived> &values) {
        return addNZ(findEntry(i, j), values);
    }

    template<class Derived>
    _Index addNZ(_Index i, _Index j, const Eigen::EigenBase<Derived> &values, _Index hint) {
        if ((hint < nz) && (Ai[hint] == i) && (hint < Ap[j + 1]) && (hint >= Ap[j]))
            return addNZ(hint, values);
        return addNZ(i, j, values);
    }

    // Add a sequence of values to the compressed nonzero entries starting at "idx"
    template<class Derived>
    _Index addNZ(_Index idx, const Eigen::EigenBase<Derived> &values) {
        static_assert(Derived::ColsAtCompileTime == 1, "Only row vectors can be added with addNZ");
        Eigen::Map<Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data() + idx, values.size()) += values;
        return idx + values.size();
    }

    _Index addNZ(_Index idx, _Real val) {
        Ax[idx] += val;
        return idx + 1;
    }

    CSCMatrix &operator=(const CSCMatrix  &b) { Ap = b.Ap           ; Ai = b.Ai           ; Ax = b.Ax           ; m = b.m; n = b.n; nz = b.nz; symmetry_mode = b.symmetry_mode; return *this; }
    CSCMatrix &operator=(      CSCMatrix &&b) { Ap = std::move(b.Ap); std::move(Ai = b.Ai); std::move(Ax = b.Ax); m = b.m; n = b.n; nz = b.nz; symmetry_mode = b.symmetry_mode; return *this; }
    template<typename _Real2>
    CSCMatrix &operator=(const CSCMatrix<_Index, _Real2> &b) {
        Ap = b.Ap; Ai = b.Ai;
        Ax.clear();
        Ax.reserve(b.Ax.size());
        for (const auto &v : b.Ax) Ax.emplace_back(v);
        m = b.m; n = b.n; nz = b.nz; symmetry_mode = SymmetryMode(b.symmetry_mode);
        return *this;
    }

    _Real max()    const { return Eigen::Map<const Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data(), Ax.size()).maxCoeff(); }
    _Real absMax() const { return Eigen::Map<const Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data(), Ax.size()).cwiseAbs().maxCoeff(); }
    _Real maxRelError(CSCMatrix &b) const {
        CSCMatrix diff(*this);
        diff.addWithIdenticalSparsity(b, -1.0);
        diff.cwiseDivide(b);
        return diff.absMax();
    }

    // (*this) = (*this) ./ b, assuming b's sparsity pattern is identical to ours.
    // Entries that are zero in both matrices are left zero (even if they exist
    // in the sparsity pattern).
    void cwiseDivide(const CSCMatrix &b) {
        if (nz != b.nz) throw std::runtime_error("Mismatched sparisty patterns");
        for (_Index i = 0; i < nz; ++i) { if ((Ax[i] != 0) || (b.Ax[i] != 0)) Ax[i] /= b.Ax[i]; }
    }

    // (*this) += alpha * b, assuming b's sparsity pattern is identical to ours.
    void addWithIdenticalSparsity(const CSCMatrix &b, _Real alpha = 1.0) {
        if (alpha == 1.0) { Eigen::Map<Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data(), Ax.size()) +=         Eigen::Map<const Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(b.Ax.data(), b.Ax.size()); }
        else              { Eigen::Map<Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data(), Ax.size()) += alpha * Eigen::Map<const Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(b.Ax.data(), b.Ax.size()); }
    }

    void scale(_Real alpha) { Eigen::Map<Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data(), Ax.size()) *= alpha; }

    // (*this) += alpha * b, assuming b's sparsity pattern is a subset of ours.
    // offset: offset to be applied to the row and column indices of b
    void addWithSubSparsity(const CSCMatrix &b, const _Real alpha = 1.0, const _Index offset = 0) {
        auto it  = begin(), bit  = b.begin(),
             ite = end(),   bite = b.end();
        auto bi = [&]() { return offset + bit.get_i(); };
        auto bj = [&]() { return offset + bit.get_j(); };
        while ((it != ite) && (bit != bite)) {
            if ((it.get_j() == bj())) {
                if (it.get_i() == bi()) {
                    Ax[it.get_idx()] += alpha * b.Ax[bit.get_idx()];
                    ++it; ++bit;
                }
                else {
                    assert(it.get_i() < bi() && "b's sparsity not a subset of ours");
                    ++it;
                }
            }
            else {
                assert(it.get_j() < bj() && "b's sparsity not a subset of ours");
                ++it;
            }
        }
        assert(bit == bite && "b's sparsity not a subset of ours");
    }

    // Set from a triplet matrix
    // Side effect: mat's triplets are sorted and compressed.
    template<typename TMatrix>
    void setFromTMatrix(TMatrix &&mat) {
        symmetry_mode = static_cast<SymmetryMode>(mat.symmetry_mode);
        mat.sumRepeated();

        m = mat.m, n = mat.n;
        nz = mat.nnz();
        Ap.resize(n + 1);
        Ai.resize(nz);
        Ax.resize(nz);

        mat.getCompressedColumn(&Ap[0], &Ai[0], &Ax[0]);
    }

    // A sparse matrix holding "diag" on the diagonal.
    void setDiag(const Eigen::Ref<const Eigen::Matrix<_Real, Eigen::Dynamic, 1>> &diag, bool preserveSparsity = false) {
        if (preserveSparsity) {
            if ((size_t(m) != size_t(diag.size())) ||
                (size_t(n) != size_t(diag.size()))) throw std::runtime_error("Size mismatch");
            setZero();
            for (_Index i = 0; i < m; ++i)
                Ax[findDiagEntry(i)] = diag[i];
        }
        else {
            m = n = nz = diag.size();
            Ap.resize(n + 1);
            Ai.resize(nz);
            Ax.resize(nz);
            std::iota(Ap.begin(), Ap.end(), 0);
            std::iota(Ai.begin(), Ai.end(), 0);
            Eigen::Map<Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data(), Ax.size()) = diag;
        }
    }

    void sumRepeated() { /* nothing to do; here for compatibility with TripletMatrix interface */ }
    bool needsSumRepated() const { return false; }

    void dump(const std::string &path) const {
        std::ofstream cscout(path);
        cscout.precision(19);
        for (size_t i = 0; i < Ai.size(); ++i)
            cscout << Ai[i] << "\t" << Ax[i] << "\n";
        for (size_t i = 0; i < Ap.size(); ++i)
            cscout << Ap[i] << "\n";
    }

    ////////////////////////////////////////////////////////////////////////////
    // Iteration over the nonzero entries stored in this matrix, as Triplet<>s.
    ////////////////////////////////////////////////////////////////////////////
    struct TripletIterator {
        TripletIterator(const CSCMatrix &mat_, _Index idx_) : mat(mat_) {
            idx = idx_;
            if (idx < mat.nz) {
                // Find the column immediately AFTER the one containing "idx"; this is the first holding a greater nnz index than "idx".
                // This ensures that empty columns are skipped properly.
                auto nextCol = std::upper_bound(mat.Ap.begin(), mat.Ap.end(), idx);
                assert((nextCol != mat.Ap.begin()) && (nextCol != mat.Ap.end())); // We're guaranteed Ap[0] == 0 <= idx < mat.nz == Ap.back(), so upper_bound should have found a valid entry after the first.
                j = std::distance(mat.Ap.begin(), --nextCol);
            }
            else if (idx == mat.nz) { j = mat.n; } // end iterator
            else throw std::runtime_error("Index for constructing TripletIterator out of bounds.");
        }

        _Index get_idx() { return idx; }
        _Index get_i  () { return mat.Ai[idx]; }
        _Index get_j  () { return j; }
        _Real  get_val() { return mat.Ax[idx]; }

        Triplet<_Real> operator*() const { return Triplet<_Real>(mat.Ai[idx], j, mat.Ax[idx]); }
        bool operator==(const TripletIterator &b) const { return idx == b.idx; }
        bool operator!=(const TripletIterator &b) const { return !(*this == b); }
        // Preincrement
        TripletIterator &operator++() {
            ++idx;
            while ((j < mat.n) && (idx >= mat.Ap[j + 1])) ++j; // Advance column index to the column containing this triplet.
            if ((j >= mat.n) && (idx != mat.nz)) {
                std::cerr << "Ran out of column pointers when searching for entry idx " << idx << std::endl;
            }
            assert((j < mat.n) || (idx == mat.nz)); // We should only run out of column pointers when we reach the end of the triplets.
            return *this;
        }
    private:
        _Index idx, j; // nonzero and column index
                       // (column index is cached/updated for efficiency to avoid a search on each dereference)
        const CSCMatrix &mat;
    };

    TripletIterator begin() const{ return TripletIterator(*this,  0); }
    TripletIterator   end() const{ return TripletIterator(*this, nz); }

    // Matrix-vector multiply
    template<typename _Vector>
    _Vector apply(const _Vector &x, const bool transpose = false) const {
        const size_t local_m = transpose ? n : m;
        const size_t local_n = transpose ? m : n;
        if (size_t(x.size()) != size_t(local_n)) throw std::runtime_error("Sparse matvec size mismatch.");
        _Vector result(local_m);
        applyRaw(x.data(), result.data(), transpose);
        return result;
    }

    void applyRaw(const _Real *x, _Real *result, const bool transpose = false) const {
        const bool swapIndices = transpose && (symmetry_mode != SymmetryMode::UPPER_TRIANGLE);

        std::fill(result, result + (transpose ? n : m), 0.0);

        const auto ende = end();
        for (auto it = begin(); it != ende; ++it) {
            _Index i = it.get_i(), j = it.get_j();
            if (swapIndices) std::swap(i, j);
            result[i] += it.get_val() * x[j];
            if ((symmetry_mode == SymmetryMode::UPPER_TRIANGLE) && (i != j))
                result[j] += it.get_val() * x[i];
        }
    }

    // Remove the rows i and columns j for which remove[i] and remove[j] is true, respectively
    template<class Predicate>
    void rowColRemoval(const Predicate &shouldRemove) {
        if (m != n) throw std::runtime_error("rowColRemoval only implemented for square matrices");

        // Determine the mapping from old row indices to new (reduced) row indices.
        constexpr _Index NONE = std::numeric_limits<_Index>::max();
        std::vector<_Index> replacementRowIdx(n, NONE);
        size_t toRemove = 0;
        for (_Index reducedIdx = 0, i = 0; i < m; ++i) {
            if (shouldRemove(i)) { ++toRemove; continue; }
            replacementRowIdx[i] = reducedIdx++;
        }

        if (toRemove == 0) return;

        const _Index nconst = n;
        size_t entry_back = 0, colptr_back = 0;
        _Index idx_begin = 0; // Pointer to the beginning of the current column's entries (note Ap[j] will be overwritten by the updated end pointer for the column j - 1)
        for (_Index j = 0; j < nconst; ++j) {
            // Generate/filter column pointers
            if (shouldRemove(j)) { idx_begin = Ap[j + 1]; continue; } // Skip removed columns

            // Filter entries by row index
            const _Index idx_end = Ap[j + 1]; // Actually gives a measurable performance boost!
            for (_Index idx = idx_begin; idx < idx_end; ++idx) {
                const _Index i = Ai[idx];
                if (shouldRemove(i)) continue;
                Ai[entry_back] = replacementRowIdx[i];
                Ax[entry_back] = Ax[idx];
                ++entry_back;
            }
            idx_begin = idx_end;
            Ap[++colptr_back] = entry_back; // Write the new column end pointer for the kept columns
        }

        assert(colptr_back <= size_t(m));
        assert(entry_back <= size_t(nz));

        nz = entry_back;
        m = n = colptr_back;

        Ax.resize(nz);
        Ai.resize(nz);
        Ap.resize(n + 1);
    }

    // Remove from the sparsity pattern all entries that are identically zero
    void removeZeros() {
        // Process entries one column at a time.
        const _Index nconst = n;
        _Index entry_back = 0;
        _Index idx_begin = 0; // Pointer to the beginning of the current column's entries (note Ap[j] will be overwritten by the updated end pointer for the column j - 1)
        for (_Index j = 0; j < nconst; ++j) {
            // Generate/filter column pointers
            const _Index idx_end = Ap[j + 1]; // Actually gives a measurable performance boost!
            for (_Index idx = idx_begin; idx < idx_end; ++idx) {
                if (Ax[idx] == 0.0) continue;
                Ai[entry_back] = Ai[idx];
                Ax[entry_back] = Ax[idx];
                ++entry_back;
            }
            idx_begin = idx_end;
            Ap[j + 1] = entry_back; // Write the new column end pointer for the kept columns
        }
        nz = entry_back;
        Ax.resize(nz);
        Ai.resize(nz);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Conversion to sparse triplet formats ((I, J, V) arrays or TripletMatrix)
    ////////////////////////////////////////////////////////////////////////////
    template<typename I_, typename R_>
    void getIJV(const size_t n_, I_ *i, I_ *j, R_ *v) {
        if (n_ != size_t(nz)) throw std::runtime_error("Invalid output array sizes for getIJV");
        size_t back = 0;
        for (const auto &t : (*this)) {
            i[back] = t.i;
            j[back] = t.j;
            v[back] = t.v;
            ++back;
        }
    }

    template<typename I_, typename R_>
    void getIJV(std::vector<I_> &i, std::vector<I_> &j, std::vector<R_> &v) {
        i.resize(nz), j.resize(nz), v.resize(nz);
        getIJV(nz, i.data(), j.data(), v.data());
    }

    TripletMatrix<Triplet<_Real>> getTripletMatrix() const {
        using TM = TripletMatrix<Triplet<_Real>>;
        TM result(m, n);
        result.symmetry_mode = static_cast<typename TM::SymmetryMode>(symmetry_mode);
        result.reserve(nz);
        for (const auto &t : (*this)) result.nz.emplace_back(t);
        return result;
    }
};

using SuiteSparseMatrix = CSCMatrix<SuiteSparse_long, double>;

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

inline cholmod_dense cholmod_dense_wrap_vector_ptr(const size_t n, double *data) {
    cholmod_dense result;
    result.nrow = n;
    result.ncol = 1;
    result.nzmax = n;
    result.d = n; // leading dimension
    result.x = (void *) (data);
    result.z = NULL;
    result.xtype = CHOLMOD_REAL;
    result.dtype = CHOLMOD_DOUBLE;
    return result;
}

// Wrapper for a cholmod_sparse object allocated generated by Cholmod.
// Provides RAII resource management and supports matvecs.
struct CholmodSparseWrapper {
    CholmodSparseWrapper(size_t cols, cholmod_sparse *mat, std::shared_ptr<cholmod_common> c)
        : n(cols), m_mat(mat), m_c(c) {
            if (mat == nullptr) throw std::runtime_error("CholmodSparseWrapper constructed from null matrix");
            if (m_c == nullptr) throw std::runtime_error("CholmodSparseWrapper constructed with null cholmod_common");
        }

    CholmodSparseWrapper(const CholmodSparseWrapper &b) = delete;
    CholmodSparseWrapper(CholmodSparseWrapper &&b) : n(b.n), m_mat(b.m_mat), m_c(b.m_c) { b.m_mat = nullptr; }

    // Matrix-vector multiply
    template<typename _Vector>
    _Vector apply(const _Vector &x) const {
        if (x.size() != n) throw std::runtime_error("Sparse matvec size mismatch.");
        _Vector result(n);
        applyRaw(x.data(), result.data());
        return result;
    }

    // y = mat * x
    void applyRaw(const double *x, double *y, const bool transpose = false) const {
        if (m_mat == nullptr) throw std::runtime_error("No matrix to apply");

        // Wrap x, y values into a cholmod_dense struct
        auto cholx = cholmod_dense_wrap_vector_ptr(n, const_cast<double *>(x)); // Suitesparse won't actually modify the input vector data, so this const_cast should be safe.
        auto choly = cholmod_dense_wrap_vector_ptr(n, y);
        double alpha[2] = { 1.0, 0.0 };
        double beta [2] = { 0.0, 0.0 }; // y = alpha * mat * x + beta * y_init
        cholmod_l_sdmult(m_mat, transpose, alpha, beta, &cholx, &choly, m_c.get());
    }

    CholmodSparseWrapper &operator=(const CholmodSparseWrapper  &b) = delete;
    CholmodSparseWrapper &operator=(      CholmodSparseWrapper &&b) {
        n = b.n; m_mat = b.m_mat; m_c = std::move(b.m_c);
        b.m_mat = nullptr;
        return *this;
    }

    ~CholmodSparseWrapper() { if (m_mat != nullptr) cholmod_l_free_sparse(&m_mat, m_c.get()); }
private:
    size_t n;
    cholmod_sparse *m_mat;
    std::shared_ptr<cholmod_common> m_c; // Guaranteed non-null by construction/move assignment
};

class CholmodFactorizer {
public:
    // Assumes matrix is stored in the upper triangle!
    template<typename _Triplet>
    CholmodFactorizer(const TripletMatrix<_Triplet> &tmat, bool forceSupernodal = false, bool force_ll = false) : m_AStorage(TripletMatrix<_Triplet>(tmat)) { m_init(forceSupernodal, force_ll); }

    // Warning: modifies the passed triplet matrix, tmat!
    template<typename _Triplet>
    CholmodFactorizer(TripletMatrix<_Triplet> &tmat, bool forceSupernodal = false, bool force_ll = false) : m_AStorage(tmat)           { m_init(forceSupernodal, force_ll); }
    CholmodFactorizer(const SuiteSparseMatrix &mat,  bool forceSupernodal = false, bool force_ll = false) : m_AStorage(mat)            { m_init(forceSupernodal, force_ll); }
    CholmodFactorizer(SuiteSparseMatrix &mat,        bool forceSupernodal = false, bool force_ll = false) : m_AStorage(mat)            { m_init(forceSupernodal, force_ll); }
    CholmodFactorizer(SuiteSparseMatrix &&mat,       bool forceSupernodal = false, bool force_ll = false) : m_AStorage(std::move(mat)) { m_init(forceSupernodal, force_ll); }

    // Delete unsafe copy constructors/assignment.
    // This will also suppress creation of default move constructors/assignment.
    CholmodFactorizer(const CholmodFactorizer  &b) = delete;
    CholmodFactorizer &operator=(const CholmodFactorizer  &b) = delete;

    void factorize() {
        clearFactors();
        factorizeSymbolic();
        BENCHMARK_START_TIMER("CHOLMOD Numeric Factorize");
        int success = cholmod_l_factorize(&m_A, m_L, m_c.get());
        BENCHMARK_STOP_TIMER("CHOLMOD Numeric Factorize");
        if (!success)
            throw std::runtime_error("Factorize failed.");
        if (m_c->status == CHOLMOD_NOT_POSDEF)
            throw std::runtime_error("CHOLMOD detected non-positive definite matrix!");
        BENCHMARK_ADD_MESSAGE("Peak factorization memory (MB):\t" +
                              std::to_string(peakMemoryMB()));
    }

    // Perform only the symbolic factorization with the current system matrix
    // (useful this matrix holds the sparsity pattern that will be used for
    // many numeric factorizations).
    void factorizeSymbolic(int nmethods = 0 /* Cholmod's default */) {
        BENCHMARK_START_TIMER("CHOLMOD Symbolic Factorize");
        m_c->nmethods = nmethods;
        clearFactors();
        m_L = cholmod_l_analyze(&m_A, m_c.get());
        BENCHMARK_STOP_TIMER("CHOLMOD Symbolic Factorize");
    }

    // Recompute the numeric factorization using the new system matrix "tmat",
    // resuing the symbolic factorization. For this to work, it must have the same
    // sparsity pattern as the matrix for which the symbolic factorization was computed.
    // NOTE: The check of positive definite inside this function is not sufficient,
    //       since it just uses CHOLMOD's return status. If the diagonal entry of L 
    //       is negative, CHOLMOD will not complain about it. Use checkPosDef() to 
    //       further ensure it is spd.
    template<typename Mat>
    void updateFactorization(Mat &&mat, bool isInTryCatch=false) {
        if ((m_L != nullptr) && ((size_t(m_L->n) != size_t(mat.m)) || (size_t(m_L->n) != size_t(mat.n)))) throw std::runtime_error("Wrong matrix size"); // necessary, but not sufficient! Sparsity pattern must be a subset of original A's
        if (m_A.nzmax == 0) throw std::runtime_error("Cholmod matrix wasn't allocated.");
        if (mat.nnz() > size_t(m_A.nzmax)) throw std::runtime_error("Matrix has more nonzeros than the one passed to the constructor"); // again, necessary but not sufficient!

        m_AStorage = std::forward<Mat>(mat);
        m_matrixUpdated();

        if (!hasFactorization()) return; // no symbolic factorization was computed yet; nothing needs to be updated.

        BENCHMARK_START_TIMER("CHOLMOD Numeric Factorize");
        bool oldTryCatch = m_c->try_catch;
        m_c->try_catch = isInTryCatch;
        int success = cholmod_l_factorize(&m_A, m_L, m_c.get());
        m_c->try_catch = oldTryCatch;
        BENCHMARK_STOP_TIMER("CHOLMOD Numeric Factorize");
        if (!success)
            throw std::runtime_error("Factor update failed");
        // NOTE: Be careful. This check is not sufficient for ensuring positive definite.
        if (m_c->status == CHOLMOD_NOT_POSDEF)
            throw std::runtime_error("CHOLMOD detected non-positive definite matrix!");
    }

    // Solve Ax =     b when sys = CHOLMOD_A,
    //       Lx =     b when sys = CHOLMOD_L,
    //    L^T x =     b when sys = CHOLMOD_Lt,
    //        x = P   b when sys = CHOLMOD_P
    //        x = P^T b when sys = CHOLMOD_Pt
    template<typename _Vec1, typename _Vec2>
    void solve(const _Vec1 &b, _Vec2 &x, int sys = CHOLMOD_A) {
        assert(size_t(b.size()) == size_t(m_A.nrow));
        x.resize(m_A.ncol);
        solveRaw(&b[0], &x[0], sys);
    }

    template<typename _Vec>
    _Vec solve(const _Vec &b, int sys = CHOLMOD_A) {
        assert(size_t(b.size()) == size_t(m_A.nrow));
        _Vec x(m_A.ncol);
        solveRaw(&b[0], &x[0], sys);
        return x;
    }

    template<typename _Vec1, typename _Vec2>
    void solveExistingFactorization(const _Vec1 &b, _Vec2 &x, int sys = CHOLMOD_A) const {
        assert(size_t(b.size()) == size_t(m_A.nrow));
        x.resize(m_A.ncol);
        solveRawExistingFactorization(&b[0], &x[0], sys);
    }

    template<typename _Vec>
    _Vec solveExistingFactorization(const _Vec &b, int sys = CHOLMOD_A) const {
        assert(size_t(b.size()) == size_t(m_A.nrow));
        _Vec x(m_A.ncol);
        solveRawExistingFactorization(&b[0], &x[0], sys);
        return x;
    }

    void solveRawExistingFactorization(const Real *b, Real *x, int sys = CHOLMOD_A) const {
        if (!hasFactorization()) throw std::runtime_error("Factorization doesn't exist");
        static_assert(std::is_same<Real, double>::value, "Right-hand side must be an array of doubles");

        const size_t m = m_A.nrow, n = m_A.ncol;

        // Wrap b values into a cholmod_dense struct
        auto cholb = cholmod_dense_wrap_vector_ptr(m, const_cast<Real *>(b)); // Suitesparse won't actually modify the RHS data, so this const_cast should be safe.
        auto cholx = cholmod_dense_wrap_vector_ptr(n, x);
        auto cholx_ptr = &cholx;

        BENCHMARK_START_TIMER("CHOLMOD Backsub");
        // Solve A x = b re-using the workspace vectors x, Y, and E
        cholmod_l_solve2(sys, m_L, &cholb, NULL, &cholx_ptr, NULL, &m_Y, &m_E, m_c.get());

        if (cholx_ptr != &cholx) throw std::runtime_error("Cholmod reallocated x vector.");

        BENCHMARK_STOP_TIMER("CHOLMOD Backsub");
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRaw(const Real *b, Real *x, int sys = CHOLMOD_A) {
        if (!hasFactorization()) factorize();
        solveRawExistingFactorization(b, x, sys);
    }

    bool hasFactorization() const { return m_L != nullptr; }

    // Store a copy of the current factorization so that it can be applied again
    // even after updateFactorization is called.
    void stashFactorization() {
        if (m_L_stashed != nullptr) cholmod_l_free_factor(&m_L_stashed, m_c.get());
        m_L_stashed = cholmod_l_copy_factor(m_L, m_c.get());
    }

    bool hasStashedFactorization() const { return m_L_stashed != nullptr; }

    // Exchange the roles of m_L and m_L_stashed, making the stash the active factorization.
    void swapStashedFactorization() { std::swap(m_L, m_L_stashed); }

    // Get the (unpermuted) Cholesky factor L as a sparse matrix that can be applied
    // to a vector.
    CholmodSparseWrapper getL() {
        // According to the documentation, cholmod_copy_factor will convert our numeric
        // factorization m_L back into a symbolic one, which will break future solves.
        // So we operate on a copy of m_L.
        if (!hasFactorization()) throw std::runtime_error("Factorization doesn't exist");
        cholmod_factor *factorCopy = cholmod_l_copy_factor(m_L, m_c.get());
        if (factorCopy == nullptr) throw std::runtime_error("Factor copy failed");
        auto result = CholmodSparseWrapper(m_A.nrow, cholmod_l_factor_to_sparse(factorCopy, m_c.get()), m_c);
        cholmod_l_free_factor(&factorCopy, m_c.get());
        return result;
    }

    double peakMemoryMB() const {
        return ((double) m_c->memory_usage) / (1 << 20);
    }

    void clearFactors() {
        if (m_L)         { cholmod_l_free_factor(&m_L,         m_c.get()); m_L         = nullptr; }
        if (m_L_stashed) { cholmod_l_free_factor(&m_L_stashed, m_c.get()); m_L_stashed = nullptr; }
    }

    ~CholmodFactorizer() {
        clearFactors();

        if (m_Y) cholmod_l_free_dense(&m_Y, m_c.get());
        if (m_E) cholmod_l_free_dense(&m_E, m_c.get());

        cholmod_l_finish(m_c.get());
    }

    static void error_handler(int status, const char *file, int line, const char *message) {
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

    // Check if the matrix for which factor "L" was computed is positive definite.
    bool checkPosDef() const {
        if (!m_L) throw std::runtime_error("Matrix wasn't factorized");
        if (m_L->is_ll) return true; // LL^T factorization only succeeds if the matrix was positive definite
        // We have an LDL^T factorization; we need to check that all entries of D are positive.
        // Cholmod stores these entries on the diagonal of "L"
        const size_t numCols = m_L->n;
        assert(numCols == n());
        SuiteSparse_long *colPointers = (SuiteSparse_long *) m_L->p;
        double *values = (double *) m_L->x;
        assert((colPointers != nullptr) && (values != nullptr));
        for (size_t j = 0; j < numCols; ++j) {
            auto colBegin = colPointers[j];
            assert(colBegin < colPointers[j + 1]); // column better be nonempty!
            // Diagonal entry is the first entry of this column
            if (values[colBegin] <= 1e-16) {
                return false;
            }
        }
        return true;
    }

    // Size of the factorized matrix.
    size_t m() const { return m_A.nrow; }
    size_t n() const { return m_A.ncol; }

private:
    std::shared_ptr<cholmod_common> m_c;
    cholmod_sparse m_A;
    cholmod_factor *m_L = nullptr, *m_L_stashed = nullptr;

    mutable cholmod_dense *m_Y = nullptr, *m_E = nullptr; // result/workspace for cholmod_l_solve2

    SuiteSparseMatrix m_AStorage;

    void m_matrixUpdated() {
        m_A.p = m_AStorage.Ap.data();
        m_A.i = m_AStorage.Ai.data();
        m_A.x = m_AStorage.Ax.data();
    }

    void m_init(bool forceSupernodal, bool force_ll) {
        if (m_c) { cholmod_l_finish(m_c.get()); }
        m_c = std::make_shared<cholmod_common>();
        cholmod_l_start(m_c.get());

#ifdef TOO_LARGE_FOR_METIS
         // Use NESDIS since plain Metis is failing on large matrices.
         // This can be slower for some matrices, so we make this an option.
        m_c->default_nesdis = 1.0;
#endif

        if (forceSupernodal) m_c->supernodal = CHOLMOD_SUPERNODAL;
        m_c->final_ll = force_ll;

        // Try many different orderings searching for the best.
        // m_c->nmethods = 9;

        // Completely bypass Metis/NESDIS (for large matrices, this fails...)
        // Note: this shouldn't be done for smaller matrices because it results in slower solves.
        //// This version avoids Metis, but fails for even more matrices due to fill-in.
        //// m_c->nmethods = 1;
        //// m_c->method[0].ordering = CHOLMOD_AMD;
        //// m_c->postorder = 1; // TRUE
        //// m_c->error_handler = error_handler;
        //
        // // This puts us in LDL' mode
        // // "To factorize a large indefinite matrix, set Common->supernodal to
        // // CHOLMOD_SIMPLICIAL, and the simplicial LDL' method will always be
        // // used. This will be significantly slower than a supernodal LL'
        // // factorization, however.
        // m_c->supernodal = CHOLMOD_SIMPLICIAL;
        m_c->grow2 = 0; // We don't plan to use the modify routines
        m_c->quick_return_if_not_posdef = true;

        m_A.nrow   = m_AStorage.m;
        m_A.ncol   = m_AStorage.n;
        m_A.nzmax  = m_AStorage.nnz();
        m_A.nz     = nullptr; /* not needed because m_A is packed. */
        m_A.z      = nullptr; /* not needed because m_A is real. */
        m_A.stype  = 1; // upper triangle stored.
        m_A.itype  = CHOLMOD_LONG;
        m_A.xtype  = CHOLMOD_REAL;
        m_A.dtype  = CHOLMOD_DOUBLE;
        m_A.sorted = true;
        m_A.packed = true;

        m_matrixUpdated();
    }
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

    // Set a SPSD system.
    // Only use `keepFactorization = true` if K's sparsity pattern is a subset
    // of the original matrix factorized--then we re-use the original symbolic
    // factorization.
    template<class TMat> // TMatrix or SuiteSparseMatrix
    void set(const TMat &K, bool keepFactorization = false) {
        clear(keepFactorization);
        m_AUpper.setUpperTriangle(K);
        m_AUpper.needs_sum_repeated = K.needsSumRepated();
        m_isSPD = true;
        m_numVars = m_AUpper.m;

        m_initReducedVariables();
    }

    // Ensure the sparsity pattern is filled with 1 so that Cholmod knows where
    // all nonzeros are.
    void setSparsityPattern(SuiteSparseMatrix pat) {
        pat.fill(1.0);
        set(pat, false);
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
    // Only use `keepFactorization = true` if the resulting reduced matrix's
    // sparsity pattern a subset of the original matrix factorized--then we
    // re-use the original symbolic factorization.
    void fixVariables(const std::vector<size_t> &fixedVars,
                      const std::vector<_Real>  &fixedVarValues = std::vector<_Real>(), // variables fixed to zero if unspecified
                      bool keepFactorization = false) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("fixVariables");
        if (fixedVars.size() == 0) return;
        if ((fixedVarValues.size() != 0) && (fixedVarValues.size() != fixedVars.size())) throw std::runtime_error("Incorrect number of fixedVarValues");
        if (!keepFactorization) clearFactorization();
        else                    m_needsNumericFactorization = true;
        if (m_AUpper.nnz() == 0)
            throw std::runtime_error("Empty triplets--attempted to modify system post-solve in economy mode?");

        const bool fixToZero = fixedVarValues.size() == 0;

        // replacementIndex tracks what the current reduced variable indices are
        // remapped to. Initially it is used to flag (reduced) variables for
        // elimination (with -1), but afterward the full array is filled in.
        std::vector<int> replacementIndex(m_AUpper.m, 0);

        // The value to which each (reduced) variable will be fixed, or zero if
        // the variable will not be fixed. Needed for efficiently computing RHS
        // contribution of fixedVarValues
        std::vector<_Real> rvNewlyFixedValue;
        if (!fixToZero) {
            rvNewlyFixedValue.assign(m_AUpper.m, 0.0);
            for (size_t i = 0; i < fixedVars.size(); ++i) {
                int rv = m_reducedVarForVar[fixedVars[i]];
                if (rv < 0) continue;
                assert(size_t(rv) < rvNewlyFixedValue.size());
                rvNewlyFixedValue[rv] = fixedVarValues[i];
            }
        }

        // Mark fixed variables for elimination and store their values in
        // m_fixedVarValues for post-solve recovery.
        {
            int fixedVarIdx = m_fixedVarValues.size(); // index in the full collection of fixed variables (not just the ones added in this call...)
            m_fixedVarValues.resize(m_fixedVarValues.size() + fixedVars.size());
            for (size_t i = 0; i < fixedVars.size(); ++i) {
                size_t toFix = fixedVars[i];
                assert(toFix < m_reducedVarForVar.size());

                // Get the current reduced index of the variable.
                int curr = m_reducedVarForVar[toFix];
                if (curr < 0) throw std::runtime_error("Variable already fixed.");
                assert(size_t(curr) < replacementIndex.size());

                replacementIndex[curr] = -1;
                m_reducedVarForVar[toFix] = -1 - fixedVarIdx;
                if (!fixToZero) m_fixedVarValues[fixedVarIdx] = fixedVarValues[i];
                ++fixedVarIdx;
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

        if (!fixToZero) {
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
        }

        // Remove entries in the newly fixed rows/columns of A
        // and apply the reindexing to the remaining entries.
        {
            auto back = m_AUpper.nz.begin();
            for (auto it = m_AUpper.nz.begin(); it != m_AUpper.nz.end(); ++it) {
                const auto &t = *it;
                int i = replacementIndex[t.i];
                if (i < 0) continue;
                int j = replacementIndex[t.j];
                if (j < 0) continue;
                *back++ = Triplet<_Real>(i, j, t.v);
            }
            m_AUpper.nz.erase(back, m_AUpper.nz.end());
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

    void factorizeSymbolic(int nmethods = 0 /* Cholmod's default */) {
        if (m_isSPD) {
            BENCHMARK_START_TIMER_SECTION("Construct Factorizer");
            m_LLT = std::unique_ptr<_LLTFactorizer>(new _LLTFactorizer(m_AUpper, m_forceSupernodal));
            BENCHMARK_STOP_TIMER_SECTION("Construct Factorizer");

            m_LLT->factorizeSymbolic(nmethods);
            m_needsNumericFactorization = true;
        }
        else { throw std::runtime_error("Unimplemented"); }
    }

    // Solve K u = f under any existing constraints/fixed variables.
    template<class _Vec, class _SolnVec>
    void solve(const _Vec &f, _SolnVec &u) {
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
                BENCHMARK_START_TIMER_SECTION("Construct Factorizer");
                m_LLT = std::unique_ptr<_LLTFactorizer>(new _LLTFactorizer(m_AUpper, m_forceSupernodal));
                m_needsNumericFactorization = false;
                if (m_economyMode) m_clearAUpperTriplets();
                BENCHMARK_STOP_TIMER_SECTION("Construct Factorizer");
            }

            if (m_needsNumericFactorization) {
                m_LLT->updateFactorization(m_AUpper);
                m_needsNumericFactorization = false;
            }

            m_LLT->solve(bReduced, uReduced);
        }
        else {
            // Expand m_AUpper into a full matrix.
            if (!m_LU) {
                BENCHMARK_START_TIMER_SECTION("Construct Factorizer");
                TMatrix A;
                A.reserve(m_AUpper.nnz() + m_AUpper.strictUpperTriangleNNZ());
                A = m_AUpper;
                if (m_economyMode) m_clearAUpperTriplets();
                A.reflectUpperTriangle();
                m_LU = std::unique_ptr<_LUFactorizer>(new _LUFactorizer(A));
                m_needsNumericFactorization = false;
                BENCHMARK_STOP_TIMER_SECTION("Construct Factorizer");
            }
            if (m_needsNumericFactorization) {
                m_LU->updateFactorization(m_AUpper);
                m_needsNumericFactorization = false;
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

    bool checkPosDef() const {
        if (!m_LLT) throw std::runtime_error("Matrix wasn't factorized as LL or LDL.");
        return m_LLT->checkPosDef();
    }

    bool factorized() const {
        return (m_isSPD && m_LLT) || (!m_isSPD && m_LU);
    }

    void clearFactorization() {
        m_LU = NULL;
        m_LLT = NULL;
    }

    void clear(bool keepFactorization = false) {
        if (!keepFactorization) clearFactorization();
        m_needsNumericFactorization = true;
        m_AUpper.init(0, 0);
        m_numVars = 0;
        m_initReducedVariables();
    }

    // Note: changes to forceSupernodal only take effect for the next factorization.
    void setForceSupernodal(bool forceSupernodal) { m_forceSupernodal = forceSupernodal; }
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

    // Whether to force a supernodal factorization when using a Cholesky
    // factorization. This seems to be the only way to reliably detect
    // an indefinite matrix with CHOLMOD (if its heuristics decide to use
    // a simplicial factorization, then it typically succeeds in factorizing
    // an indefinite matrix).
    bool m_forceSupernodal = false;

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

    // If we update the matrix while requesting to `keepFactorization`, then
    // the factorization object already exists but must be updated before
    // solving.
    bool m_needsNumericFactorization = false;
};

#endif /* end of include guard: SPARSEMATRICES_HH */
