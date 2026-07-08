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
#include <cstdint>
#include <MeshFEMCore/Parallelism.hh>
#include <MeshFEMCore/ParallelVectorOps.hh>
#include <MeshFEMCore/Utilities/binary_search.hh>
#include <MeshFEMCore/unused.hh>

#include <MeshFEMCore/Types.hh>
#include <MeshFEMCore/GlobalBenchmark.hh>
#include <MeshFEMCore/AutomaticDifferentiation.hh>

#if MESHFEM_WITH_CHOLMOD
#include <SuiteSparse_config.h>
#else
#ifndef SuiteSparse_long
using SuiteSparse_long = long;
#endif
#endif

namespace MeshFEM {

template<typename Real>
struct Triplet
{
    using value_type = Real;
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

namespace spmat_helper {
    // Helper structs and functions useful for implementing algorithms in a
    // uniform syntax that work when matrix entries are scalars or dense
    // blocks.

    template<typename T, class Enable = void>
    struct value_traits;

    template<typename T, class Enable = void>
    struct ContainerType { using type = std::vector<T>; }; // assume alignement unnecessary

    template<typename EigenT>
    struct ContainerType<EigenT, std::enable_if_t<sizeof(EigenT) % 16 == 0>> {
        using type = aligned_std_vector<EigenT>; // alignment needed
    };

    // Plain arithmetic type version
    template<typename T>
    struct value_traits<T, std::enable_if_t<std::is_arithmetic<T>::value>> {
        static constexpr size_t rows = 1;
        static constexpr size_t cols = 1;
        using Scalar = T;
        using container_type = typename ContainerType<T>::type;
        static Scalar valueMagnitudeSq(const T &v) { return v * v; }
        static void setZero(T &v) { v = 0; }
        static void setZero(T *v, size_t count) { Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(v, count).setZero(); }
        static T Zero() { return 0.0; }
    };

    // Eigen autodiff type version
    template<typename StorageType>
    struct value_traits<Eigen::AutoDiffScalar<StorageType>> {
        using T = Eigen::AutoDiffScalar<StorageType>;
        static constexpr size_t rows = 1;
        static constexpr size_t cols = 1;
        using Scalar = T;
        using container_type = typename ContainerType<T>::type;
        static Scalar valueMagnitudeSq(const T &v) { return v * v; }
        static void setZero(T &v) { v = 0; }
        static void setZero(T *v, size_t count) { for (size_t i = 0; i < count; ++i) setZero(v[i]); }
        static T Zero() { return 0.0; }
    };

    // Eigen type version
    template<typename EigenT>
    struct value_traits<EigenT, std::enable_if_t<isEigenType<EigenT>()>> {
        static constexpr size_t rows = EigenT::RowsAtCompileTime;
        static constexpr size_t cols = EigenT::ColsAtCompileTime;
        using Scalar = typename EigenT::Scalar;
        using container_type = typename ContainerType<EigenT>::type;
        static Scalar valueMagnitudeSq(const EigenT &v) { return v.squaredNorm(); }
        static void setZero(EigenT &v) {
            static_assert((rows != Eigen::Dynamic) && (cols != Eigen::Dynamic), "Only fixed-size blocks are currently supported");
            v.setZero();
        }
        static void setZero(EigenT *v, size_t count) { for (size_t i = 0; i < count; ++i) setZero(v[i]); }
        static auto Zero() { return EigenT::Zero().eval(); }
    };

    // Get an N-dimensional segment of a vector of this number type
    // 1-dimensional segments are just raw scalars.
    template<size_t N, class VType, class Enable = void>
    struct SegmentGetter;

    // Get vector-valued entries of a block vector.
    template<size_t N, class VType>
    struct SegmentGetter<N, VType, std::enable_if_t<(N > 1) && isEigenType<typename VType::value_type>()>> {
        using ElemType = std::decay_t<decltype(std::declval<VType>()[0])>;
        using ScratchVec = Eigen::Matrix<typename ElemType::Scalar, N, 1>; // Storage-backed version of ElemType
        static_assert((ElemType::RowsAtCompileTime == N) && (ElemType::ColsAtCompileTime == 1), "Inner block elements must already be of the correct vector type.");
        template<typename T> static auto  get(const T &v, size_t i) { return v[i]; }
        template<typename T> static auto &get(      T &v, size_t i) { return v[i]; }
    };

    // Get rows of an `Eigen::Dynamic x N` matrix.
    template<size_t N, class VType>
    struct SegmentGetter<N, VType, std::enable_if_t<(N > 1) && (VType::ColsAtCompileTime == N) && !isEigenType<typename VType::value_type>()>> {
        using ScratchVec = Eigen::Matrix<typename VType::Scalar, N, 1>;
        template<typename T> static auto get(const T &V, size_t i) { return V.row(i).transpose(); }
        template<typename T> static auto get(      T &V, size_t i) { return V.row(i).transpose(); }
    };

    // Get segments of an ordinary (flattened) vector.
    template<size_t N, class VType>
    struct SegmentGetter<N, VType, std::enable_if_t<(N > 1) && (VType::ColsAtCompileTime == 1) && !isEigenType<typename VType::value_type>()>> {
        using ScratchVec = Eigen::Matrix<typename VType::Scalar, N, 1>;
        template<typename T> static auto get(const T &v, size_t i) { return v.template segment<N>(N * i); }
        template<typename T> static auto get(      T &v, size_t i) { return v.template segment<N>(N * i); }
    };

    // Get a 1-dimensional segment of an ordinary vector (i.e., just a scalar)
    template<class VType>
    struct SegmentGetter<1, VType> {
        using ScratchVec = typename VType:: Scalar;
        template<typename T> static auto  get(const T &v, size_t i) { static_assert(std::is_arithmetic<typename T::Scalar>::value || isAutoDiffType<typename T:: Scalar>(), "Scalar type must be arithmetic!"); return v[i]; }
        template<typename T> static auto &get(      T &v, size_t i) { static_assert(std::is_arithmetic<typename T::Scalar>::value || isAutoDiffType<typename T:: Scalar>(), "Scalar type must be arithmetic!"); return v[i]; }
    };

    template<typename T> void          setZero(     T &&v) {        value_traits<std::decay_t<T>>::setZero(v); }
    template<typename T> void      setZero(T *v, size_t c) {        value_traits<std::decay_t<T>>::setZero(v, c); }
    template<typename T> auto valueMagnitudeSq(const T &v) { return value_traits<std::decay_t<T>>::valueMagnitudeSq(v); }

    // Transpose of a dense matrix block
    template<class Derived> auto transpose_block(const Eigen::MatrixBase<Derived> &A) { return A.transpose(); }

    // Transpose of scalar
    template<typename T> std::enable_if_t<std::is_arithmetic<T>::value, T> transpose_block(T a) { return a; }
}

template<typename _Triplet = Triplet<Real>>
struct TripletMatrix {
    typedef enum {APPEND_ABOVE, APPEND_BELOW,
                  APPEND_LEFT , APPEND_RIGHT} AppendPos;

    // Rudimentary support for tagging symmetric/nonsymmetric matrices. This
    // effects, e.g., the interpretation of matrix multiplication.
    enum class SymmetryMode { NONE, UPPER_TRIANGLE };
    SymmetryMode symmetry_mode = SymmetryMode::NONE;

    static constexpr size_t INDEX_NONE = std::numeric_limits<size_t>::max();

    TripletMatrix(size_t mm = 0, size_t nn = 0) : m(mm), n(nn) { }

    typedef TripletMatrix<_Triplet>         TMatrix;
    typedef _Triplet                        Triplet;
    typedef typename _Triplet::value_type   Real;
    typedef Real                            value_type;
    size_t m, n;
    aligned_std_vector<Triplet> nz;

    decltype(spmat_helper::valueMagnitudeSq(std::declval<Real>())) pruneTol = 0.0;

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
    void addNZUnpruned(size_t i, size_t j, Real v) {
        assert((i < m) && (j < n));
        nz.emplace_back(i, j, v);
    }
    void addNZ(size_t i, size_t j, Real v) {
        if (spmat_helper::valueMagnitudeSq(v) <= this->pruneTol) return;
        addNZUnpruned(i, j, v);
    }

    void addDiagEntry(size_t i, Real v) { addNZ(i, i, v); }

    // Add a vertical strip of contiguous nonzero values starting at (i, j),
    // (For compatibility with CSCMatrix interface--we can't actually gain a speedup here.)
    template<class Derived>
    int addNZStrip(long i, long j, const Eigen::DenseBase<Derived> &values, int hint = -1) {
        for (int ii = 0; ii < values.rows(); ++ii)
            addNZ(i + ii, j, values[ii]);
        return hint;
    }

    // Add a block of contiguous nonzero values starting at (i, j)
    // (For compatibility with CSCMatrix interface--we can't actually gain a speedup here.)
    // Note: if symmetry_mode UPPER_TRIANGLE and and we are adding a diagonal
    // block, only the upper triangle is added.
    template<class Derived>
    void addNZBlock(long i, long j, const Eigen::DenseBase<Derived> &values) {
        if ((symmetry_mode == SymmetryMode::UPPER_TRIANGLE) && (i == j)) {
            for (int jj = 0; jj < values.cols(); ++jj)
                for (int ii = 0; (ii < values.rows()) && (ii <= jj); ++ii)
                    addNZ(i + ii, j + jj, values(ii, jj));
        }
        else {
            for (int jj = 0; jj < values.cols(); ++jj)
                for (int ii = 0; ii < values.rows(); ++ii)
                    addNZ(i + ii, j + jj, values(ii, jj));
        }
    }

    // FOR COMPATIBILITY WITH CSCMatrix only!
    int findEntry(long /* i */, long /* j */) const {
        return -1;
    }

    // FOR COMPATIBILITY WITH CSCMatrix only!
    template<bool /* _knownGood */ = true>
    int addNZAtLoc(long i, long j, const Real &v, int loc) {
        addNZUnpruned(i, j, v);
        return loc;
    }


    // Sort and sum of repeated entries
    bool needsSumRepated() const { return needs_sum_repeated && (nz.size() > 1); }
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
        parallel_for_range(origNNZ, [&](size_t ti) {
            ++bucketStart[nz[ti].j + 1];
        }
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
        parallel_for_range(origNNZ, [&](size_t ti) { placeInBucket(ti); });
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
                nz[k].i = INDEX_NONE;
        };

        parallel_for_range(n, [&](size_t j) { sortAndSumBucket(j); });

        // remove identically zero entries (numerical tolerance)
        auto back = std::remove_if(nz.begin(), nz.end(),
                [this](const Triplet &t) -> bool { return (spmat_helper::valueMagnitudeSq(t.v) <= this->pruneTol) || t.i == INDEX_NONE; });
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
    template<typename _Index, typename _RowIndex, typename _Real>
    void getCompressedColumn(_Index *Ap, _RowIndex *Ai, _Real *Ax) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("getCompressedColumn");
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
    void append(const TMatrix &B, AppendPos pos = APPEND_BELOW, bool pad = false,
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
        dumpBinaryToStream(os);
    }

    void dumpBinaryToStream(std::ostream &os) const {
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
        readBinaryFromStream(is);
    }

    void readBinaryFromStream(std::istream &is) {
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
        Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>>(result.data(), result.size()).setZero();
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
template<typename _RowIndex, typename _Index>
_Index binary_search(_RowIndex i, const _RowIndex *Ai, _Index lb, _Index ub) {
#if 1
    return std::distance(Ai, sb_lower_bound(Ai + lb, Ai + ub, i));
#else
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
template<typename _Index, typename _Real, typename _Real2>
size_t csc_add_nz(size_t /* nz */, const _Index *Ai, const _Index *Ap, _Real *Ax, _Index i, _Index j, const _Real2 &v) {
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

// Matrix in Compressed Sparse Column format
template<typename _Index, typename _Real, class IdxVector = std::vector<_Index>>
struct CSCMatrix {
    using index_type = _Index;
    using value_type = _Real;
    using container_type = typename spmat_helper::value_traits<value_type>::container_type;

    using DataType = Eigen::Matrix<_Real, Eigen::Dynamic, 1>;

    template<int Rows = Eigen::Dynamic>
    using SizedDataMap  = Eigen::Map<      Eigen::Matrix<_Real, Rows, 1>>;
    template<int Rows = Eigen::Dynamic>
    using SizedDataCMap = Eigen::Map<const Eigen::Matrix<_Real, Rows, 1>>;

    using DataMap  = SizedDataMap<Eigen::Dynamic>;
    using DataCMap = SizedDataCMap<Eigen::Dynamic>;

    IdxVector Ap; // Column pointer array

    // using RowIndex = int32_t; // Can differ from the index type of `Ap`/nonzero count (since we expect the number of rows to be much less than the number of nonzeros).
    using RowIndex = _Index; // Can differ from the index type of `Ap`/nonzero count (since we expect the number of rows to be much less than the number of nonzeros).
    VecX_T<RowIndex> Ai; // Row index array (must be sorted!)

    container_type Ax;   // Value array (aligned if necessary)
    _Index m, n, nz;     // Number of rows, columns, and nonzeros

    // Rudimentary support for tagging symmetric/nonsymmetric matrices (used by CSCMatrix::apply). This
    // effects, e.g., the interpretation of matrix multiplication.
    enum class SymmetryMode : uint32_t { NONE = 0, UPPER_TRIANGLE = 1, LOWER_TRIANGLE = 2 };
    SymmetryMode symmetry_mode = SymmetryMode::NONE;
    static constexpr _Index INDEX_NONE = std::numeric_limits<_Index>::max();

    size_t nnz() const { return nz; }
    index_type col_nnz(size_t j) const { return Ap[j + 1] - Ap[j]; }
    void reserve(size_t nnz_request) { if (_Index(nnz_request) > nz) throw std::runtime_error("CSCMatrix cannot be resized by `reserve` (" + std::to_string(nnz_request) + " vs " + std::to_string(nz) + ")"); }

    CSCMatrix(_Index mm = 0, _Index nn = 0)
        : m(mm), n(nn), nz(0) { }

    // Construct with a given sparsity pattern and uninitialized values.
    // (Note: if _Real is a plain arithmetic type, `container_type` is really
    // just a std::vector<_Real>, so the values will actually be initialized to
    // zero).
    CSCMatrix(_Index mm, _Index nn, const IdxVector &Ap_in, const IdxVector &Ai_in)
        : Ap(Ap_in), Ai(Ai_in), Ax(Ai.size()), m(mm), n(nn), nz(Ai_in.size()) { }

    CSCMatrix(const CSCMatrix  &b, bool sparsityOnly=false) : Ap(b.Ap), Ai(b.Ai), m(b.m), n(b.n), nz(b.nz), symmetry_mode(b.symmetry_mode) { if (!sparsityOnly) Ax = b.Ax; }
    CSCMatrix(      CSCMatrix &&b) noexcept : Ap(std::move(b.Ap)), Ai(std::move(b.Ai)), Ax(std::move(b.Ax)), m(b.m), n(b.n), nz(b.nz), symmetry_mode(b.symmetry_mode) { }

    CSCMatrix(const std::string &path) { readBinary(path); }

    template<typename T> CSCMatrix(TripletMatrix<T>  &mat) { setFromTMatrix(mat); }
    template<typename T> CSCMatrix(TripletMatrix<T> &&mat) { setFromTMatrix(std::move(mat)); }

    DataMap  data()       { return DataMap (Ax.data(), Ax.size()); }
    DataCMap data() const { return DataCMap(Ax.data(), Ax.size()); }

    // Set each nonzero entry to a particular value, preserving the sparsity pattern.
    void fill(_Real val) {
        Ax.assign(nz, val);
    }

    bool isSparsityOnly() const { return Ax.size() == 0; }

    // Overwrite the numerical values to zero, preserving the sparsity pattern.
    // If this is a sparsity-only matrix, upgrade it to an ordinary one.
    template<bool multithreaded = true> void setZero() {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CSCMatrix::setZero");
        if (Ax.size() == 0) {
            Ax.resize(nz); // upgrade sparsity-only matrix
            return;
        }

        if (multithreaded) {
            parallel_for_range(Ax.size(), [&](size_t i) {
                spmat_helper::setZero(Ax[i]);
            });
        }
        else {
            const size_t nnz = Ax.size();
            for (size_t i = 0; i < nnz; ++i)
                spmat_helper::setZero(Ax[i]);
        }
    }
    void clear() { Ap.clear(); Ai.resize(0); Ax.clear(); nz = 0; }

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

    struct InOrderBuilder {
        // Construct a CSCMatrix with a known number of nonzeros in each column by
        // inserting each in sorted order exactly once.
        // This is convenient for cases like transposing a CSCMatrix.
        template<typename SizeCalculator>
        InOrderBuilder(CSCMatrix &mat, SizeCalculator &&columnSizeCalculator, bool sparsityOnly = false)
            : m_result(mat)
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer("InOrderBuilder constructor");
            const _Index out_n = mat.n;
            m_result.Ap.assign(out_n + 1, 0);
            // We use a trick to avoid using any additional storage to hold
            // the partial column end pointers as we fill in the new matrix:
            // We initially construct the column *start* pointer for column i in Ap[i + 1]
            // Then we increment these as we fill in the columns until Ap[i + 1] becomes
            // the column *end* pointer for column i as desired.

            // Compute the size of output column j in Ap[j + 1];
            _Index *colSizes = m_result.Ap.data() + 1;
            {
                BENCHMARK_SCOPED_TIMER_SECTION t2("columnSizeCalculator");
                columnSizeCalculator(colSizes);
            }

            // Next calculate the start pointer for column i in Ap[i + 1]; this is
            // the cumulative size of the previous output columns, which we
            // maintain in `accum_nz`
            _Index accum_nz = 0;
            {
                _Index *colBegin = m_result.Ap.data() + 1;
                for (_Index j = 0; j < out_n; ++j) {
                    _Index colsize_j = colSizes[j];
                    colBegin[j] = accum_nz;
                    accum_nz += colsize_j;
                }
            }

            BENCHMARK_SCOPED_TIMER_SECTION t2("allocate Ai, Ax");

            m_result.Ai.resize(accum_nz);
            if (!sparsityOnly) m_result.Ax.resize(accum_nz);
            m_result.nz = accum_nz;

            // Current column end pointers for each incomplete column bucket.
            m_colEnd = m_result.Ap.data() + 1;
        }

        void insert(_Index i, _Index j, _Real v) {
            _Index entry = m_colEnd[j];
            m_result.Ai[entry] = i;
            m_result.Ax[entry] = v;
            ++m_colEnd[j];
        }

        // Version that only sets sparsity pattern.
        void insert(_Index i, _Index j) {
            _Index entry = m_colEnd[j];
            m_result.Ai[entry] = i;
            ++m_colEnd[j];
        }
    private:
        _Index *m_colEnd;
        CSCMatrix &m_result;
    };

    CSCMatrix transpose(bool force = false) const {
        return transposeImpl(force, [&](size_t ii) { return Ax[ii]; });
    }

    // Note: assumes entries within each column are sorted and unique.
    // Produces a sorted, unique output matrix.
    template<typename T = _Real, class ValueGetter>
    CSCMatrix<_Index, T> transposeImpl(bool force, const ValueGetter &value) const {
        using Result = CSCMatrix<_Index, T>;
        Result result(n, m);
        if ((symmetry_mode != SymmetryMode::NONE) && !force) {
            // No-op.
            result.Ap = Ap;
            result.Ai = Ai;
            result.Ax.reserve(Ax.size());
            for (SuiteSparse_long ii = 0; ii < nz; ++ii)
                result.Ax[ii] = value(ii);
            return result;
        }

        typename Result::InOrderBuilder builder(result, [this](_Index *colSizes) { for (_Index i : Ai) ++colSizes[i]; });
        assert(result.nz == nz);

        // Add entries into the transposed matrix in sorted order
        for (_Index c = 0; c < n; ++c) {
            for (_Index inLoc = Ap[c]; inLoc < Ap[c + 1]; ++inLoc)
                builder.insert(c, Ai[inLoc], spmat_helper::transpose_block(value(inLoc)));
        }

        return result;
    }

    template<typename T = _Real, class ValueGetter>
    CSCMatrix<_Index, T> toSymmetryModeImpl(SymmetryMode newMode, const ValueGetter &value) const {
        using Result = CSCMatrix<_Index, T>;
        if (newMode == symmetry_mode) {
            Result result(m, n);
            result.Ap = Ap;
            result.Ai = Ai;
            result.Ax.reserve(Ax.size());
            for (SuiteSparse_long ii = 0; ii < nz; ++ii)
                result.Ax.push_back(value(ii));
            return result;
        }
        if (m != n) throw std::runtime_error("Matrix is not symmetric");

        // Replicating an upper/lower triangle matrix to a full matrix.
        if (newMode == SymmetryMode::NONE) {
            Result result(m, n);
            typename Result::InOrderBuilder builder(result, [this](_Index *colSizes) {
                for (_Index j = 0; j < n; ++j) {
                    for (_Index inLoc = Ap[j]; inLoc < Ap[j + 1]; ++inLoc) {
                        const _Index i = Ai[inLoc];
                        if (((symmetry_mode == SymmetryMode::UPPER_TRIANGLE) && (i > j)) ||
                            ((symmetry_mode == SymmetryMode::LOWER_TRIANGLE) && (i < j))) throw std::runtime_error("Symmetry mode violation");
                        ++colSizes[j];
                        if (i != j) ++colSizes[i];
                    }
                }
            });

            // Accumulate entries in the output matrix in sorted order.
            // Note: from our requirement that this->Ai be sorted, we can
            // assume input triplets are visited in lexicographically
            // sorted (j, i) order.
            // Therefore, entries will be added to the output matrix columns in sorted
            // order as well. However, we must make sure in the LOWER_TRIANGLE
            // case that all transposed (upper tri) entries are added first.
            if (symmetry_mode == SymmetryMode::UPPER_TRIANGLE) {
                // Insert upper triangle entries of each column followed by
                // strict lower triangle entries (in the reflected part)
                for (_Index j = 0; j < n; ++j) {
                    for (_Index inLoc = Ap[j]; inLoc < Ap[j + 1]; ++inLoc) {
                        const _Index i = Ai[inLoc];
                        builder.insert(i, j, value(inLoc));
                        if (i != j) builder.insert(j, i, value(inLoc));
                    }
                }
            }
            else {
                if (symmetry_mode != SymmetryMode::LOWER_TRIANGLE) throw std::logic_error("Unexpected conversion request");
                // Insert strict upper triangle (reflected part) first...
                for (_Index j = 0; j < n; ++j) {
                    for (_Index inLoc = Ap[j]; inLoc < Ap[j + 1]; ++inLoc) {
                        const _Index i = Ai[inLoc];
                        if (i != j) builder.insert(j, i, value(inLoc));
                    }
                }
                // Followed by lower triangle
                for (_Index j = 0; j < n; ++j) {
                    for (_Index inLoc = Ap[j]; inLoc < Ap[j + 1]; ++inLoc) {
                        const _Index i = Ai[inLoc];
                        builder.insert(i, j, value(inLoc));
                    }
                }
            }
            return result;
        }
        // Discard the strict lower or upper triangle of a symmetric matrix
        // with both parts explicitly stored.
        if (symmetry_mode == SymmetryMode::NONE) {
            Result result(m, n);
            result.symmetry_mode = typename Result::SymmetryMode(newMode);
            typename Result::InOrderBuilder builder(result, [this, newMode](_Index *colSizes) {
                for (_Index j = 0; j < n; ++j) {
                    for (_Index inLoc = Ap[j]; inLoc < Ap[j + 1]; ++inLoc) {
                        const _Index i = Ai[inLoc];
                        colSizes[j] += (i == j) || ((i < j) == (newMode == SymmetryMode::UPPER_TRIANGLE));
                    }
                }
            });

            for (_Index j = 0; j < n; ++j) {
                for (_Index inLoc = Ap[j]; inLoc < Ap[j + 1]; ++inLoc) {
                    const _Index i = Ai[inLoc];
                    if ((i == j) || ((i < j) == (newMode == SymmetryMode::UPPER_TRIANGLE)))
                        builder.insert(i, j, value(inLoc));
                }
            }

            return result;
        }

        // Conversion between UPPER_TRIANGLE and LOWER_TRIANGLE
        {
            // Transpose the triangular part stored into the opposite triangle.
            Result result = transposeImpl<T>(/* force = */ true, value);
            result.symmetry_mode = typename Result::SymmetryMode(newMode);
            return result;
        }
    }

    // Construct a new CSCMatrix with a different symmetry mode. Currently
    // only conversions between upper- and lower-triangle symmetric matrix
    // representations are supported.
    CSCMatrix toSymmetryMode(SymmetryMode newMode) const {
        return toSymmetryModeImpl(newMode, [this](_Index ii) { return Ax[ii]; });
    }

    // Convert the sparsity pattern of this matrix into a different symmetry mode
    // (discarding values).
    CSCMatrix toSymmetryModeSparsityOnly(SymmetryMode newMode, _Real val = 0.0) const {
        return toSymmetryModeImpl(newMode, [val](_Index /* ii */) { return val; });
    }

    // Treating this matrix's sparsity pattern as a block sparsity pattern with
    // block size `UniformBlockSize`, expand it into the corresponding scalar sparsity pattern.
    template<size_t UniformBlockSize, bool AssumeDiagonalExists = false>
    CSCMatrix expandSparsityPattern() const {
        if (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only SymmetryMode::UPPER_TRIANGLE is supported");

        static constexpr size_t N = UniformBlockSize;
        size_t n_scalar = n * UniformBlockSize;
        CSCMatrix result(n_scalar, n_scalar);
        result.symmetry_mode = SymmetryMode::UPPER_TRIANGLE;

        const CSCMatrix &blockHsp = *this;
        typename CSCMatrix::InOrderBuilder builder(result, [&blockHsp](_Index *colSizes) {
                // Count the number of nonzeros in each column of the scalar Hessian sparsity pattern.
                for (_Index block_j = 0; block_j < blockHsp.n; ++block_j) {
                    size_t gvar_j = block_j * N;
                    size_t numBlocks = blockHsp.Ap[block_j + 1] - blockHsp.Ap[block_j];
                    bool hasDiagonal = AssumeDiagonalExists || (blockHsp.Ai[blockHsp.Ap[block_j + 1] - 1] == block_j);
                    if (hasDiagonal) {
                        size_t colSize = (numBlocks - 1) * N + 1;
                        for (size_t c_j = 0; c_j < N; ++c_j)
                            colSizes[gvar_j + c_j] = colSize++;
                    }
                    else {
                        for (size_t c_j = 0; c_j < N; ++c_j)
                            colSizes[gvar_j + c_j] = N * numBlocks;
                    }
                }
            }, /* sparsityOnly = */ true);

        // BENCHMARK_SCOPED_TIMER_SECTION timer2("builderFiller");
        // Filling out the index arrays (can be done in parallel)
        for (_Index block_j = 0; block_j < blockHsp.n; ++block_j) {
            size_t gvar_j = block_j * N;
            for (_Index ii = blockHsp.Ap[block_j]; ii < blockHsp.Ap[block_j + 1]; ++ii) {
                _Index block_i = blockHsp.Ai[ii];
                if (block_i < block_j) {
                    size_t gvar_i = block_i * N;
                    for (size_t c_j = 0; c_j < N; ++c_j)
                        for (size_t c_i = 0; c_i < N; ++c_i)
                            builder.insert(gvar_i + c_i, gvar_j + c_j);
                }
                else {
                    for (size_t c_j = 0; c_j < N; ++c_j)
                        for (size_t c_i = 0; c_i <= c_j; ++c_i)
                            builder.insert(gvar_j + c_i, gvar_j + c_j);
                }
            }
        }
        return result;
    }

    void reflectUpperTriangle() {
        if (symmetry_mode == SymmetryMode::NONE) throw std::runtime_error("Matrix is not in symmetric lower/upper triangle respresentation");
        *this = toSymmetryMode(SymmetryMode::NONE);
    }

    // Copy the strict lower triangle into the strict upper triangle.
    // This method assumes that the sparsity pattern is already symmetric and can therefore
    // avoid the allocation of new Ai/Ap/Ax entries.
    // Furthermore, by processing lower triangle cols from left to right we know that
    // entries will be added each column of the upper triangle from top to bottom,
    // avoiding the need to search for entries entirely.
    void reflectLowerTriangleInPlace() {
        auto insertion_point = Ap; // index in each column's bucket where the next entry should be added (start at beginning)
        for (index_type j = 0; j < n; ++j) {
            index_type ii = findEntry(j, j) + 1;
            index_type end = Ap[j + 1];
            for (; ii < end; ++ii) {
                size_t i = Ai[ii];
                // Add entry (j, i)
                index_type ip = insertion_point[i];
                if (Ai[ip] != j) throw std::runtime_error("Failed to find lower tri entry in the sparsity pattern!");
                Ax[ip] = spmat_helper::transpose_block(Ax[ii]);
                insertion_point[i] = ip + 1;
            }
        }
    }

    template<bool _detectMissing = false>
    void reflectUpperTriangleInPlaceParallel() {
        parallel_for_range(n, [&](index_type j) {
            index_type i;
            for (index_type ii = Ap[j]; (i = Ai[ii]) < j; ++ii)
                Ax[findEntry<_detectMissing>(j, i)] += spmat_helper::transpose_block(Ax[ii]);
        });
    }

    // Set this matrix to have the same sparsity pattern as b, but with zeros
    template<bool multithreaded = true>
    void zeros_like(const CSCMatrix &b) {
        m = b.m; n = b.n; nz = b.nz;
        Ap = b.Ap; Ai = b.Ai;
        bool previouslyEmpty = Ax.empty();
        Ax.resize(Ai.size()); // already zero-initializes new elmeents for arithmetic types!
        if (!std::is_arithmetic<_Real>::value || !previouslyEmpty)
            setZero<multithreaded>();
    }

    template<bool _detectMissing = false>
    _Index findDiagEntry(_Index i) const {
        if (symmetry_mode == SymmetryMode::UPPER_TRIANGLE) {
            _Index idx = Ap[i + 1] - 1; // Diagonal element is the last entry in the column "i"
            if (_detectMissing && ((idx < Ap[i]) || (Ai[idx] != i))) return INDEX_NONE;
            assert((idx >= Ap[i]) && (Ai[idx] == i));
            return idx;
        }
        else if (symmetry_mode == SymmetryMode::LOWER_TRIANGLE) {
            _Index idx = Ap[i]; // Diagonal element is the first entry in the column "i"
            if (_detectMissing && ((idx >= Ap[i + 1]) || (Ai[idx] != i))) return INDEX_NONE;
            assert((idx < Ap[i + 1]) && (Ai[idx] == i));
            return idx;
        }
        return findEntry<_detectMissing>(i, i);
    }

    // Add the NxN block `B` to this matrix, placing its upper-left corner at (i, i).
    // (Assumes the block already exists in the sparsity pattern)
    // Only implemented for matrices with upper- or lower-triangle symmetry mode;
    // we cannot achieve a performance advantage over the block version of
    // addNZ for general sparse matrices.
    template<typename Derived>
    void addDiagBlock(_Index i, const Eigen::MatrixBase<Derived> &B) {
        constexpr _Index N = Derived::ColsAtCompileTime;
        static_assert((N == Derived::RowsAtCompileTime) && (N != Eigen::Dynamic), "Intended for fixed-size square blocks only");

        if (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only implemented for UPPER_TRIANGLE matrices");
        // TODO: LOWER_TRIANGLE version

        for (_Index l = 0; l < N; ++l) {
            _Index idx = findDiagEntry(i + l); // bottom of column strip to add
            for (SuiteSparse_long k = l; k >= 0; --k) { // upper triangle only
                assert((Ai[idx] == i + k) && "Entry absent from sparsity pattern");
                Ax[idx--] += B(k, l);
            }
        }
    }

    // Add a vertical strip of contiguous nonzero values *ending* at (i, i)
    template<class Derived>
    void addDiagNZStrip(_Index i, const Eigen::DenseBase<Derived> &values) {
        static_assert(Derived::ColsAtCompileTime == 1, "Only column vectors can be added with addNZStrip");
        if (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only implemented for UPPER_TRIANGLE matrices");

        _Index idx = findDiagEntry(i);
        for (SuiteSparse_long k = values.size() - 1; k >= 0; --k, --idx) { // upper triangle only
            Ax[idx] += values[k];
        }
    }

    template<typename _Real2>
    void addDiagEntry(_Index i, const _Real2 &v) { Ax[findDiagEntry(i)] += v; }

    void addScaledIdentity(_Real v) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("addScaledIdentity");
        if (m > 8192) {
            tbb::parallel_for(tbb::blocked_range<_Index>(0, m), [this, v](const tbb::blocked_range<_Index> &r) {
                for (_Index i = r.begin(); i < r.end(); ++i)
                    addDiagEntry(i, v);
            });
        }
        else {
            for (_Index i = 0; i < m; ++i)
                addDiagEntry(i, v);
        }
    }

    template<class _InVector>
    void addDiag(const _InVector &d) {
        for (_Index i = 0; i < m; ++i) addDiagEntry(i, d[i]);
    }

    template<bool _detectMissing = false>
    _Index findEntry(RowIndex i, _Index j) const {
        // Find the entry in the sparsity pattern.
        // Row indices are sorted, so we can use a binary search.
        const _Index colend = Ap[j + 1];
        _Index idx = binary_search(i, Ai.data(), Ap[j], colend);
        if (_detectMissing && ((idx == colend) || (Ai[idx] != i))) return INDEX_NONE;
        assert((idx != colend) && (Ai[idx] == i) && "Entry absent from sparsity pattern");
        return idx;
    }

    // Find with guess (hint).
    template<bool _detectMissing = false>
    _Index findEntry(_Index i, _Index j, _Index hint) const {
        if ((hint < Ap[j + 1]) && (Ai[hint] == i) && (hint >= Ap[j])) {
            return hint;
        }
        return findEntry(i, j);
    }

    // Accumulate a value to (i, j)
    // Note: (i, j) must exist in the sparsity pattern!
    // Complexity: O(log(n_j)) where "n_j" is the number of nonzeros in column j
    template<typename _Real2>
    size_t addNZ(_Index i, _Index j, const _Real2 &v) {
        assert((i < m) && (j < n) && "Index out of bounds");
        return csc_add_nz(nz, Ai.data(), Ap.data(), Ax.data(), i, j, v);
    }

    // Insert (i, j, v), with a guess that it should go at location "hint"
    template<typename _Real2>
    size_t addNZ(const _Index i, const _Index j, const _Real2 &v, _Index hint) {
        if ((hint < Ap[j + 1]) && (Ai[hint] == i) && (hint >= Ap[j])) {
            Ax[hint] += v;
            return hint + 1;
        }
        return addNZ(i, j, v);
    }

    // Add a vertical strip of contiguous nonzero values starting at (i, j),
    // return the index of the next nonzero entry after the written strip.
    // (so that the adjacent strip below can be written by directly calling addNZ(idx, values))
    template<class Derived>
    _Index addNZStrip(_Index i, _Index j, const Eigen::DenseBase<Derived> &values) {
        static_assert(Derived::ColsAtCompileTime == 1, "Only column vectors can be added with addNZStrip");
        return addNZStrip(findEntry(i, j), values);
    }

    template<class Derived>
    _Index addNZStrip(_Index i, _Index j, const Eigen::DenseBase<Derived> &values, _Index hint) {
        if ((hint < nz) && (Ai[hint] == i) && (hint < Ap[j + 1]) && (hint >= Ap[j]))
            return addNZStrip(hint, values);
        return addNZStrip(i, j, values);
    }

    // Add a sequence of values to the compressed nonzero entries starting at "idx"
    template<class Derived>
    _Index addNZStrip(_Index idx, const Eigen::DenseBase<Derived> &values) {
        static_assert(Derived::ColsAtCompileTime == 1, "Only column vectors can be added with addNZStrip");
        DataMap(Ax.data() + idx, values.rows()) += values;
        return idx + values.size();
    }

    template<typename _Real2>
    _Index addNZ(_Index idx, const _Real2 &val) {
        Ax[idx] += val;
        return idx + 1;
    }

    // Add a block of contiguous nonzero values starting at (i, j)
    // Note: if symmetry_mode UPPER_TRIANGLE and and we are adding a diagonal
    // block, only the upper triangle is added.
    template<class Derived>
    void addNZBlock(_Index i, _Index j, const Eigen::DenseBase<Derived> &values) {
        if ((symmetry_mode == SymmetryMode::UPPER_TRIANGLE) && (i == j)) {
            for (int c = 0; c < values.cols(); ++c)
                addNZStrip(findEntry(i, j + c), values.col(c).head(c + 1));
        }
        else {
            for (int c = 0; c < values.cols(); ++c) {
                addNZStrip(findEntry(i, j + c), values.col(c));
            }
        }
    }

    // Add a nonzero entry using a guess `loc` of the location where it appears
    // in column `j`'s sparsity pattern that is *guaranteed to be a lower bound*.
    // A binary search is dispensed in favor of a linear search initiated at
    // `loc`. The location of the *subsequent* entry in the column is returned.
    // This is a more persistent version of `addNZ(i, j, v, hint)` that
    // does not fall back to a binary search on an incorrect guess.
    // If `_knownGood` is `false`, we allow for the possibility that `loc` is invalid
    // and needs to be searched afresh.
    template<bool _knownGood = true, typename _Real2>
    _Index addNZAtLoc(_Index i, _Index j, const _Real2 &val, int loc) {
        if (!_knownGood) {
            if ((loc < Ap[j]) || (loc >= Ap[j + 1]) || (Ai[loc] > i))
                return csc_add_nz(nz, Ai.data(), Ap.data(), Ax.data(), i, j, val);
        }
        while (Ai[loc] < i) ++loc;
        Ax[loc] += val;
        return loc + 1;
    }

    CSCMatrix &operator=(const CSCMatrix  &b) { Ap = b.Ap           ; Ai = b.Ai           ; Ax = b.Ax           ; m = b.m; n = b.n; nz = b.nz; symmetry_mode = b.symmetry_mode; return *this; }
    CSCMatrix &operator=(      CSCMatrix &&b) { Ap = std::move(b.Ap); Ai = std::move(b.Ai); Ax = std::move(b.Ax); m = b.m; n = b.n; nz = b.nz; symmetry_mode = b.symmetry_mode; return *this; }
    template<typename _Real2>
    CSCMatrix &operator=(const CSCMatrix<_Index, _Real2> &b) {
        Ap = b.Ap; Ai = b.Ai;
        Ax.clear();
        Ax.reserve(b.Ax.size());
        for (const auto &v : b.Ax) Ax.emplace_back(v);
        m = b.m; n = b.n; nz = b.nz; symmetry_mode = SymmetryMode(b.symmetry_mode);
        return *this;
    }

    // Copy just the shape/sparsity pattern of `b` (but not the values).
    template<typename _Real2>
    void copySparsityPattern(const CSCMatrix<_Index, _Real2> &b) {
        m = b.m; n = b.n;
        Ap = b.Ap; Ai = b.Ai;
        nz = Ai.size();
        symmetry_mode = SymmetryMode(b.symmetry_mode);
        Ax.resize(nz);
    }

    _Real max()    const { return DataCMap(Ax.data(), Ax.size()).maxCoeff(); }
    _Real absMax() const { return DataCMap(Ax.data(), Ax.size()).cwiseAbs().maxCoeff(); }
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
        if (nz != b.nz) throw std::runtime_error("Mismatched sparsity patterns");
        for (_Index i = 0; i < nz; ++i) { if ((Ax[i] != 0) || (b.Ax[i] != 0)) Ax[i] /= b.Ax[i]; }
    }

    // (*this) += alpha * b, assuming b's sparsity pattern is identical to ours.
    template<typename IdxVector2>
    void addWithIdenticalSparsity(const CSCMatrix<_Index, _Real, IdxVector2> &b, double alpha = 1.0) {
        if (b.Ax.size() != Ax.size()) throw std::runtime_error("nnz mismatch");
        if constexpr (std::is_arithmetic_v<_Real>) {
            DataMap   AEigen(  Ax.data(),   Ax.size());
            DataCMap bAEigen(b.Ax.data(), b.Ax.size());
            addScaledInPlace(AEigen, bAEigen, alpha);
        }
        else {
            if (alpha == 1.0) { parallel_for_range(Ax.size(), [&](size_t i) { Ax[i] +=         b.Ax[i]; }); }
            else              { parallel_for_range(Ax.size(), [&](size_t i) { Ax[i] += alpha * b.Ax[i]; }); }
        }
    }

    void scale(_Real alpha) { DataMap(Ax.data(), Ax.size()) *= alpha; }
    CSCMatrix &operator*=(_Real alpha) { scale(alpha); return *this; }

    // Perform the operation:
    //  (*this)[offset:, offset:] += alpha * b[blockStart:blockEnd, blockStart:blockEnd]
    // Assumes RHS sparsity pattern is a subset of LHS.
    // offset: offset to be applied to the row and column indices of b
    void addWithSubSparsity(const CSCMatrix &b, const _Real alpha = 1.0, const _Index offset = 0, const _Index blockStart = 0, const _Index blockEnd = std::numeric_limits<_Index>::max()) {
        auto it  = begin(), bit  = b.begin(),
             ite = end(),   bite = b.end();
        auto inRange = [&](_Index i) { return (i >= blockStart) && (i < blockEnd); };
        auto bi = [&]() { return offset + bit.get_i(); };
        auto bj = [&]() { return offset + bit.get_j(); };
        while ((it != ite) && (bit != bite)) {
            if (!inRange(bit.get_i()) || !inRange(bit.get_j())) { ++bit; continue; }
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

    template<bool parallel = true>
    void addWithSubSparsityFast(const CSCMatrix &b, const _Real alpha = 1.0, const _Index offset = 0) {
        auto addColumn = [&](_Index jA) {
            _Index jB = jA - offset;

            _Index iiA = Ap[jA], iiB = b.Ap[jB];
            _Index endA = Ap[jA + 1], endB = b.Ap[jB + 1];
            while ((iiA != endA) && (iiB != endB)) {
                _Index rA = Ai[iiA], rB = b.Ai[iiB];
                if (rA == rB) {
                    Ax[iiA] += alpha * b.Ax[iiB];
                    ++iiA, ++iiB;
                }
                else {
                    assert(rA < rB && "b's sparsity not a subset of ours");
                    ++iiA;
                }
            }
            assert(iiB == endB && "Hit end of column A before end of column B :(");
        };

        if (parallel) {
#if 0
            // This one seems slower :(
            static tbb::affinity_partitioner ap;
            tbb::parallel_for(tbb::blocked_range<_Index>(offset, n),
                [&](const tbb::blocked_range<_Index> &r) {
                    for (_Index i = r.begin(); i < r.end(); ++i)
                        addColumn(i);
                }, ap);
#else
            parallel_for_range(offset, n, addColumn);
#endif
        }
        else {
            for (_Index jA = offset; jA < n; ++jA)
                addColumn(jA);
        }
    }

    // Perform the operation:
    // a[offset:, offset:] + alpha * b[blockStart:blockEnd, blockStart:blockEnd]
    // Sparsity pattern of RHS can be arbitrary.
    template<bool SparsityOnly = false>
    static CSCMatrix addWithDistinctSparsityPattern(const CSCMatrix &a, const CSCMatrix &b, const _Real alpha = 1.0, const _Index offset = 0, const _Index blockStart = 0, const _Index blockEnd = std::numeric_limits<_Index>::max()) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CSCMatrix.addWithDistinctSparsityPattern");

        _Index inputSize = std::min(b.m, blockEnd) - blockStart;
        if (b.m != b.n) throw std::runtime_error("Only square matrices are supported");
        if ((a.m != inputSize + offset) || (a.n != inputSize + offset)) throw std::runtime_error("Size mismatch");
        if (a.symmetry_mode != b.symmetry_mode) throw std::runtime_error("Symmetry mode mismatch");
        if (b.nz == 0) return a;
        if (a.nz == 0) return b;

        auto it  = a.begin(), bit  = b.begin(),
             ite = a.end(),   bite = b.end();
        auto inRange = [&](_Index i) { return (i >= blockStart) && (i < blockEnd); };
        auto bi   = [&]() { return offset + bit.get_i();   };
        auto bj   = [&]() { return offset + bit.get_j();   };
        auto bval = [&]() { return  alpha * bit.get_val(); };
        UNUSED(bval);

        CSCMatrix result(a.m, a.n);
        result.symmetry_mode = a.symmetry_mode;
        auto &newAp = result.Ap;
        std::vector<index_type> newAi;
        newAp.reserve(a.Ap.size());
        newAi.reserve(a.Ai.size());

        auto &newAx = result.Ax;
        if constexpr (!SparsityOnly) newAx.reserve(a.Ax.size());

        // Merge sorted triplets into the new result
        _Index currCol = 0;
        newAp.push_back(0);
        auto insertEntry = [&](_Index row, _Index col, _Real val) {
            UNUSED(val);
            assert(col >= currCol);
            // End all columns up to `col - 1`, begin column `col`
            for (_Index c = currCol + 1; c <= col; ++c)
                newAp.push_back(newAi.size());
            currCol = col;
            newAi.push_back(row);
            if constexpr (!SparsityOnly) newAx.push_back(val);
        };

        while ((it != ite) || (bit != bite)) {
            if ((bit != bite) && (!inRange(bit.get_i()) || !inRange(bit.get_j()))) { ++bit; continue; } // filter entries outside the input block
            bool takeA = ( it !=  ite);
            bool takeB = (bit != bite);

            // If both A and B entries are available, pick the first one in sorted order (or pick both and sum if they are at the same location).
            if (takeA && takeB) {
                std::pair<_Index, _Index> a_colrow{it.get_j(), it.get_i()}, b_colrow{bj(), bi()};
                takeA = a_colrow <= b_colrow; // (col, row) lexicographic ordering
                takeB = b_colrow <= a_colrow;
            }

            if constexpr (!SparsityOnly) {
                if ( takeA && !takeB) { insertEntry(it.get_i(), it.get_j(), it.get_val()         ); ++it;        }
                if ( takeA &&  takeB) { insertEntry(it.get_i(), it.get_j(), it.get_val() + bval()); ++it; ++bit; }
                if (!takeA &&  takeB) { insertEntry(      bi(),       bj(),                bval());       ++bit; }
            }
            else {
                if ( takeA && !takeB) { insertEntry(it.get_i(), it.get_j(), 0); ++it;        }
                if ( takeA &&  takeB) { insertEntry(it.get_i(), it.get_j(), 0); ++it; ++bit; }
                if (!takeA &&  takeB) { insertEntry(      bi(),       bj(), 0);       ++bit; }
            }
        }
        if (currCol >= a.n) throw std::runtime_error("Column index out of bounds");

        // Terminate all remaining columns
        for (_Index c = currCol; c < a.n; ++c)
            newAp.push_back(newAi.size());

        result.Ai.resize(newAi.size());
        std::copy(newAi.begin(), newAi.end(), result.Ai.data());

        assert(newAp.size() == size_t(a.n + 1));
        result.nz = newAi.size();
        return result;
    }

    // Perform the operation:
    //  (*this)[offset:, offset:] += alpha * b[blockStart:blockEnd, blockStart:blockEnd]
    template<bool SparsityOnly = false>
    void addWithDistinctSparsityPattern(const CSCMatrix &b, const _Real alpha = 1.0, const _Index offset = 0, const _Index blockStart = 0, const _Index blockEnd = std::numeric_limits<_Index>::max()) {
        if (b.nz == 0) return;
        *this = addWithDistinctSparsityPattern<SparsityOnly>(*this, b, alpha, offset, blockStart, blockEnd);
    }

    // Produces a sparsity-only matrix (with `Ax` empty)!
    void mergeSparsityPattern(const CSCMatrix &b, const _Index offset = 0, const _Index blockStart = 0, const _Index blockEnd = std::numeric_limits<_Index>::max()) {
        if (b.nz == 0) return;
        *this = addWithDistinctSparsityPattern</* SparsityOnly = */ true>(*this, b, 1.0, offset, blockStart, blockEnd);
    }

    bool sparsityPatternsMatch(const CSCMatrix &b) const {
        return (m == b.m) && (n == b.n) && (nnz() == b.nnz())
               && std::equal(Ap.begin(), Ap.end(), b.Ap.begin(), b.Ap.end())
               && std::equal(Ai.begin(), Ai.end(), b.Ai.begin(), b.Ai.end());
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
            DataMap(Ax.data(), Ax.size()) = diag;
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

    // More efficient binary output format:
    //      number of rows     (_Index)
    //      number of column   (_Index)
    //      number of nonzeros (_Index)
    //      symmetry mode      (uint32_t)
    //      Ap                 (#cols + 1 items of type _Index)
    //      Ai                 (nz        items of type _Index)
    //      Ax                 (nz        items of type _Real)
    // Note, output files are not portable across architectures (e.g., byte order)
    void dumpBinary(const std::string &path) const {
        std::ofstream os(path);
        if (!os.is_open()) throw std::runtime_error("Failed to open output file " + path);
        dumpBinaryToStream(os);
    }

    void dumpBinaryToStream(std::ostream &os) const {
        if ((Ap.size() != size_t(n + 1)) || (Ai.size() != size_t(nz)) || (Ax.size() != size_t(nz)))
                throw std::runtime_error("Inconsistent matrix size metadata");

        os.write((const char *) & m, sizeof(_Index));
        os.write((const char *) & n, sizeof(_Index));
        os.write((const char *) &nz, sizeof(_Index));
        os.write((const char *) &symmetry_mode, sizeof(uint32_t));
        os.write((const char *) Ap.data(), Ap.size() * sizeof(_Index));
        os.write((const char *) Ai.data(), Ai.size() * sizeof(_Index));
        os.write((const char *) Ax.data(), Ax.size() * sizeof( _Real));
    }

    void readBinary(const std::string &path) {
        std::ifstream is(path);
        if (!is.is_open()) throw std::runtime_error("Failed to open input file " + path);
        readBinaryFromStream(is);
    }

    void readBinaryFromStream(std::istream &is) {
        is.read((char *) & m, sizeof(_Index));
        is.read((char *) & n, sizeof(_Index));
        is.read((char *) &nz, sizeof(_Index));
        is.read((char *) &symmetry_mode, sizeof(uint32_t));

        if ((symmetry_mode != SymmetryMode::NONE) &&
            (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) &&
            (symmetry_mode != SymmetryMode::LOWER_TRIANGLE)) {
            throw std::runtime_error("Invalid symmetry_mode: " + std::to_string(uint32_t(symmetry_mode)));
        }

        Ap.resize(n + 1);
        Ai.resize(nz);
        Ax.resize(nz);

        is.read((char *) Ap.data(), Ap.size() * sizeof(_Index));
        is.read((char *) Ai.data(), Ai.size() * sizeof(_Index));
        is.read((char *) Ax.data(), Ax.size() * sizeof( _Real));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Iteration over the nonzero entries stored in this matrix, as Triplet<>s.
    ////////////////////////////////////////////////////////////////////////////
    struct TripletIterator {
        TripletIterator(const CSCMatrix &mat_, _Index idx_) : mat(mat_) {
            idx = idx_;
            if ((idx >= 0) && idx < mat.nz) {
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
    protected:
        TripletIterator(const CSCMatrix &mat_) : mat(mat_) { }
        _Index idx, j; // nonzero entry and column index
                       // (column index is cached/updated for efficiency to avoid a search on each dereference)
        const CSCMatrix &mat;
    };

    struct ColumnTripletIterator : public TripletIterator {
        ColumnTripletIterator(const CSCMatrix &mat_, _Index j_) : TripletIterator(mat_) {
            j = j_;
            if ((j < 0) || (j > mat.n)) throw std::runtime_error("Column index out of bounds");
            idx = mat.Ap[j];
        }
    protected:
        using TripletIterator::j;
        using TripletIterator::mat;
        using TripletIterator::idx;
    };

    struct ColumnRange {
        ColumnRange(const CSCMatrix &mat_, _Index j_) : j(j_), mat(mat_) { }
        ColumnTripletIterator begin() { return ColumnTripletIterator(mat,     j); }
        ColumnTripletIterator   end() { return ColumnTripletIterator(mat, j + 1); }
    private:
        _Index j;
        const CSCMatrix &mat;
    };

    TripletIterator begin() const { return TripletIterator(*this,  0); }
    TripletIterator   end() const { return TripletIterator(*this, nz); }

    ColumnRange col(_Index j) const { return ColumnRange(*this, j); }

    _Index columnSize(_Index j) const { return Ap[j + 1] - Ap[j]; }

    // Matrix-vector multiply
    template<typename _Vector>
    _Vector apply(const _Vector &x, const bool transpose = false) const {
        const size_t local_m = transpose ? n : m;
        const size_t local_n = transpose ? m : n;
        if (size_t(x.size()) != local_n) throw std::runtime_error("Sparse matvec size mismatch.");
        _Vector result(local_m);
        applyRaw(x.data(), result.data(), transpose);
        return result;
    }

    template<typename _Real2> // Templated to support, e.g., application of non-autodiff matrix to autodiff vector.
    void applyRaw(const _Real2 *x, _Real2 *result, const bool transpose = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CSCMatrix.applyRaw");
        const bool swapIndices = transpose && (symmetry_mode == SymmetryMode::NONE);

        size_t len = transpose ? n : m;
        spmat_helper::setZero(result, len);

        if (swapIndices) {
            // Use applyTransposeParallel?
            for (_Index j = 0; j < n; ++j) {
                auto r_j = result[j];
                for (_Index ii = Ap[j]; ii < Ap[j + 1]; ++ii)
                    r_j += spmat_helper::transpose_block(Ax[ii]) * x[Ai[ii]];
                result[j] = r_j;
            }
        }
        else {
            if (transpose) {
                assert(symmetry_mode != SymmetryMode::NONE); // asymmetric transpose handled above...
                for (_Index j = 0; j < n; ++j) {
                    auto x_j = x[j];
                    auto r_j = result[j];
                    for (_Index ii = Ap[j]; ii < Ap[j + 1]; ++ii) {
                        _Index i = Ai[ii];
                        r_j += spmat_helper::transpose_block(Ax[ii]) * x[i];
                        if (i != j) result[i] += Ax[ii] * x_j;
                    }
                    result[j] = r_j;
                }
            }
            else {
                for (_Index j = 0; j < n; ++j) {
                    auto x_j = x[j];
                    for (_Index ii = Ap[j]; ii < Ap[j + 1]; ++ii) {
                        _Index i = Ai[ii];
                        result[i] += Ax[ii] * x_j;
                        if ((symmetry_mode != SymmetryMode::NONE) && (i != j))
                            result[j] += spmat_helper::transpose_block(Ax[ii]) * x[i];
                    }
                }
            }
        }
    }

    // Parallelized matvec `A x`, where `A` is either an plain sparse matrix
    // or a sparse matrix in block form  (i.e., an m x n matrix whose entires
    // are p x q dense blocks).
    // When the matrix is in block form, `x` can be either also be in block
    // form (an m-vector whose elements are p-vectors) or in "flattened" form
    // (a plain vector of length m * p).
    //
    // Currently we only support applying the *transpose* of this matrix to `x`
    // since in the CSC representation, only that can be done lock-free and
    // without accumulating and subsequently combining a partial "result"
    // vectors for each thread.
    // Furthermore, the full matrix (upper and lower tri) must be stored even
    // in the symmetric case.
    //
    // The following operations can be performed, depending on the template
    // parameters:
    //      result =  A x,   result += A x      (Negate = False, ZeroInit = True, False)
    //      result = -A x,   result -= A x      (Negate =  True, ZeroInit = True, False)
    // If `ZeroInit` is false, the operation is `result += A x`
    template<bool ZeroInit = true, bool Negate = false, class _InVector, class _Result>
    void applyTransposeParallel(const _InVector &x, _Result &&result) const {
        using   Result = std::decay_t<_Result>;
        using InVector = std::decay_t<_InVector>;

        static_assert(isEigenType<InVector>(), "`x` must be an Eigen type!");
        if (symmetry_mode != SymmetryMode::NONE) throw std::runtime_error("applyTransposeParallel requires explicit storage of both upper and lower triangle (i.e., a symmetry_mode of NONE)");

        // We handle the following three cases:
        //  a) matrix entries [p x q], vector flattened as [x0_0, ..., x0_{p - 1}, x1_0, ... ]
        //  b) matrix entries [p x q], vector entries [p x 1]
        //  c) matrix entries [p x q], "vector" [Dynamic x q]
        //  d) matrix entries scalar, vector entries scalar.
        using vtraits_A = spmat_helper::value_traits<value_type>;
        using vtraits_x = spmat_helper::value_traits<typename InVector::value_type>;

        static_assert(std::is_same<typename Result::Scalar, typename InVector::Scalar>::value
                        && (long(Result::RowsAtCompileTime) == long(InVector::RowsAtCompileTime))
                        && (long(Result::ColsAtCompileTime) == long(InVector::ColsAtCompileTime)), "Result currently must be same underlying type");

        constexpr size_t p = vtraits_A::rows;
        constexpr size_t q = vtraits_A::cols;
        static_assert(((InVector::ColsAtCompileTime == 1) || (vtraits_x::rows == 1)) &&
                       (InVector::ColsAtCompileTime != Eigen::Dynamic), "vector 'blocks' must either be stored in an inner vector type or in rows of `x`");

        constexpr size_t vec_block = std::max<size_t>(vtraits_x::rows, InVector::ColsAtCompileTime);
        static_assert(((vec_block == p) || (vec_block == 1)) && (vtraits_x::cols == 1), "x elements must be scalars or [p x 1] vectors");
        static_assert( (vec_block == q) || (vec_block == 1), "result elements must be scalars for [q x 1] vectors"); // Note: we currently assume result is the same type as x...

        // Note: this transposed matrix we're applying is `(n * q) x (m * p)`,
        // so `x` should be of size `mm` and the result of size `nn`.
        if (size_t(x.rows()) * vec_block != size_t(m * p)) throw std::runtime_error("Sparse matvec size mismatch.");
        using SG = spmat_helper::SegmentGetter<p, InVector>;

        const size_t result_size = (n * q) / vec_block;
        result.resize(result_size, Result::ColsAtCompileTime);

        // Process one matrix column at a time, generating an entry of the transposed matvec result.
        auto computeEntry = [&x, &result, this](index_type j) {
			const index_type col_begin = Ap[j],
			                 col_end   = Ap[j + 1];
            if (col_begin == col_end) { spmat_helper::setZero(SG::get(result, j)); return; } // zero-column case: output zero
            if (ZeroInit) {
                typename SG::ScratchVec tmp = spmat_helper::transpose_block(Ax[col_begin]) * SG::get(x, Ai[col_begin]);
                for (index_type entry = col_begin + 1; entry < col_end; ++entry)
                    tmp += spmat_helper::transpose_block(Ax[entry]) * SG::get(x, Ai[entry]);
                if (Negate) SG::get(result, j) = -tmp;
                else        SG::get(result, j) =  tmp;
            }
            else {
                typename SG::ScratchVec tmp = SG::get(result, j);
                for (index_type entry = col_begin; entry < col_end; ++entry) {
                    if (Negate) tmp -= spmat_helper::transpose_block(Ax[entry]) * SG::get(x, Ai[entry]);
                    else        tmp += spmat_helper::transpose_block(Ax[entry]) * SG::get(x, Ai[entry]);
                }
                SG::get(result, j) = tmp;
            }
        };

        parallel_for_range(n, computeEntry, 32, 100);
    }

    template<typename _Vector>
    auto applyTransposeParallel(const _Vector &x) const {
        using Result = Eigen::Matrix<typename _Vector::Scalar, _Vector::RowsAtCompileTime, _Vector::ColsAtCompileTime>;
        Result result;
        applyTransposeParallel(x, result);
        return result;
    }

    template<class _InVector>
    value_type evalQuadraticForm(const _InVector &x) const {
        // BENCHMARK_SCOPED_TIMER_SECTION timer("CSCMatrix.evalQuadraticForm");
        if (size_t(x.rows()) != size_t(m)) throw std::runtime_error("evalQuadraticForm size mismatch.");
        if (symmetry_mode == SymmetryMode::UPPER_TRIANGLE) {
            return summation_parallel([&](size_t j) {
                const index_type col_begin = Ap[j],
                                 col_end   = Ap[j + 1];
                value_type result = 0;
                for (index_type entry = col_begin; entry < col_end; ++entry) {
                    size_t i = Ai[entry];
                    result += ((i == j) ? 1 : 2) * Ax[entry] * x[i];
                }
                return result * x[j];
            }, n);
        }
        if (symmetry_mode == SymmetryMode::NONE) {
            return summation_parallel([&](size_t j) {
                const index_type col_begin = Ap[j],
                                 col_end   = Ap[j + 1];
                value_type result = 0;
                for (index_type entry = col_begin; entry < col_end; ++entry)
                    result += Ax[entry] * x[Ai[entry]];
                return result * x[j];
            }, n);
        }

        throw std::runtime_error("Unsupported SymmetryMode");
    }

    // Remove the rows i and columns j for which remove[i] and remove[j] is true, respectively.
    // If passed, `entryForReducedEntry` will be filled with the source
    // location of each entry in the new, smaller `Ax`.
    // (or INDEX_NONE if that entry was removed).
    // If no rows/cols are removed, then `entryForReducedEntry` will be resized to zero.
    template<class Predicate>
    void rowColRemoval(const Predicate &shouldRemove, std::vector<_Index> *reducedRowForRow = nullptr, std::vector<_Index> *entryForReducedEntry = nullptr) {
        if (m != n) throw std::runtime_error("rowColRemoval only implemented for square matrices");

        // Determine the mapping from old row indices to new (reduced) row indices.
        std::vector<_Index> rr_tmp; // internal local version used only if `reducedRowForRow` was not passed
        auto rr_ptr = reducedRowForRow ? reducedRowForRow : &rr_tmp;
        std::vector<_Index> &replacementRowIdx = *rr_ptr;
        replacementRowIdx.assign(n, INDEX_NONE);

        size_t toRemove = 0;
        for (_Index reducedIdx = 0, i = 0; i < m; ++i) {
            if (shouldRemove(i)) { ++toRemove; continue; }
            replacementRowIdx[i] = reducedIdx++;
        }

        if (entryForReducedEntry) {
            if (toRemove == 0) entryForReducedEntry->clear(); // rowColRemoval was a NOP operation; entryForReducedEntry is the identity.
            else               entryForReducedEntry->resize(nz, INDEX_NONE);
        }

        if (toRemove == 0) return;

        const bool sparsityOnly = Ax.empty();

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
                if (!sparsityOnly) Ax[entry_back] = Ax[idx];
                if (entryForReducedEntry) (*entryForReducedEntry)[entry_back] = idx;
                ++entry_back;
            }
            idx_begin = idx_end;
            Ap[++colptr_back] = entry_back; // Write the new column end pointer for the kept columns
        }

        assert(colptr_back <= size_t(m));
        assert(entry_back <= size_t(nz));

        nz = entry_back;
        m = n = colptr_back;

        if (!sparsityOnly) Ax.resize(nz);
        Ai.conservativeResize(nz);
        Ap.resize(n + 1);
        if (entryForReducedEntry) (*entryForReducedEntry).resize(nz);
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
        Ai.conservativeResize(nz);
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
        for (const auto t : (*this)) result.nz.emplace_back(t);
        return result;
    }
};

// Efficient conversion of a triplet-style sparsity pattern into a CSC format
// (arrays Ap and Ai). Note that only the sparsity pattern is
// referenced/converted, and no values are accessed.
template<class TripletList, class IndexVec>
void sparsityPatternToCSC(size_t n, const TripletList &nz, IndexVec &Ap, IndexVec &Ai) {
    static constexpr size_t INDEX_NONE = std::numeric_limits<size_t>::max();

    std::vector<size_t> bucketStart(n + 1);
    {
        // Compute bucket offsets in bucketStart[1:]: first calculate size
        for (const auto &t : nz) ++bucketStart[t.j + 1];
        // Next, compute bucketStart[2:] = cumsum(bucketStart[1:])
        size_t cumsum = 0;
        for (size_t j = 1; j <= n; ++j) {
            size_t colsize_j = bucketStart[j];
            bucketStart[j] = cumsum;
            cumsum += colsize_j;
        }
        assert(cumsum == nz.size());
    }

    Eigen::Matrix<size_t, Eigen::Dynamic, 1> columnBuckets(nz.size());
    {
        // Fill the index buckets; note incrementing the offsets in
        // bucketStart[1:] by the size of each bucket converts these into the
        // end offsets.
        size_t *bucketBack = bucketStart.data() + 1;
        for (const auto &t : nz) {
            size_t newEntry = bucketBack[t.j]++;
            columnBuckets[newEntry] = t.i;
        }
    }

    Ap.resize(n + 1);

    // Sort each bucket in parallel and deduplicate.
    parallel_for_range(n, [&](size_t j) {
        auto start = columnBuckets.data() + bucketStart[j];
        auto end   = columnBuckets.data() + bucketStart[j + 1];
        std::sort(start, end);
        end = std::unique(start, end);
        Ap[j] = std::distance(start, end); // Write deduplicated bucket size
    });

    // Calculate column pointer array using cumulative sum.
    size_t newNNZ = 0;
    for (size_t j = 0; j < n; ++j) {
        size_t colsize_j = Ap[j];
        Ap[j] = newNNZ;
        newNNZ += colsize_j;
    }
    Ap[n] = newNNZ;

    // Fill row index array `Ai`
    Ai.resize(newNNZ);
    // for (size_t j = 0; j < n; ++j) { // could be parallelized
    parallel_for_range(n, [&](size_t j) {
        size_t offset = bucketStart[j];
        for (size_t ii = Ap[j]; ii < Ap[j + 1]; ++ii)
            Ai[ii] = columnBuckets[offset++];
    });
}

using SuiteSparseMatrix = CSCMatrix<SuiteSparse_long, double>;

} // namespace MeshFEM

#endif /* end of include guard: SPARSEMATRICES_HH */
