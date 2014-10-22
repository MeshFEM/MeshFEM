////////////////////////////////////////////////////////////////////////////////
// SymmetricMatrix.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Classes to wrap a flattened symmetric rank 2 tensor (a 6-vector in 3D),
//  allowing it to be treated as a plain symmetric matrix.
//
//  Nearly all implementation is done in ConstSymmetricMatrixBase and
//  SymmetricMatrixBase which encapsulate the flattened data and provide most of
//  the operations needed. These classes work with both reference types
//  (aliases SymmetricMatrixRef and ConstSymmetricMatrixRef) and value types
//  (subclass SymmetricMatrix)
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/18/2014 17:59:39
////////////////////////////////////////////////////////////////////////////////
#ifndef SYMMETRICMATRIX_HH
#define SYMMETRICMATRIX_HH

#include "Flattening.hh"
#include <Eigen/Dense>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
template<size_t t_N, typename Storage>
class SymmetricMatrix;

template<typename _Real, size_t t_N,
         typename _Storage_t, typename _ConstStorageRef_t>
class ConstSymmetricMatrixBase {
public:
    static_assert(t_N > 0, "Dimension must be positive");
    ConstSymmetricMatrixBase(const _Storage_t &data) : m_data(data) { }
    static constexpr size_t N = t_N;
    static constexpr size_t flatSize() { return (N * (N + 1)) / 2; }

    _Real operator()(size_t i, size_t j) const {
        assert((i < N) && (j < N));
        return operator[](flattenIndices(N, i, j));
    }

    Eigen::Matrix<_Real, N, 1> eigenvalues() const {
        Eigen::Matrix<_Real, N, N> mat;
        for (size_t j = 0; j < N; ++j)
            for (size_t i = 0; i <= j; ++i)
                mat(i, j) = operator()(i, j);
        return mat.template selfadjointView<Eigen::Upper>().eigenvalues();
    }

    _Real maxEigenvalue() const { return eigenvalues().maxCoeff(); }
    _Real minEigenvalue() const { return eigenvalues().minCoeff(); }
    _Real maxMagnitudeEigenvalue() const {
        auto eigs = eigenvalues();
        _Real maxEig = eigs.maxCoeff();
        _Real minEig = eigs.minCoeff();
        return (maxEig > std::abs(minEig)) ? maxEig : minEig;
    }

    _Real doubleContract(const ConstSymmetricMatrixBase &b) const {
        // Diagonals
        _Real result(m_data[0] * b.m_data[0]);
        for (size_t i = 1; i < t_N; ++i)
            result += m_data[i] * b.m_data[i];
        // Off diagonals
        for (size_t i = t_N; i < flatSize(); ++i)
            result += 2 * m_data[i] * b.m_data[i];
        return result;
    }

    // Flattened accessors
    _ConstStorageRef_t flattened() const { return m_data; }
    _Real operator[](size_t i) const { return m_data[i]; }

    // Allow us to masquarade as an eigen vector too.
    size_t rows() const { return flatSize(); }

    friend std::ostream &operator<<(std::ostream &os, const ConstSymmetricMatrixBase &m) {
        for (size_t i = 0; i < t_N; ++i) {
            os << m(i, 0);
            for (size_t j = 1; j < t_N; ++j)
                os << " " << m(i, j);
            os << std::endl;
        }
        return os;
    }
protected:
    _Storage_t m_data;

    // SymmetricMatrix needs access to our m_data for efficient copy
    // construction.
    template<size_t t_N2, typename Storage>
    friend class SymmetricMatrix;
};

template<typename _Real, size_t t_N,
         typename _Storage_t, typename _ConstStorageRef_t, typename _StorageRef_t>
class SymmetricMatrixBase : public ConstSymmetricMatrixBase<_Real, t_N, _Storage_t, _ConstStorageRef_t> {
    typedef ConstSymmetricMatrixBase<_Real, t_N, _Storage_t, _ConstStorageRef_t> Base;
public:
    SymmetricMatrixBase(const _Storage_t &data) : Base(data) { }

    _Real &operator()(size_t i, size_t j) {
        assert((i < t_N) && (j < t_N));
        return operator[](flattenIndices(t_N, i, j));
    }

    template<typename FType>
    void assign(const FType &f) {
        assert(f.rows() == Base::flatSize());
        for (size_t i = 0; i < Base::flatSize(); ++i)
            operator[](i) = f[i];
    }

    // Warning: template hidden by derived class's default operator=, preventing
    // mixed derived assignments unless subclass has "using Base::operator="
    // Note: this can work both for flatened data types and
    // ConstSymmetricMatrixBase due to symmetric matrices' flattened access
    // operators.
    template<typename FType>
    SymmetricMatrixBase &operator=(const FType &f) { assign(f); return *this; }

    SymmetricMatrixBase &operator*=(_Real s) {
        for (size_t i = 0; i < Base::flatSize(); ++i)
            operator[](i) *= s;
        return *this;
    }

    template<typename FType>
    SymmetricMatrixBase &operator+=(const FType &b) {
        assert(b.rows() == Base::flatSize());
        for (size_t i = 0; i < Base::flatSize(); ++i)
            operator[](i) += b[i];
        return *this;
    }

    void clear() {
        for (size_t i = 0; i < Base::flatSize(); ++i)
            operator[](i) = 0.0;
    }

    // Flattened accessors
    // Bring in the base classes' definitions so they aren't hidden!!!
    using Base::operator[];
    using Base::flattened;
    _StorageRef_t flattened() { return m_data; }
    _Real &operator[](size_t i) { return m_data[i]; }

protected:
    using Base::m_data;
};

// SymmetricMatrixRef and ConstStorageRef are now just aliases to the bases...
template<size_t t_N, typename StorageRef, typename ConstStorageRef>
using SymmetricMatrixRef = SymmetricMatrixBase<typename StorageRef::Scalar, t_N,
                StorageRef, ConstStorageRef, StorageRef>;
template<size_t t_N, typename ConstStorageRef>
using ConstSymmetricMatrixRef = ConstSymmetricMatrixBase<typename
                ConstStorageRef::Scalar, t_N, ConstStorageRef, ConstStorageRef>;

// SymmetricMatrix needs to provide a few extra features that only make sense
// for storage-backed, non-reference-type matrices.
template<size_t t_N, typename Storage>
class SymmetricMatrix
    : public SymmetricMatrixBase<typename Storage::Scalar, t_N,
                                Storage, const Storage &, Storage &>
{
    typedef typename Storage::Scalar _Real;
    typedef SymmetricMatrixBase<_Real, t_N, Storage, const Storage &, Storage &> Base;
public:
    using Base::Base;
    SymmetricMatrix() : Base(Storage::Zero()) { }
    SymmetricMatrix(size_t i) : Base(Storage::Zero()) {
        if (i >= Base::flatSize())
            throw std::runtime_error("Illegal basis element number.");
        this->operator[](i) = (i < t_N) ? 1.0 : 0.5;
    }
    template<typename _ST2, typename _CSRT2>
    SymmetricMatrix(const ConstSymmetricMatrixBase<_Real, t_N, _ST2, _CSRT2> &b) : Base(b.m_data) { }

    // Construct a unit canonical basis symmetric matrix:
    // e_ij = .5 * (e_i e_j^T + e_j e_i^T)
    static SymmetricMatrix CanonicalBasis(size_t i) {
        if (i >= Base::flatSize())
            throw std::runtime_error("Illegal basis element number.");
        SymmetricMatrix e_ij(Storage::Zero());
        e_ij[i] = (i < t_N) ? 1.0 : 0.5;
        return e_ij;
    }

    using Base::operator=; // Would be hidden by default operator=!!!
    SymmetricMatrix operator-() const { SymmetricMatrix result(*this); result *= -1.0; return result; }
};

////////////////////////////////////////////////////////////////////////////////
// Arithmetic operators--always have a storage-backed result.
////////////////////////////////////////////////////////////////////////////////
template<typename _Real, size_t t_N,
         typename _Storage_t, typename _ConstStorageRef_t>
SymmetricMatrix<t_N, Eigen::Matrix<Real, flatLen(t_N), 1>> operator*(_Real s, const ConstSymmetricMatrixBase<_Real, t_N, _Storage_t, _ConstStorageRef_t> &mat)
{
    SymmetricMatrix<t_N, Eigen::Matrix<_Real, flatLen(t_N), 1>> result(mat);
    result *= s;
    return result;
}

template<typename _Real, size_t t_N,
         typename _Storage_t, typename _ConstStorageRef_t>
SymmetricMatrix<t_N, Eigen::Matrix<_Real, flatLen(t_N), 1>> operator*(const ConstSymmetricMatrixBase<_Real, t_N, _Storage_t, _ConstStorageRef_t> &mat, _Real s)
{
    SymmetricMatrix<t_N, Eigen::Matrix<_Real, flatLen(t_N), 1>> result(mat);
    result *= s;
    return result;
}

#endif /* end of include guard: SYMMETRICMATRIX_HH */
