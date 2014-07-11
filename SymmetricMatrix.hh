////////////////////////////////////////////////////////////////////////////////
// SymmetricMatrix.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Classes to wrap a flattened symmetric rank 2 tensor (a 6-vector in 3D),
//  allowing it to be treated as a plain symmetric matrix.
//
//  The *SymmetricMatrixRef types provide a wrapper for reference types (e.g.
//  Eigen's ColXpr). This is useful for the SymmetricMatrixField class that
//  stores all symmetric matrix values as flattened columns in a
//  flatSize x domainSize() array; it allows values to be accessed in-place.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/18/2014 17:59:39
////////////////////////////////////////////////////////////////////////////////
#ifndef SYMMETRICMATRIX_HH
#define SYMMETRICMATRIX_HH

#include "Flattening.hh"
#include <iostream>

template<typename _Real, size_t t_N, typename _ConstSymmetricMatrix>
class ConstSymmetricMatrixBase {
public:
    static constexpr size_t N()        { return t_N; }
    static constexpr size_t flatSize() { return (N() * (N() + 1)) / 2; }

    _Real operator()(size_t i, size_t j) const {
        assert((i < N()) && (j < N()));
        return operator[](flattenIndices(N(), i, j));
    }

    Eigen::Matrix<_Real, N(), 1> eigenvalues() const {
        Eigen::Matrix<_Real, N(), N()> mat;
        for (size_t j = 0; j < N(); ++j)
            for (size_t i = 0; i <= j; ++i)
                mat(i, j) = operator()(i, j);
        return mat.template selfadjointView<Eigen::Upper>().eigenvalues();
    }

    _Real maxEigenvalue() const { return eigenvalues().maxCoeff(); }
    _Real minEigenvalue() const { return eigenvalues().minCoeff(); }

    _Real doubleContract(const ConstSymmetricMatrixBase &b) const {
        _Real result(0);
        for (size_t i = 0; i < t_N; ++i) {
            for (size_t j = 0; j < t_N; ++j) {
                result += operator()(i, j) * b(i, j);
            }
        }
        return result;
    }

    // Flattened addressing
    _Real operator[](size_t i) const {
        return (*static_cast<const _ConstSymmetricMatrix *>(this))[i];
    }

    friend std::ostream &operator<<(std::ostream &os, const ConstSymmetricMatrixBase &m) {
        for (size_t i = 0; i < t_N; ++i) {
            os << m(i, 0);
            for (size_t j = 1; j < t_N; ++j)
                os << " " << m(i, j);
            os << std::endl;
        }
        return os;
    }
};

template<typename _Real, size_t t_N, typename _SymmetricMatrix>
class SymmetricMatrixBase : public ConstSymmetricMatrixBase<_Real, t_N, _SymmetricMatrix> {
    typedef ConstSymmetricMatrixBase<_Real, t_N, _SymmetricMatrix> Base;
public:
    using Base::operator();
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

    SymmetricMatrixBase &operator*=(_Real s) {
        for (size_t i = 0; i < Base::flatSize(); ++i)
            operator[](i) *= s;
        return *this;
    }

    SymmetricMatrixBase &operator+=(const SymmetricMatrixBase &b) {
        for (size_t i = 0; i < Base::flatSize(); ++i)
            operator[](i) += b[i];
        return *this;
    }

    void clear() {
        for (size_t i = 0; i < Base::flatSize(); ++i)
            operator[](i) = 0.0;
    }

    // Flattened addressing
    using Base::operator[];
    _Real  &operator[](size_t i) { return (*static_cast<_SymmetricMatrix *>(this))[i]; }
};

template<size_t t_N, typename StorageRef, typename ConstStorageRef>
class SymmetricMatrixRef
    : public SymmetricMatrixBase<typename StorageRef::Scalar, t_N,
                              SymmetricMatrixRef<t_N, StorageRef, ConstStorageRef> >
{
    typedef typename StorageRef::Scalar _Real;
    typedef SymmetricMatrixBase<_Real, t_N, SymmetricMatrixRef<t_N, StorageRef, ConstStorageRef> > Base;
public:
    SymmetricMatrixRef(const StorageRef &values) : m_data(values) { }
    _Real &operator[](size_t i) { return m_data[i]; }
    _Real  operator[](size_t i) const { return m_data[i]; }
    StorageRef      flattened()       { return m_data; }
    ConstStorageRef flattened() const { return m_data; }
    template<typename FType>
    SymmetricMatrixRef &operator=(const FType &f) { Base::assign(f); return *this; }
private:
    StorageRef m_data;
};

template<size_t t_N, typename ConstStorageRef>
class ConstSymmetricMatrixRef
    : public SymmetricMatrixBase<typename ConstStorageRef::Scalar, t_N,
                                 ConstSymmetricMatrixRef<t_N, ConstStorageRef> >
{
    typedef typename ConstStorageRef::Scalar _Real;
public:
    ConstSymmetricMatrixRef(const ConstStorageRef &values) : m_data(values) { }
    _Real operator[](size_t i) const { return m_data[i]; }
    ConstStorageRef flattened() { return m_data; }
private:
    ConstStorageRef m_data;
};

template<size_t t_N, typename Storage>
class SymmetricMatrix
    : public SymmetricMatrixBase<typename Storage::Scalar, t_N,
                                 SymmetricMatrix<t_N, Storage> >
{
    typedef typename Storage::Scalar _Real;
    typedef SymmetricMatrixBase<_Real, t_N, SymmetricMatrix<t_N, Storage> > Base;
public:
    SymmetricMatrix() { }

    SymmetricMatrix(size_t i) : m_data(Storage::Zero()) {
        if (i >= Base::flatSize())
            throw std::runtime_error("Illegal basis element number.");
        m_data[i] = (i < t_N) ? 1.0 : 0.5;
    }

    SymmetricMatrix(const Storage &values) : m_data(values) { }

    // Construct a unit canonical basis symmetric matrix:
    // e_ij = .5 * (e_i e_j^T + e_j e_i^T)
    static SymmetricMatrix CanonicalBasis(size_t i) {
        if (i >= Base::flatSize())
            throw std::runtime_error("Illegal basis element number.");
        SymmetricMatrix e_ij(Storage::Zero());
        e_ij[i] = (i < t_N) ? 1.0 : 0.5;
        return e_ij;
    }

    _Real &operator[](size_t i)       { return m_data[i]; }
    _Real  operator[](size_t i) const { return m_data[i]; }
    Storage &flattened()             { return m_data; }
    const Storage &flattened() const { return m_data; }
    template<typename FType>
    SymmetricMatrix &operator=(const FType &f) { Base::assign(f); return *this; }
    SymmetricMatrix  operator-() const { SymmetricMatrix copy(*this); copy *= -1.0; return copy; }
private:
    Storage m_data;
};

#endif /* end of include guard: SYMMETRICMATRIX_HH */
