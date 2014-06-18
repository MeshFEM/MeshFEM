////////////////////////////////////////////////////////////////////////////////
// Fields.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Classes implementing vector/scalar/tensor fields. Each class also specifies
//  how the quantities are flattened into a single 1D array. Field samples are
//  stored as columns of a dim x |D| 2D array that is then flattened in column
//  major format. Here |D| is the size of the discrete domain.
//
//  This means, for symmetric tensor fields, there are two flattenings: first
//  each sample is flattened into a 6-vector (in 3D) using Voigt notation, then
//  each 6-vector is stored as a column in a 6 x |D| array, which is flattened
//  into a 6 |D| vector.
//
//  For vector fields, the resulting flattened vector looks like:
//      [v_0x, v_0y, v_0z, v_1x, ..., v_|D|z]
//  This vector can be obtained with the getFlattened() method.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/13/2013 16:27:14
////////////////////////////////////////////////////////////////////////////////
#ifndef FIELDS_HH
#define FIELDS_HH
#include <Eigen/Dense>
#include <string>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "Flattening.hh"
#include "SymmetricMatrix.hh"

typedef enum { FIELD_SCALAR, FIELD_VECTOR, FIELD_MATRIX} FieldType;

template<typename Real, size_t t_dim>
class VectorField {
public:
    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> FlattenedType;
    typedef Eigen::Matrix<Real, t_dim, Eigen::Dynamic> ArrayType;
    typedef typename ArrayType::ColXpr         ValueType;
    typedef typename ArrayType::ConstColXpr    ConstValueType;

    // Note: copies data
    VectorField(const FlattenedType &values)
    {
        size_t domainSize = values.rows() / t_dim;
        assert(t_dim * domainSize == (size_t) values.rows());
        m_values = Eigen::Map<const ArrayType>(values.data(), t_dim,
                                               domainSize);
    }

    template<typename Real2>
    VectorField(const std::vector<Real2> &values) {
        size_t domainSize = values.size() / t_dim;
        assert(t_dim * domainSize == values.size());
        m_values = Eigen::Map<const Eigen::Matrix<Real2, t_dim, Eigen::Dynamic> >
            (&values[0], t_dim, domainSize);
    }

    VectorField(size_t domainSize = 0)
        : m_values(t_dim, domainSize) { }

    VectorField &operator*=(Real scalar) {
        m_values *= scalar;
        return *this;
    }

    ConstValueType operator()(size_t i) const {
        assert(i < (size_t) m_values.cols());
        return m_values.col(i);
    }

    ValueType operator()(size_t i) {
        assert(i < (size_t) m_values.cols());
        return m_values.col(i);
    }

    void clear() {
        m_values = ArrayType::Zero(dim(), domainSize());
    }

    // Normalize data so that the maximum column magnitude is 1.
    void maxColumnNormalize() {
        m_values /= maxMag();
    }

    Real maxMag() const {
        Real maxNorm = 0;
        for (size_t i = 0; i < domainSize(); ++i)
            maxNorm = std::max(maxNorm, m_values.col(i).norm());
        return maxNorm;
    }

    Real minMag() const {
        Real minNorm = std::numeric_limits<Real>::max();
        for (size_t i = 0; i < domainSize(); ++i)
            minNorm = std::min(minNorm, m_values.col(i).norm());
        return minNorm;
    }

    const ArrayType &data() const { return m_values; }
          ArrayType &data()       { return m_values; }

    template<typename Real2>
    void getFlattened(std::vector<Real2> &v) const {
        size_t size = domainSize() * dim();
        v.resize(size);
        for (size_t i = 0; i < size; ++i)
            v[i] = (Real2) m_values.data()[i];
    }

    size_t dim() const { return t_dim; }
    size_t domainSize() const { return m_values.cols(); }
    FieldType fieldType() const { return FIELD_VECTOR; }

    // stub for interchangeability with SymmetricMatrixField
    size_t N() const { assert(false); }

    void resizeDomain(size_t dSize) {
        m_values.resize(Eigen::NoChange, dSize);
        clear();
    }

    void dump(const std::string &path) const {
        std::ofstream of(path);
        if (!of.is_open())
            throw std::runtime_error(std::string("Couldn't open '") +
                        path + "' for writing.");
        of << std::scientific << std::setprecision(16);
        size_t N = domainSize();
        for (size_t i = 0; i < N; ++i) {
            ConstValueType v = (*this)(i);
            of << v[0];
            for (size_t j = 1; j < t_dim; ++j) {
                of << '\t' << v[j];
            }
            of << std::endl;
        }
    }

protected:
    /** Data storage */
    ArrayType m_values;
};

template<typename Real>
class ScalarField : public VectorField<Real, 1> {
public:
    using typename VectorField<Real, 1>::FlattenedType;
    typedef Real value_type;

    ScalarField(const FlattenedType &values)
        : VectorField<Real, 1>(values) { }
    ScalarField(size_t domainSize = 0)
        : VectorField<Real, 1>(domainSize) { }
    template<typename Real2>
    ScalarField(const std::vector<Real2> &values)
        : VectorField<Real, 1>(values) { }

    FieldType fieldType() const { return FIELD_SCALAR; }

    // Also provide direct access to values in the scalar field case
    // (So this looks just like an array)
    Real  operator[](size_t i) const { return m_values[i]; }
    Real &operator[](size_t i)       { return m_values[i]; }
    size_t size() const { return this->domainSize(); }

    Real min() const { return m_values.minCoeff(); }
    Real max() const { return m_values.maxCoeff(); }
    // Return the magnitude of the entry with maximum magnitude
    Real maxMag() const { return std::max(std::abs(min()), std::abs(max())); }
    // Return the (signed) entry with maximum magnitude
    Real signedMaxMag() const {
        Real minVal = min(), maxVal = max();
        return (std::abs(minVal) > std::abs(maxVal)) ? minVal : maxVal;
    }

    const Real *data() const { return m_values.data(); }
          Real *data()       { return m_values.data(); }
    template<size_t dim>
    VectorField<Real, dim> unflatten() const {
        return VectorField<Real, dim>(m_values);
    }

    // this = min(this, b)
    void minRelax(const ScalarField<Real> &b) {
        m_values = m_values.cwiseMin(b.m_values);
    }

    // this = max(this, b)
    void maxRelax(const ScalarField<Real> &b) {
        m_values = m_values.cwiseMax(b.m_values);
    }

    void dump(const std::string &path) const {
        std::ofstream of(path);
        if (!of.is_open())
            throw std::runtime_error(std::string("Couldn't open '") +
                        path + "' for writing.");
        of << std::scientific << std::setprecision(16);
        size_t N = size();
        for (size_t i = 0; i < N; ++i) {
            of << (*this)[i] << std::endl;
        }
    }

private:
    using VectorField<Real, 1>::m_values;
};

template<typename Real>
std::ostream &operator<<(std::ostream &os, const ScalarField<Real> &sf)
{
    size_t N = sf.size();
    for (size_t i = 0; i < N; ++i) {
        os << sf[i] << std::endl;
    }

    return os;
}


// Symmetric matrix NxN fields need only store the upper triangle of the NxN
// matrix. This triangle is flattened into a 1D vector following Voigt notation.
//  [ 0 2 ]   [ 0 5 4 ]  ...  [ 0  N*(N+1)/2 -1  ]
//  [   1 ]   [   1 3 ]       [    1             ]
//            [     2 ]       [        2     ... ]
//                            [         ..   N+1 ]
//                            [           .. N   ]
//                            [              N-1 ]
// This is the typical stress/strain flattening that
// collects the diagonal xx, yy, ... entries at the beginning
// The total number of entries is sum_{i=1}^N i = (N * (N + 1)) / 2
// (because there are i entries in the ith column).
template<typename Real, size_t t_N>
class SymmetricMatrixField {
public:
    enum { FIELD_DIM = (t_N * (t_N + 1)) / 2 };
    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> FlattenedType;
    typedef Eigen::Matrix<Real, FIELD_DIM, Eigen::Dynamic> ArrayType;

    typedef SymmetricMatrixRef<t_N, typename ArrayType::ColXpr,
            typename ArrayType::ConstColXpr> ValueType;
    typedef ConstSymmetricMatrixRef<t_N,
            typename ArrayType::ConstColXpr> ConstValueType;

    SymmetricMatrixField(size_t domainSize, const FlattenedType &values) {
        assert(dim() * domainSize == values.rows());
        m_values = Eigen::Map<const ArrayType>(values.data(), dim(),
                                               domainSize);
    }

    SymmetricMatrixField(size_t domainSize = 0)
        : m_values(dim(), domainSize) { }
    
    size_t dim() const { return ((t_N * (t_N + 1)) / 2); }
    size_t N()   const { return t_N; }
    size_t domainSize() const { return m_values.cols(); }
    FieldType fieldType() const { return FIELD_MATRIX; }

    void clear() { m_values = ArrayType::Zero(dim(), domainSize()); }
    void resizeDomain(size_t dSize) {
        m_values.resize(Eigen::NoChange, dSize);
        clear();
    }

    ConstValueType operator()(size_t i) const {
        return ConstValueType(m_values.col(i));
    }

    ValueType operator()(size_t i) {
        return ValueType(m_values.col(i));
    }

    SymmetricMatrixField &operator*=(Real scalar) {
        m_values *= scalar;
        return *this;
    }

    const ArrayType &data() const { return m_values; }
          ArrayType &data()       { return m_values; }

private:
    /** Data storage */
    ArrayType m_values;
};

#endif // FIELDS_HH
