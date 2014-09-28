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
#include <vector>
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

    // Copy constructor
    VectorField(const VectorField &b) : m_values(b.m_values) { }

    // Flattened data constructor
    // Note: copies data
    VectorField(const FlattenedType &values) {
        size_t domainSize = values.rows() / t_dim;
        assert(t_dim * domainSize == (size_t) values.rows());
        m_values = Eigen::Map<const ArrayType>(values.data(), t_dim,
                                               domainSize);
    }

    // Flattened data constructor (std::vector version)
    template<typename Real2>
    VectorField(const std::vector<Real2> &values) {
        size_t domainSize = values.size() / t_dim;
        assert(t_dim * domainSize == values.size());
        m_values = Eigen::Map<const Eigen::Matrix<Real2, t_dim, Eigen::Dynamic> >
            (&values[0], t_dim, domainSize);
    }


    // Uninitialized allocation constructor
    VectorField(size_t domainSize = 0)
        : m_values(t_dim, domainSize) { }

    ConstValueType operator()(size_t i) const {
        assert(i < (size_t) m_values.cols());
        return m_values.col(i);
    }

    ValueType operator()(size_t i) {
        assert(i < (size_t) m_values.cols());
        return m_values.col(i);
    }

    // Arithmetic operations
    VectorField &operator*=(Real scalar) {
        m_values *= scalar;
        return *this;
    }

    VectorField &operator+=(const VectorField &b) {
        assert(domainSize() == b.domainSize());
        for (size_t i = 0; i < domainSize(); ++i) {
            m_values.col(i) += b.m_values.col(i);
        }
        return *this;
    }

    VectorField &operator-=(const VectorField &b) {
        assert(domainSize() == b.domainSize());
        for (size_t i = 0; i < domainSize(); ++i) {
            m_values.col(i) -= b.m_values.col(i);
        }
        return *this;
    }

    VectorField operator*(Real s)               const { VectorField result(*this); result *= s; return result; }
    VectorField operator+(const VectorField &b) const { VectorField result(*this); result += b; return result; }
    VectorField operator-(const VectorField &b) const { VectorField result(*this); result -= b; return result; }

    void clear() { m_values = ArrayType::Zero(dim(), domainSize()); }

    // Normalize data so that the maximum column magnitude is 1.
    void maxColumnNormalize() { m_values /= maxMag(); }

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

    // Component wise abs.
    VectorField cwiseAbs() const { auto r = VectorField(*this); r.m_values = r.m_values.cwiseAbs(); return r; }

    // Set all coefficients to a constant
    void setConstant(Real val) { m_values.setConstant(val); }

    // Sum of squared norms of each vector.
    Real frobeniusNormSq() const {
        Real normSq = 0;
        for (size_t i = 0; i < domainSize(); ++i)
            normSq += m_values.col(i).squaredNorm();
        return normSq;
    }

    const ArrayType &data() const { return m_values; }
          ArrayType &data()       { return m_values; }

    size_t dim() const { return t_dim; }
    size_t N()   const { return dim(); }
    size_t domainSize() const { return m_values.cols(); }
    FieldType fieldType() const { return FIELD_VECTOR; }

    void resizeDomain(size_t dSize) {
        m_values.resize(Eigen::NoChange, dSize);
        clear();
    }

    // Flattened access
    size_t size() const { return dim() * domainSize(); }
    void resize(size_t i) { assert(i % dim() == 0); resizeDomain(i / 3); }
    Real &operator[](size_t i)       { assert(i < size()); return m_values.data()[i]; }
    Real  operator[](size_t i) const { assert(i < size()); return m_values.data()[i]; }

    template<typename Real2>
    void getFlattened(std::vector<Real2> &v) const {
        v.resize(size());
        for (size_t i = 0; i < size(); ++i)
            v[i] = operator[](i);
    }

    void print(std::ostream &os, const std::string &componentSeparator = "\t",
               const std::string &elementPrefix = "",
               const std::string &elementSuffix = "",
               const std::string &elementSeparator = "\n") const {
        for (size_t i = 0; i < domainSize(); ++i) {
            if (i) os << elementSeparator;
            ConstValueType v = (*this)(i);
            os << elementPrefix << v[0];
            for (size_t j = 1; j < t_dim; ++j) {
                os << componentSeparator << v[j];
            }
            os << elementSuffix;
        }
    }

    void dump(const std::string &path) const {
        std::ofstream of(path);
        if (!of.is_open())
            throw std::runtime_error(std::string("Couldn't open '") +
                        path + "' for writing.");
        of << std::scientific << std::setprecision(16);
        print(of);
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

    Real norm() const { return m_values.norm(); }
    Real  sum() const { return m_values.sum(); }
    Real  min() const { return m_values.minCoeff(); }
    Real  max() const { return m_values.maxCoeff(); }

    // Return the entry with maximum/minimum magnitude
    Real minMag() const { Real m = min(), M = max(); return (std::abs(m) < M) ? m : M; }
    Real maxMag() const { Real m = min(), M = max(); return (std::abs(m) > M) ? m : M; }

    // Component wise abs.
    ScalarField cwiseAbs() const { auto r = ScalarField(*this); r.m_values = r.m_values.cwiseAbs(); return r; }

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

private:
    using VectorField<Real, 1>::m_values;
};

// Handles both VectorField and ScalarField output.
template<typename Real, size_t N>
std::ostream &operator<<(std::ostream &os, const VectorField<Real, N> &vf) {
    for (size_t i = 0; i < vf.domainSize(); ++i) {
        for (size_t c = 0; c < N; ++c) {
            os << (c ? "\t" : "") << vf(i)[c];
        }
        os << std::endl;
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
    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> FlattenedType;
    typedef Eigen::Matrix<Real, flatLen(t_N), Eigen::Dynamic> ArrayType;

    typedef SymmetricMatrixRef<t_N, typename ArrayType::ColXpr,
            typename ArrayType::ConstColXpr> ValueType;
    typedef ConstSymmetricMatrixRef<t_N,
            typename ArrayType::ConstColXpr> ConstValueType;

    SymmetricMatrixField(size_t domainSize, const FlattenedType &values) {
        assert(dim() * domainSize == values.rows());
        m_values = Eigen::Map<const ArrayType>(values.data(), dim(),
                                               domainSize);
    }

    // Eigen ArrayType constructor
    SymmetricMatrixField(const ArrayType values) : m_values(values) { }

    SymmetricMatrixField(size_t domainSize = 0)
        : m_values(dim(), domainSize) { }
    
    constexpr size_t dim() const { return flatLen(t_N); }
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

    // Component wise abs.
    SymmetricMatrixField cwiseAbs() const { return SymmetricMatrixField(m_values.cwiseAbs()); }
    // Set all coefficients to a constant
    void setConstant(Real val) { m_values.setConstant(val); }

    SymmetricMatrixField &operator=(const SymmetricMatrixField &b) {
        if (this == &b) return *this;
        m_values = b.m_values;
        return *this;
    }

    const ArrayType &data() const { return m_values; }
          ArrayType &data()       { return m_values; }

    void dump(const std::string &path) const {
        std::ofstream of(path);
        if (!of.is_open())
            throw std::runtime_error(std::string("Couldn't open '") +
                        path + "' for writing.");
        of << std::scientific << std::setprecision(16);
        for (size_t i = 0; i < domainSize(); ++i) {
            ConstValueType v = (*this)(i);
            of << v[0];
            for (size_t j = 1; j < dim(); ++j) {
                of << '\t' << v[j];
            }
            of << std::endl;
        }
    }

    void load(const std::string &path) {
        std::ifstream is(path);
        if (!is.is_open())
            throw std::runtime_error(std::string("Couldn't open '") + path);

        std::string line;
        std::vector<Real> data;
        while (std::getline(is >> std::ws, line)) {
            std::vector<Real> v;
            std::istringstream iss(line);
            Real c;
            size_t i = 0;
            while (iss >> c) { data.push_back(c); ++i; }
            if (i != dim()) throw std::runtime_error("Read wrong number of components.");
        }
        assert(data.size() % dim() == 0);
        int domSize = data.size() / dim();
        m_values = Eigen::Map<const ArrayType>(&data[0], dim(), domSize);
    }

private:
    /** Data storage */
    ArrayType m_values;
};

template<typename Real, size_t N>
std::ostream &operator<<(std::ostream &os, const SymmetricMatrixField<Real, N> &smf)
{
    for (size_t i = 0; i < smf.domainSize(); ++i) {
        for (size_t c = 0; c < smf.dim(); ++c) {
            os << (c ? "\t" : "") << smf(i)[c];
        }
        os << std::endl;
    }
    return os;
}

// Simple field class that can change dimension but is less efficient/statically
// checked.
// Stores in flattened x0 y0 x1 y1 ... format
class DynamicField {
public:
    DynamicField(size_t dimensions, size_t domSize) {
        resize(dimensions, domSize);
    }

    DynamicField(const DynamicField &b) {
        m_dim = b.m_dim;
        m_storage = b.m_storage;
    }

    template<size_t _N>
    DynamicField(const VectorField<Real, _N> &vf) {
        resize(vf.dim(), vf.domainSize());
        for (size_t i = 0; i < vf.dim(); ++i)
            for (size_t j = 0; j < vf.domainSize(); ++j)
                (*this)(i, j) = vf(j)[i];
    }

    void resize(size_t domSize) { m_storage.resize(domSize * m_dim); }
    void resize(size_t dim, size_t domSize) { m_dim = dim; resize(domSize); }

    size_t domainSize() const {
        assert(m_storage.size() % m_dim == 0);
        return m_storage.size() / m_dim;
    }

    size_t dim() const { return m_dim; }

    // Flattened access
    Real &operator[](size_t i)       { return m_storage.at(i); }
    Real  operator[](size_t i) const { return m_storage.at(i); }

    Real &operator()(size_t i, size_t j) {
        if (i >= dim() || j >= domainSize()) throw std::runtime_error("out of bounds access");
        return m_storage.at(j * dim() + i);
    }

    Real  operator()(size_t i, size_t j) const {
        if (i >= dim() || j >= domainSize()) throw std::runtime_error("out of bounds access");
        return m_storage.at(j * dim() + i);
    }

    // Casts to Field types.
    operator ScalarField<Real>() const {
        if (m_dim != 1) throw std::runtime_error("Illegal cast of vector field to scalar field.");
        return ScalarField<Real>(m_storage);
    }
    template<size_t _dim>
    operator VectorField<Real, _dim>() const {
        if (m_dim != _dim) throw std::runtime_error("Vector field cast dimension mismatch.");
        return VectorField<Real, _dim>(m_storage);
    }
    template<size_t _dim>
    operator SymmetricMatrixField<Real, _dim>() const {
        if (m_dim != _dim) throw std::runtime_error("Vector field cast dimension mismatch.");
        return SymmetricMatrixField<Real, _dim>(m_storage);
    }

    friend std::ostream &operator<<(std::ostream &os, const DynamicField &f) {
        for (size_t j = 0; j < f.domainSize(); ++j) {
            for (size_t i = 0; i < f.dim(); ++i)
                os << (i ? "\t" : "") << f(i, j);
            os << std::endl;
        }
        return os;
    }
    
private:
    size_t m_dim;
    std::vector<Real> m_storage;
};

#endif // FIELDS_HH
