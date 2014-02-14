////////////////////////////////////////////////////////////////////////////////
// Fields.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Classes to wrap vector/tensor fields that whose values have been
//  flattened into 1D arrays as follows:
//      v(f) -> [v_1(f), v_2(f), ..., v_dim(f)]^T
//  where dim is the number of scalars specifying each vector/tensor value.
//  This allows the full field to be written as a (v_dim \times |D|) array
//  where |D| is the size of the discrete domain.
//
//  Thus, for a vector field v represented by a 2D array V, v(i) = V.col(i).
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
        Real maxNorm = 0;
        for (size_t i = 0; i < domainSize(); ++i)
            maxNorm = std::max(maxNorm, m_values.col(i).norm());
        m_values /= maxNorm;
    }

    const ArrayType &data() const { return m_values; }
          ArrayType &data()       { return m_values; }

    size_t dim() const { return t_dim; }
    size_t domainSize() const { return m_values.cols(); }

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
// matrix. This triangle is flattened into a 1D vector as follows:
//    [ 0  N  N+2 ...    ]
//    [    1  N+3 ...    ]
//    [        2         ]
//    [         ..       ]
//    [           ..     ]
//    [              N-1 ]
// This flattening is compatible with the typical stress/strain flattening that
// collects the diagonal xx, yy, ... entries at the beginning
// The total number of entries is sum_{i=1}^N i = (N * (N + 1)) / 2
// (because there are i entries in the ith column).
template<typename Real, size_t t_N>
class SymmetricMatrixField {
public:
    enum { FIELD_DIM = (t_N * (t_N + 1)) / 2 };
    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> FlattenedType;
    typedef Eigen::Matrix<Real, FIELD_DIM, Eigen::Dynamic> ArrayType;
    typedef typename ArrayType::ColXpr      ValueStorageType;
    typedef typename ArrayType::ConstColXpr ConstValueStorageType;

    static size_t compute1DIndex(size_t i, size_t j) {
        assert((i < t_N) && (j < t_N));
        size_t idx1D = -1;
        if (i == j) {
            idx1D = i;
        }
        else {
            if (j < i) {
                // First, map all array accesses to the upper triangle
                std::swap(i, j);
            }
            // Note: j > 0 because j == 0 contradicts j > i     (i >= 0)
            idx1D = t_N + ((j - 1) * j) / 2 + i;
        }

        return idx1D;
    }

    class SymmetricMatrix {
        public:
            SymmetricMatrix(const ValueStorageType &values)
                : m_data(values) { }
            size_t N() const { return t_N; }
            Real &operator()(size_t i, size_t j) {
                return m_data[compute1DIndex(i, j)];
            }

            Real operator()(size_t i, size_t j) const {
                return m_data[compute1DIndex(i, j)];
            }

            // Flattened addressing
            Real &operator[](size_t i) {
                return m_data[i];
            }

            // Flattened addressing
            Real operator[](size_t i) const {
                return m_data[i];
            }

        private:
            ValueStorageType m_data;
    };

    class ConstSymmetricMatrix {
        public:
            ConstSymmetricMatrix(const ConstValueStorageType &values)
                : m_data(values) { }
            size_t N() const { return t_N; }
            Real operator()(size_t i, size_t j) const {
                return m_data[compute1DIndex(i, j)];
            }

            // Flattened addressing
            Real operator[](size_t i) const {
                return m_data[i];
            }

        private:
            const ConstValueStorageType m_data;
    };

    SymmetricMatrixField(size_t domainSize, const FlattenedType &values)
    {
        assert(dim() * domainSize == values.rows());
        m_values = Eigen::Map<const ArrayType>(values.data(), dim(),
                                               domainSize);
    }

    SymmetricMatrixField(size_t domainSize = 0)
        : m_values(dim(), domainSize) { }
    
    size_t dim() const { return ((t_N * (t_N + 1)) / 2); }
    size_t N()   const { return t_N; }
    size_t domainSize() const { return m_values.cols(); }

    ConstSymmetricMatrix operator()(size_t i) const {
        return ConstSymmetricMatrix(m_values.col(i));
    }

    SymmetricMatrix operator()(size_t i) {
        return SymmetricMatrix(m_values.col(i));
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
