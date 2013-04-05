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
#include <cassert>
#include <algorithm>

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

    const ArrayType &data() const { return m_values; }
          ArrayType &data()       { return m_values; }

    size_t dim() const { return t_dim; }
    size_t domainSize() const { return m_values.cols(); }

    void resizeDomain(size_t dSize) {
        m_values.resize(Eigen::NoChange, dSize);
        clear();
    }

protected:
    /** Data storage */
    ArrayType m_values;
};

template<typename Real>
class ScalarField : public VectorField<Real, 1> {
public:
    using typename VectorField<Real, 1>::FlattenedType;

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
    const Real *data() const { return m_values.data(); }
          Real *data()       { return m_values.data(); }
private:
    using VectorField<Real, 1>::m_values;
};


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

    const ArrayType &data() const { return m_values; }
          ArrayType &data()       { return m_values; }

private:
    /** Data storage */
    ArrayType m_values;
};

#endif // FIELDS_HH
