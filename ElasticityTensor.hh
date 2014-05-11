////////////////////////////////////////////////////////////////////////////////
// ElasticityTensor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Implements a rank 4 tensor with the symmetries of an elasticity tensor:
//		E_ijkl = E_jikl = E_ijlk = E_klij
//		This allows the tensor to be stored as a symmetric 6x6 matrix.
//		See doc/meshless_fem/TensorFlattening.pdf
//		for details of this transformation.
//
//      Currently the major symmetry isn't exploited to simplify homogenization
//      double contraction operations, but it may be as a future optimization.
//
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/11/2014 15:14:13
////////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICITYTENSOR_HH
#define ELASTICITYTENSOR_HH

#include <Eigen/Dense>

// Length of a flattened rank 2 tensor in "dim" dimensions.
// This is also the row and column size of the flattened rank 4 tensor.
constexpr int flatLen(int dim) { return (dim == 3) ? 6 : 3; }

template<typename Real, int _Dim>
class ElasticityTensor {
public:
    typedef Eigen::Matrix<Real, flatLen(_Dim), flatLen(_Dim)> DType;
    typedef typename DType::RowXpr                            RowXpr;
    typedef typename DType::ConstRowXpr                       ConstRowXpr;
    typedef Eigen::Matrix<Real, flatLen(_Dim), 1> FlattenedRank2Tensor;

    ElasticityTensor() : m_d(DType::Zero()) { }
    // Construct the elasticity tensor with a Young's modulus and Poisson ratio
    ElasticityTensor(Real E, Real nu) { setIsotropic(E, nu); }

    // Configure the elasticity tensor with a Young's modulus and Poisson ratio
    void setIsotropic(Real E, Real nu) {
        // Lame formula:
        // stress = lamda * trace(strain) + mu (strain + strain^T)
        // (We write it this way so that the implied elasticity tensor has the
        //  correct symmetries.
        Real lambda = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 + 2.0 * nu);

        m_d =  DType::Zero();
        if (_Dim == 3) {
            m_d(0, 0) = lambda + 2 * mu; m_d(0, 1) = lambda;          m_d(0, 2) = lambda;
            m_d(1, 0) = lambda;          m_d(1, 1) = lambda + 2 * mu; m_d(1, 2) = lambda;
            m_d(2, 0) = lambda;          m_d(1, 1) = lambda;          m_d(1, 2) = lambda + 2 * mu;
            m_d(3, 3) = m_d(4, 4) = m_d(5, 5) = mu;
        }
        else {
            m_d(0, 0) = lambda + 2 * mu; m_d(0, 1) = lambda;
            m_d(1, 0) = lambda;          m_d(1, 1) = lambda + 2 * mu;
            m_d(2, 2) = mu;
        }
    }

    // Implements flattening of symmetric 2D indices into 1D indices
    constexpr size_t flattenIndices(size_t i, size_t j) const {
        return (i == j) ? i :
               ((i < j) ? (_Dim * (_Dim + 1) - j * (j - 1)) / 2 - (i + 1)
                        : (_Dim * (_Dim + 1) - i * (i - 1)) / 2 - (j + 1));
    }

    Real operator()(size_t i, size_t j, size_t k, size_t l) const {
        assert((i < _Dim) && (j < _Dim) && (k < _Dim) && (l < _Dim));
        size_t ij = flattenIndices(i, j);
        size_t kl = flattenIndices(k, l);
        return D(ij, kl);
    }

    Real D(size_t i, size_t j) const {
        assert((i < (size_t) m_d.rows()) && (j < (size_t) m_d.cols()));
        return m_d(i, j);
    }

    Real &D(size_t i, size_t j) {
        assert((i < (size_t) m_d.rows()) && (j < (size_t) m_d.cols()));
        return m_d(i, j);
    }

    ElasticityTensor &operator*=(Real s) { m_d *= s; }
    ElasticityTensor  operator*(Real s) const {
        ElasticityTensor E(*this);
        E *= s;
        return E;
    }

    // Access rows of the flattened  elasticity tensor.
    // (Useful for implementing the periodic homogenization equations).
    const DType &D() const { return m_d; }
          DType &D()       { return m_d; }

    // Doubles the off-diagonal entries of a flattened symmetric rank 2 tensor.
    FlattenedRank2Tensor shearDoubler(FlattenedRank2Tensor t) const {
        for (size_t i = _Dim; i < t.rows(); ++i)
            t[i] *= 2.0;
        return t;
    }

    // The operation is D * S * strain, where S is the "Shear doubling" matrix
    // need to implement contraction E_ijkl e_kl
    // (see doc/meshless_fem/TensorFlattening.pdf)
    FlattenedRank2Tensor doubleContract(const FlattenedRank2Tensor &in) const {
        return m_d * shearDoubler(in);
    }

private:
    DType m_d;
};

#endif /* end of include guard: ELASTICITYTENSOR_HH */
