////////////////////////////////////////////////////////////////////////////////
// ElasticityTensor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a rank 4 tensor with the symmetries of an elasticity tensor:
//          E_ijkl = E_jikl = E_ijlk = E_klij
//      This allows the tensor to be stored as a symmetric 6x6 matrix "D"
//      See doc/meshless_fem/TensorFlattening.pdf
//      for details of this transformation.
//      
//      Major symmetry is enforced by only storing the upper triangle of D
//      internally. This means matrix element accesses must be done through
//      method D(i, j), and matrix operations need to be performed with
//      Eigen's "selfadjointView<Eigen::Upper>" view. For safety, because of
//      this complexity, m_d is kept entirely private, with no direct accessor.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/11/2014 15:14:13
////////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICITYTENSOR_HH
#define ELASTICITYTENSOR_HH

#include <iostream>
#include <Eigen/Dense>
#include "Flattening.hh"
#include "SymmetricMatrix.hh"

template<typename Real, int _Dim>
class ElasticityTensor {
public:
    typedef Eigen::Matrix<Real, flatLen(_Dim), flatLen(_Dim)> DType;
    typedef typename DType::RowXpr                            RowXpr;
    typedef typename DType::ConstRowXpr                       ConstRowXpr;
    // Wraps a row of the flattened elasticity tensor with a symmetric matrix
    // interface--useful for periodic homogenization formulas where rows of the
    // flattened homogenized elasticity tensor are modulated by flattened
    // fluctuation stresses.
    typedef SymmetricMatrixRef<_Dim, RowXpr, ConstRowXpr>     SMRowWrapper;
    typedef ConstSymmetricMatrixRef<_Dim, ConstRowXpr>        ConstSMRowWrapper;
    typedef Eigen::Matrix<Real, flatLen(_Dim), 1> FlattenedRank2Tensor;

    ElasticityTensor() : m_d(DType::Zero()) { }
    // Construct the elasticity tensor with a Young's modulus and Poisson ratio
    ElasticityTensor(Real E, Real nu) { setIsotropic(E, nu); }
    // Copy constructor
    ElasticityTensor(const ElasticityTensor &b) { m_d = b.m_d; }

    // Configure the elasticity tensor with a Young's modulus and Poisson ratio
    void setIsotropic(Real E, Real nu) {
        // Lame formula:
        // stress = lamda * trace(strain) + mu (strain + strain^T)
        // (We write it this way so that the implied elasticity tensor has the
        //  correct symmetries.)
        Real lambda = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 + 2.0 * nu);

        m_d =  DType::Zero();
        if (_Dim == 3) {
            m_d(0, 0) = lambda + 2 * mu; m_d(0, 1) = lambda;          m_d(0, 2) = lambda;
                                         m_d(1, 1) = lambda + 2 * mu; m_d(1, 2) = lambda;
                                                                      m_d(2, 2) = lambda + 2 * mu;
            m_d(3, 3) = m_d(4, 4) = m_d(5, 5) = mu;
        }
        else {
            // For 2D (plane strain), lambda is actually different...
            // This can be found by inverting 2D orthotropic tensor with equal
            // Young's moduli
            Real lambda = (nu * E) / (1.0 - nu * nu);
            m_d(0, 0) = lambda + 2 * mu; m_d(0, 1) = lambda;
                                         m_d(1, 1) = lambda + 2 * mu;
            m_d(2, 2) = mu;
        }
    }

    void setOrthotropic3D(Real   Ex, Real   Ey, Real   Ez,
                          Real nuYX, Real nuZX, Real nuZY,
                          Real muYZ, Real muZX, Real muXY) {
        if (_Dim != 3)
            throw std::runtime_error("setOrthotropic3D call on non-3D tensor");
        // Note: this isn't the flattened compliance tensor! Rather, it is the
        // matrix inverse of the flattened elasticity tensor. See the tensor
        // flattening writeup.
        m_d << 1.0 / Ex, -nuYX / Ey, -nuZX / Ez,        0.0,        0.0,        0.0,
                    0.0,   1.0 / Ey, -nuZY / Ez,        0.0,        0.0,        0.0,
                    0.0,        0.0,   1.0 / Ez,        0.0,        0.0,        0.0,
                    0.0,        0.0,        0.0, 1.0 / muYZ,        0.0,        0.0,
                    0.0,        0.0,        0.0,        0.0, 1.0 / muZX,        0.0,
                    0.0,        0.0,        0.0,        0.0,        0.0, 1.0 / muXY;
        m_d = m_d.template selfadjointView<Eigen::Upper>();
        m_d = m_d.inverse().eval();
    }

    void setOrthotropic2D(Real Ex, Real Ey, Real nuYX, Real muXY) {
        if (_Dim != 2)
            throw std::runtime_error("setOrthotropic2D call on non-2D tensor");
        // Note: this isn't the flattened compliance tensor! Rather, it is the
        // matrix inverse of the flattened elasticity tensor.
        m_d << 1.0 / Ex, -nuYX / Ey,        0.0,
                    0.0,   1.0 / Ey,        0.0,
                    0.0,        0.0, 1.0 / muXY;
        m_d = m_d.template selfadjointView<Eigen::Upper>();
        m_d = m_d.inverse().eval();
    }

    // Get the orthotropic material paramters (assuming the material is in fact
    // 3D orthotropic)
    void getOrthotropic3D(Real &  Ex, Real &  Ey, Real &  Ez,
                          Real &nuYX, Real &nuZX, Real &nuZY,
                          Real &muYZ, Real &muZX, Real &muXY) const {
        if (_Dim != 3)
            throw std::runtime_error("getOrthotropic3D call on non-3D tensor");
        ElasticityTensor Einv = this->inverse();
        Ex = 1.0 / Einv.D(0, 0);
        Ey = 1.0 / Einv.D(1, 1);
        Ez = 1.0 / Einv.D(2, 2);
        nuYX = -Einv.D(0, 1) * Ey;
        nuZX = -Einv.D(0, 2) * Ez;
        nuZY = -Einv.D(1, 2) * Ez;
        // Recall: shear terms in the compliance tensor are actually 1/(4mu)
        // (See Tensor Flatteneing writeup)
        muYZ = 0.25 / Einv.D(3, 3);
        muZX = 0.25 / Einv.D(4, 4);
        muXY = 0.25 / Einv.D(5, 5);
    }

    // Get the orthotropic material paramters (assuming the material is in fact
    // 2D orthotropic)
    void getOrthotropic2D(Real &Ex, Real &Ey, Real &nuYX, Real &muXY) const {
        if (_Dim != 2)
            throw std::runtime_error("getOrthotropic2D call on non-2D tensor");
        ElasticityTensor Einv = this->inverse();
        Ex = 1.0 / Einv.D(0, 0);
        Ey = 1.0 / Einv.D(1, 1);
        nuYX = -Einv.D(0, 1) * Ey;
        // Recall: shear terms in the compliance tensor are actually 1/(4mu)
        // (See Tensor Flattening writeup)
        muXY = 0.25 / Einv.D(2, 2);
    }

    void printOrthotropic(std::ostream &os) const {
        if (_Dim == 2) {
            Real Ex, Ey, nuYX, muXY;
            getOrthotropic2D(Ex, Ey, nuYX, muXY);
            os << Ex << "\t" << Ey << "\t" << nuYX << "\t" << muXY << std::endl;
        }
        else {
            Real Ex, Ey, Ez, nuYX, nuZX, nuZY, muYZ, muZX, muXY;
            getOrthotropic3D(Ex, Ey, Ez, nuYX, nuZX, nuZY, muYZ, muZX, muXY);
            os << Ex << "\t" << Ey << "\t" << Ez << "\t"
               << nuYX << "\t" << nuZX << "\t" << nuZY << "\t"
               << muYZ << "\t" << muZX << "\t" << muXY << std::endl;
        }
    }

    Real anisotropy() const {
        Real mu_avg, E_avg, nu_avg;
        if (_Dim == 2) {
            Real Ex, Ey, nuYX, muXY;
            getOrthotropic2D(Ex, Ey, nuYX, muXY);
            E_avg = (Ex + Ey) / 2.0;
            nu_avg = nuYX;
            mu_avg = muXY;
        }
        else {
            Real Ex, Ey, Ez, nuYX, nuZX, nuZY, muYZ, muZX, muXY;
            getOrthotropic3D(Ex, Ey, Ez, nuYX, nuZX, nuZY, muYZ, muZX, muXY);
            E_avg = (Ex + Ey + Ez) / 3.0;
            nu_avg = (nuYX + nuZX + nuZY) / 3.0;
            mu_avg = (muYZ + muZX + muXY) / 3.0;
        }
        return mu_avg / (E_avg / (2 * (1 + nu_avg)));
    }

    void clear() {
        m_d =  DType::Zero();
    }

    Real operator()(size_t i, size_t j, size_t k, size_t l) const {
        assert((i < _Dim) && (j < _Dim) && (k < _Dim) && (l < _Dim));
        size_t ij = flattenIndices(_Dim, i, j);
        size_t kl = flattenIndices(_Dim, k, l);
        return D(ij, kl);
    }

    Real D(size_t i, size_t j) const {
        assert((i < (size_t) m_d.rows()) && (j < (size_t) m_d.cols()));
        return (i <= j) ? m_d(i, j) : m_d(j, i);
    }

    Real &D(size_t i, size_t j) {
        assert((i < (size_t) m_d.rows()) && (j < (size_t) m_d.cols()));
        return (i <= j) ? m_d(i, j) : m_d(j, i);
    }

    ConstRowXpr DRow(size_t i) const { assert(i < (size_t) m_d.rows()); return m_d.row(i); }
    RowXpr      DRow(size_t i)       { assert(i < (size_t) m_d.rows()); return m_d.row(i); }
    ConstSMRowWrapper DRowAsSymMatrix(size_t i) const { return ConstSMRowWrapper(DRow(i)); }
         SMRowWrapper DRowAsSymMatrix(size_t i)       { return      SMRowWrapper(DRow(i)); }

    // Get the flattened tensor's diagonal
    Eigen::Matrix<Real, flatLen(_Dim), 1> diag() const {
        return m_d.diagonal();
    }

    ElasticityTensor &operator*=(Real s) { m_d *= s; return *this; }
    ElasticityTensor &operator/=(Real s) { m_d /= s; return *this; }
    ElasticityTensor  operator*(Real s) const { ElasticityTensor E(*this); E *= s; return E; }
    ElasticityTensor  operator/(Real s) const { ElasticityTensor E(*this); E /= s; return E; }

    ElasticityTensor &operator+=(const ElasticityTensor &b) { m_d += b.m_d; return *this; }
    ElasticityTensor &operator-=(const ElasticityTensor &b) { m_d -= b.m_d; return *this; }
    ElasticityTensor  operator+ (const ElasticityTensor &b) const { ElasticityTensor E(*this); E += b; return E; }
    ElasticityTensor  operator- (const ElasticityTensor &b) const { ElasticityTensor E(*this); E -= b; return E; }
    ElasticityTensor  operator- () const { ElasticityTensor E(*this); E.m_d = -E.m_d; return E; }

    // Get the tensor Einv such that E : Einv = Identity
    // Note this is different from just inverting the flattened representation:
    // F(E^-1) = S^-1 F(E)^-1 S^-1
    ElasticityTensor inverse() const {
        ElasticityTensor result;
        result.m_d = m_d.template selfadjointView<Eigen::Upper>();
        result.m_d = result.m_d.inverse().eval();
         leftApplyShearDoublerInverse(result.m_d);
        rightApplyShearDoublerInverse(result.m_d);
        return result;
    }

    template<class T>
    void leftApplyShearDoubler(T &val) const {
        // Applying on right doubles "shear rows" of a matrix or vector
        assert(val.rows() == flatLen(_Dim));
        for (size_t i = _Dim; i < flatLen(_Dim); ++i)
            for (size_t j = 0; j < (size_t) val.cols(); ++j)
                val(i, j) *= 2.0;
    }

    template<class T>
    void rightApplyShearDoubler(T &val) const {
        // Applying on left doubles "shear columns" of a matrix or row vector
        assert(val.cols() == flatLen(_Dim));
        for (size_t j = _Dim; j < flatLen(_Dim); ++j)
            for (size_t i = 0; i < (size_t) val.rows(); ++i)
                val(i, j) *= 2.0;
    }

    template<class T>
    void leftApplyShearDoublerInverse(T &val) const {
        // Applying on right halves "shear rows" of a matrix or vector
        assert(val.rows() == flatLen(_Dim));
        for (size_t i = _Dim; i < flatLen(_Dim); ++i)
            for (size_t j = 0; j < (size_t) val.cols(); ++j)
                val(i, j) *= 0.5;
    }

    template<class T>
    void rightApplyShearDoublerInverse(T &val) const {
        // Applying on left halves "shear columns" of a matrix or row vector
        assert(val.cols() == flatLen(_Dim));
        for (size_t j = _Dim; j < flatLen(_Dim); ++j)
            for (size_t i = 0; i < (size_t) val.rows(); ++i)
                val(i, j) *= 0.5;
    }

    // Doubles the off-diagonal entries of a flattened symmetric rank 2 tensor.
    FlattenedRank2Tensor shearDoubled(FlattenedRank2Tensor t) const {
        for (size_t i = _Dim; i < (size_t) t.rows(); ++i)
            t[i] *= 2.0;
        return t;
    }

    // The operation is D * S * strain, where S is the "Shear doubling" matrix
    // need to implement contraction E_ijkl e_kl
    // (see doc/meshless_fem/TensorFlattening.pdf)
    FlattenedRank2Tensor doubleContract(const FlattenedRank2Tensor &in) const {
        return m_d.template selfadjointView<Eigen::Upper>() * shearDoubled(in);
    }

    // Apply matrix D itself to a vector or a matrix. For this to have physical
    // meaning, "in" should represent a (collection of) flattened engineering
    // strains.
    template<typename FlattenedType>
    FlattenedType applyD(const FlattenedType &in) const {
        return m_d.template selfadjointView<Eigen::Upper>() * in;
    }

    template<typename Real2, size_t N, class _Storage, class _ConstRef>
    SymmetricMatrix<N, FlattenedRank2Tensor>
    doubleContract(const ConstSymmetricMatrixBase<Real2, N, _Storage, _ConstRef> &b) const {
        return SymmetricMatrix<N, FlattenedRank2Tensor>(applyD(shearDoubled(b.flattened())));
    }

    // NOTE: plain tensor double contraction is forbidden because the result
    // is asymmetric, however we do support the following operation that we
    // call "double double contract" since it obtains a symmetric result:
    //      A : B : A       (A_ijpq B_pqrs A_rskl)
    // Tensor A is "this", B is passed as an argument.
    // The operation is implemented as:
    // F(A) S F(B) S F(A)
    ElasticityTensor doubleDoubleContract(const ElasticityTensor &B) const {
        ElasticityTensor result;
        result.m_d = m_d.template selfadjointView<Eigen::Upper>();
        leftApplyShearDoubler(result.m_d);
        result.m_d = B.applyD(result.m_d);
        leftApplyShearDoubler(result.m_d);
        result.m_d = applyD(result.m_d);
        return result;
    }

    Real quadrupleContract(const ElasticityTensor &b) const {
        Real result = 0;
        for (size_t i = 0; i < _Dim; ++i)
            for (size_t j = 0; j < _Dim; ++j)
                for (size_t k = 0; k < _Dim; ++k)
                    for (size_t l = 0; l < _Dim; ++l)
                        result += (*this)(i, j, k, l) * b(i, j, k, l);
        return result;
    }

    // Applies an change of coordinates to this tensor using the
    // tensor transformation rule:
    // E_ijlk' = E_pqrs R_ip R_jq R_kr R_ls
    // (When R is a rotation, this is the correct transformation rule for
    //  cartesian tensors).
    ElasticityTensor transform(const Eigen::Matrix<Real, _Dim, _Dim> &R) const {
        ElasticityTensor result;
        for (size_t i = 0; i < _Dim; ++i) {
            for (size_t j = 0; j < _Dim; ++j) {
                for (size_t k = 0; k < _Dim; ++k) {
                    for (size_t l = 0; l < _Dim; ++l) {
                        Real comp = 0;
                        for (size_t p = 0; p < _Dim; ++p)
                            for (size_t q = 0; q < _Dim; ++q)
                                for (size_t r = 0; r < _Dim; ++r)
                                    for (size_t s = 0; s < _Dim; ++s)
                                        comp += (*this)(p, q, r, s) * R(i, p) * R(j, q) * R(k, r) * R(l, s);
                        Real existing = result(i, j, k, l);
                        assert((existing == 0) || (std::abs(existing - comp) < 1e-10));

                        size_t ij = flattenIndices(_Dim, i, j);
                        size_t kl = flattenIndices(_Dim, k, l);
                        result.m_d(ij, kl) = comp;
                    }
                }
            }
        }
        return result;
    }

private:
    DType m_d;

    friend std::ostream &operator<<(std::ostream &os, const ElasticityTensor &E) {
        DType d = E.m_d.template selfadjointView<Eigen::Upper>();
        os << d;
        return os;
    }
};

template<typename Real, int _Dim>
ElasticityTensor<Real, _Dim> operator*(Real a,
        const ElasticityTensor<Real, _Dim> &E) {
    return E * a;
}

#endif /* end of include guard: ELASTICITYTENSOR_HH */
