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
//      When _MajorSymmetry == true, symmetry is enforced by only storing the
//      upper triangle of D internally. This means matrix element accesses must
//      be done through method D(i, j), and matrix operations need to be
//      performed with Eigen's "selfadjointView<Eigen::Upper>" view. For safety,
//      because of this complexity, m_d is kept entirely private, with no direct
//      accessor.
//
//      When _MajorSymmetry == false, major symmetry is not enforced, and the
//      datastructure can be used to store non-major-symmetric tensors (though
//      these shouldn't be used as true elasticity tensors). This can be useful
//      to store intermediate computation results on rank 4 tensors: e.g. the
//      double contraction of two elasticity tensors.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/11/2014 15:14:13
////////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICITYTENSOR_HH
#define ELASTICITYTENSOR_HH

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>

#include <tuple>
#include "Flattening.hh"
#include "SymmetricMatrix.hh"

#include "Algebra.hh"

// Eigenvalues/vectors are sorted in order of increasing eigenvalue.
struct ETensorEigenDecomposition {
    Eigen::MatrixXd strains;
    Eigen::VectorXd lambdas;
};

template<typename Real, size_t _Dim, bool _MajorSymmetry = true>
class ElasticityTensor : public VectorSpace<Real, ElasticityTensor<Real, _Dim, _MajorSymmetry>> {
    // We need access to other major symmetry types' members (for double
    // contraction operator), but unfortunately we can't friend a partial
    // template specialization, so...
    template<typename _Real2, size_t _Dim2, bool _MajorSymmetry2>
    friend class ElasticityTensor;
public:
    typedef Eigen::Matrix<Real, flatLen(_Dim), flatLen(_Dim)> DType;
    typedef typename DType::RowXpr                            RowXpr;
    typedef typename DType::ConstRowXpr                       ConstRowXpr;
    typedef typename DType::ColXpr                            ColXpr;
    typedef typename DType::ConstColXpr                       ConstColXpr;
    // Wraps a row of the flattened elasticity tensor with a symmetric matrix
    // interface--useful for periodic homogenization formulas where rows of the
    // flattened homogenized elasticity tensor are modulated by flattened
    // fluctuation stresses.
    typedef      SymmetricMatrixRef<_Dim, RowXpr, ConstRowXpr>      SMRowWrapper;
    typedef ConstSymmetricMatrixRef<_Dim,         ConstRowXpr> ConstSMRowWrapper;
    typedef      SymmetricMatrixRef<_Dim, ColXpr, ConstColXpr>      SMColWrapper;
    typedef ConstSymmetricMatrixRef<_Dim,         ConstColXpr> ConstSMColWrapper;

    typedef Eigen::Matrix<Real, flatLen(_Dim), 1>       FlattenedRank2Tensor;
    typedef SymmetricMatrix<_Dim, FlattenedRank2Tensor> SMatrix;

    static const bool MajorSymmetry = _MajorSymmetry;
    static const size_t Dim = _Dim;

    ElasticityTensor() : m_d(DType::Zero()) { }
    // Construct the elasticity tensor with a Young's modulus and Poisson ratio
    ElasticityTensor(Real E, Real nu) { setIsotropic(E, nu); }

    // Allow converting constructor to and from major symmetric tensors.
    // Converting to a major symmetric tensor verifies that the input is indeed
    // major symmetric.
    template<bool _MajorSymmetry2>
    ElasticityTensor(const ElasticityTensor<Real, _Dim, _MajorSymmetry2> &b) {
        if ( _MajorSymmetry == _MajorSymmetry2) m_d = b.m_d;
        if (!_MajorSymmetry && _MajorSymmetry2) {
            m_d = b.m_d.template selfadjointView<Eigen::Upper>();
        }
        if ( _MajorSymmetry && !_MajorSymmetry2) {
            if (!b.hasMajorSymmetry())
                throw std::runtime_error("Attempting to copy construct major-symmetric tensor from non-major symmetric tensor");
            m_d = b.m_d.template selfadjointView<Eigen::Upper>();
        }
    }

    // The symmetric rank 4 identity tensor corresponds to lambda = 0, mu = 1/2
    void setIdentity() { setIsotropicLame(0, 0.5); }

    // Configure the elasticity tensor with a Young's modulus and Poisson ratio
    void setIsotropic(Real E, Real nu) {
        // Lame formula:
        // stress = lamda * trace(strain) + mu (strain + strain^T)
        // (We write it this way so that the implied elasticity tensor has the
        //  correct symmetries.)
        Real lambda = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 + 2.0 * nu);

        // For 2D (plane stress), lambda is actually different...
        // This can be found by inverting the 2D orthotropic compliance
        // tensor with Ex = Ey = E.
        if (_Dim == 2)
            lambda = (nu * E) / (1.0 - nu * nu);

        setIsotropicLame(lambda, mu);
    }

    // Configure the elasticity tensor with Lame parameters
    void setIsotropicLame(Real lambda, Real mu) {
        m_d =  DType::Zero();
        if (_Dim == 3) {
            m_d(0, 0) = lambda + 2 * mu; m_d(0, 1) = lambda;          m_d(0, 2) = lambda;
                                         m_d(1, 1) = lambda + 2 * mu; m_d(1, 2) = lambda;
                                                                      m_d(2, 2) = lambda + 2 * mu;
            m_d(3, 3) = m_d(4, 4) = m_d(5, 5) = mu;
        }
        else {
            assert(_Dim == 2);
            m_d(0, 0) = lambda + 2 * mu; m_d(0, 1) = lambda;
                                         m_d(1, 1) = lambda + 2 * mu;
            m_d(2, 2) = mu;
        }

        m_d = m_d.template selfadjointView<Eigen::Upper>();
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

    // Get the orthotropic material parameters (assuming the material is
    // in fact orthotropic).
    void getOrthotropicParameters(std::vector<Real> &moduli) const {
        if (_Dim == 2) {
            moduli.resize(4);
            // Ex Ey nuYX nuXY
            getOrthotropic2D(moduli[0], moduli[1], moduli[2], moduli[3]);
        }
        else if (_Dim == 3) {
            moduli.resize(9);
            // Ex Ey Ez nuYX nuZX nuZY muYZ muZX muXY
            getOrthotropic3D(moduli[0], moduli[1], moduli[2],
                             moduli[3], moduli[4], moduli[5],
                             moduli[6], moduli[7], moduli[8]);
        }
        else {
            throw std::runtime_error("Invalid instance dimension.");
        }
    }

    std::vector<Real> getOrthotropicParameters() const {
        std::vector<Real> moduli;
        getOrthotropicParameters(moduli);
        return moduli;
    }

    // Get the orthotropic material parameters (assuming the material is in fact
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

    // Get the orthotropic material parameters (assuming the material is in fact
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

	void getUpperRight2D(std::vector<Real> & components) const {
        if (_Dim != 2)
            throw std::runtime_error("getOrthotropic2D call on non-2D tensor");
        for (size_t i = 0; i < flatLen(_Dim); ++i)
        	for (size_t j = i; j < flatLen(_Dim); ++j)
        		components.push_back(m_d(i, j));
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
        return D(flattenIndices<_Dim>(i, j), flattenIndices<_Dim>(k, l));
    }

    Real D(size_t i, size_t j) const {
        assert((i < (size_t) m_d.rows()) && (j < (size_t) m_d.cols()));
        if (_MajorSymmetry) return (i <= j) ? m_d(i, j) : m_d(j, i);
        else                return m_d(i, j);
    }

    Real &D(size_t i, size_t j) {
        assert((i < (size_t) m_d.rows()) && (j < (size_t) m_d.cols()));
        if (_MajorSymmetry) return (i <= j) ? m_d(i, j) : m_d(j, i);
        else                return m_d(i, j);
    }

    ConstRowXpr DRow(size_t i) const { assert(i < (size_t) m_d.rows()); return m_d.row(i); }
    RowXpr      DRow(size_t i)       { assert(i < (size_t) m_d.rows()); return m_d.row(i); }
    ConstSMRowWrapper DRowAsSymMatrix(size_t i) const { return ConstSMRowWrapper(DRow(i)); }
         SMRowWrapper DRowAsSymMatrix(size_t i)       { return      SMRowWrapper(DRow(i)); }

    ConstColXpr DCol(size_t i) const { assert(i < (size_t) m_d.cols()); return m_d.col(i); }
    ColXpr      DCol(size_t i)       { assert(i < (size_t) m_d.cols()); return m_d.col(i); }
    ConstSMColWrapper DColAsSymMatrix(size_t i) const { return ConstSMColWrapper(DCol(i)); }
         SMColWrapper DColAsSymMatrix(size_t i)       { return      SMColWrapper(DCol(i)); }

    // Get the flattened tensor's diagonal
    Eigen::Matrix<Real, flatLen(_Dim), 1> diag() const {
        return m_d.diagonal();
    }

    ////////////////////////////////////////////////////////////////////////////
    // VectorSpace requirements
    ////////////////////////////////////////////////////////////////////////////
    void Add(const ElasticityTensor &b) { m_d += b.m_d; }
    void Scale(Real s)                  { m_d *= s; }

    // Get the tensor Einv such that E : Einv = Identity
    // Note this is different from just inverting the flattened representation:
    // F(E^-1) = S^-1 F(E)^-1 S^-1
    ElasticityTensor inverse() const {
        ElasticityTensor result;
        if (_MajorSymmetry) result.m_d = m_d.template selfadjointView<Eigen::Upper>();
        else                result.m_d = m_d;
        result.m_d = result.m_d.inverse().eval();
         leftApplyShearDoublerInverse(result.m_d);
        rightApplyShearDoublerInverse(result.m_d);
        return result;
    }

    // For rank deficient elasticity tensors, get the tensor Epinv such that
    // E : (Epinv : E) = E.
    ElasticityTensor pseudoinverse() const {
        auto eigs = computeEigenstrains();
        // note: computeEigenstrains() performs the decomposition
        // D^1/2 F(E) D^1/2 = Q Lambda Q^T
        // And actually returns D^(-1/2) Q as the "eigenstrains".
        // We need to compute
        // F(Epinv) = [D^(-1/2) Q] Lambda_pinv [D^(-1/2) Q]^T
        for (int i = 0; i < eigs.lambdas.size(); ++i) {
            // TODO: think more about threshold here
            if (std::abs(eigs.lambdas[i]) > 1e-8)
                eigs.lambdas[i] = 1.0 / eigs.lambdas[i];
            else eigs.lambdas[i] = 0;
        }

        ElasticityTensor result;
        result.m_d = eigs.strains * (eigs.lambdas.asDiagonal() * eigs.strains.transpose());
        return result;
    }

    // Get the ***major*** transposed tensor E^T:
    // E^T_ijkl = E_klij
    // For major-symmetric tensors, this is an identity operation--returns copy.
    ElasticityTensor transpose() const {
        if (_MajorSymmetry) return *this;
        ElasticityTensor result;
        result.m_d = m_d.transpose();
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // In-place application of shear doubling matrices (both left and right)
    ////////////////////////////////////////////////////////////////////////////
    template<class T>
    void leftApplyShearDoubler(T &val) const {
        // Applying on right doubles "shear rows" of a matrix or vector
        assert(size_t(val.rows()) == flatLen(_Dim));
        for (size_t i = _Dim; i < flatLen(_Dim); ++i)
            for (size_t j = 0; j < (size_t) val.cols(); ++j)
                val(i, j) *= 2.0;
    }

    template<class T>
    void rightApplyShearDoubler(T &val) const {
        // Applying on left doubles "shear columns" of a matrix or row vector
        assert(size_t(val.cols()) == flatLen(_Dim));
        for (size_t j = _Dim; j < flatLen(_Dim); ++j)
            for (size_t i = 0; i < (size_t) val.rows(); ++i)
                val(i, j) *= 2.0;
    }

    template<class T>
    void leftApplyShearDoublerInverse(T &val) const {
        // Applying on right halves "shear rows" of a matrix or vector
        assert(size_t(val.rows()) == flatLen(_Dim));
        for (size_t i = _Dim; i < flatLen(_Dim); ++i)
            for (size_t j = 0; j < (size_t) val.cols(); ++j)
                val(i, j) *= 0.5;
    }

    template<class T>
    void rightApplyShearDoublerInverse(T &val) const {
        // Applying on left halves "shear columns" of a matrix or row vector
        assert(size_t(val.cols()) == flatLen(_Dim));
        for (size_t j = _Dim; j < flatLen(_Dim); ++j)
            for (size_t i = 0; i < (size_t) val.rows(); ++i)
                val(i, j) *= 0.5;
    }

    ////////////////////////////////////////////////////////////////////////////
    // In place application of *square root* of shear doubling matrices
    ////////////////////////////////////////////////////////////////////////////
    template<class T>
    void leftApplySqrtShearDoubler(T &val) const {
        // Applying on right doubles "shear rows" of a matrix or vector
        assert(size_t(val.rows()) == flatLen(_Dim));
        for (size_t i = _Dim; i < flatLen(_Dim); ++i)
            for (size_t j = 0; j < (size_t) val.cols(); ++j)
                val(i, j) *= sqrt(2.0);
    }

    template<class T>
    void rightApplySqrtShearDoubler(T &val) const {
        // Applying on left doubles "shear columns" of a matrix or row vector
        assert(size_t(val.cols()) == flatLen(_Dim));
        for (size_t j = _Dim; j < flatLen(_Dim); ++j)
            for (size_t i = 0; i < (size_t) val.rows(); ++i)
                val(i, j) *= sqrt(2.0);
    }

    template<class T>
    void leftApplySqrtShearDoublerInverse(T &val) const {
        // Applying on right halves "shear rows" of a matrix or vector
        assert(size_t(val.rows()) == flatLen(_Dim));
        for (size_t i = _Dim; i < flatLen(_Dim); ++i)
            for (size_t j = 0; j < (size_t) val.cols(); ++j)
                val(i, j) *= sqrt(0.5);
    }

    template<class T>
    void rightApplySqrtShearDoublerInverse(T &val) const {
        // Applying on left halves "shear columns" of a matrix or row vector
        assert(size_t(val.cols()) == flatLen(_Dim));
        for (size_t j = _Dim; j < flatLen(_Dim); ++j)
            for (size_t i = 0; i < (size_t) val.rows(); ++i)
                val(i, j) *= sqrt(0.5);
    }

    // Doubles the off-diagonal entries of a flattened symmetric rank 2 tensor.
    FlattenedRank2Tensor shearDoubled(FlattenedRank2Tensor t) const {
        for (size_t i = _Dim; i < (size_t) t.rows(); ++i)
            t[i] *= 2.0;
        return t;
    }

    // The operation is D * S * strain, where S is the "Shear doubling" matrix
    // needed to implement contraction E_ijkl e_kl
    // (see doc/meshless_fem/TensorFlattening.pdf)
    FlattenedRank2Tensor doubleContract(const FlattenedRank2Tensor &in) const {
        if (_MajorSymmetry) return m_d.template selfadjointView<Eigen::Upper>() * shearDoubled(in);
        else                return m_d * shearDoubled(in);
    }

    // Apply matrix D itself to a vector or a matrix. For this to have physical
    // meaning, "in" should represent a (collection of) flattened engineering
    // strains.
    template<typename FlattenedType>
    FlattenedType applyD(const FlattenedType &in) const {
        if (_MajorSymmetry) return m_d.template selfadjointView<Eigen::Upper>() * in;
        else                return m_d * in;
    }

    template<typename Real2, size_t N, class _Storage, class _ConstRef>
    SymmetricMatrix<N, FlattenedRank2Tensor>
    doubleContract(const ConstSymmetricMatrixBase<Real2, N, _Storage, _ConstRef> &b) const {
        return SymmetricMatrix<N, FlattenedRank2Tensor>(applyD(shearDoubled(b.flattened())));
    }

    // The following operation that we call "double double contract" maintains
    // major symmetry (result is major-symmetric if **both** A and B are):
    //      A : B : A       (A_ijpq B_pqrs A_rskl)
    // Tensor A is "this", B is passed as an argument.
    // The operation is implemented as:
    // F(A) S F(B) S F(A)
    template<bool _MajorSymmetry2>
    ElasticityTensor<Real, _Dim, _MajorSymmetry && _MajorSymmetry2>
    doubleDoubleContract(const ElasticityTensor<Real, _Dim, _MajorSymmetry2> &B) const {
        ElasticityTensor<Real, _Dim, _MajorSymmetry && _MajorSymmetry2> result;
        if (_MajorSymmetry) result.m_d = m_d.template selfadjointView<Eigen::Upper>();
        else                result.m_d = m_d;
        leftApplyShearDoubler(result.m_d);
        result.m_d = B.applyD(result.m_d);
        leftApplyShearDoubler(result.m_d);
        result.m_d = applyD(result.m_d);
        return result;
    }

    // NOTE: In general, double contraction:
    //      A : B       (A_ijpq B_pqkl)
    // gives a tensor without major symmetry. We implement this operation as:
    // F(A : B) = F(A) S F(B)
    template<bool _MajorSymmetry2>
    ElasticityTensor<Real, _Dim, false> doubleContract(const ElasticityTensor<Real, _Dim, _MajorSymmetry2> &B) const {
        ElasticityTensor<Real, _Dim, false> result;
        if (_MajorSymmetry2) result.m_d = B.m_d.template selfadjointView<Eigen::Upper>();
        else                 result.m_d = B.m_d;
        leftApplyShearDoubler(result.m_d);
        result.m_d = applyD(result.m_d);
        return result;
    }

    template<bool _MajorSymmetry2>
    Real quadrupleContract(const ElasticityTensor<Real, _Dim, _MajorSymmetry2> &b) const {
        Real result = 0;
        for (size_t i = 0; i < _Dim; ++i)
            for (size_t j = 0; j < _Dim; ++j)
                for (size_t k = 0; k < _Dim; ++k)
                    for (size_t l = 0; l < _Dim; ++l)
                        result += (*this)(i, j, k, l) * b(i, j, k, l);
        return result;
    }

    Real frobeniusNormSq() const { return this->quadrupleContract(*this); }

    // Applies an change of coordinates to this tensor using the
    // tensor transformation rule:
    // E_ijlk' = E_pqrs R_ip R_jq R_kr R_ls
    // (When R is a rotation or reflection, this is the correct transformation
    // rule for cartesian tensors).
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
                        if (_MajorSymmetry) {
                            // Validate major symmetry (when we should have it).
                            Real existing = result(i, j, k, l);
                            assert((existing == 0) || (std::abs(existing - comp) < 1e-10));
                        }

                        size_t ij = flattenIndices<_Dim>(i, j);
                        size_t kl = flattenIndices<_Dim>(k, l);
                        result.m_d(ij, kl) = comp;
                    }
                }
            }
        }
        return result;
    }

    // If major symmetry isn't enforced, check whether it exists.
    bool hasMajorSymmetry() const {
        if (_MajorSymmetry) return true;
        Real absDiff = (m_d - DType(m_d.template selfadjointView<Eigen::Upper>())).norm();
        // Permit small relative or absolute deviations.
        return (absDiff < 1e-10) || (absDiff < 1e-10 * m_d.norm());
    }

    // Compute all the eigenstrains and corresponding eigenvalues. In other words,
    // find the (s, lambda) pairs satisfying:
    //      E : s = lambda s
    // Should only be used on major-symmetric tensors.
    ETensorEigenDecomposition computeEigenstrains() const {
        assert(hasMajorSymmetry());

        // We are solving the problem:
        //    lambda = max_||s||^2_F=1   s : E : s
        // In flattened form, this is:
        //    lambda = max_(F(s)^T D F(s) = 1) F(s)^T D F(E) D F(s)
        // Where D is the shear-doubling matrix. We could solve this as a
        // generalized eigenvalue problem, but instead we transform:
        //    e = D^(1/2) F(s) ==>
        //    lambda = max_(||e||^2 = 1) e^T D^(1/2) F(E) D^(1/2) e
        // i.e. an ordinary eigenvalue problem for D^(1/2) F(E) D^(1/2).
        // We then retrieve eigenstrain F(s) = D^(-1/2) e.
        //
        DType mat(m_d.template selfadjointView<Eigen::Upper>());
        leftApplySqrtShearDoubler(mat);
        rightApplySqrtShearDoubler(mat);
        Eigen::SelfAdjointEigenSolver<DType> solver;
        solver.compute(mat);

        Eigen::MatrixXd Q = solver.eigenvectors();
        Eigen::VectorXd Lambda = solver.eigenvalues();
        leftApplySqrtShearDoublerInverse(Q);
        return ETensorEigenDecomposition{Q, Lambda};
    }

    // Computes the eigenstrains with maximum eigenvalue (and this eigenvalue).
    // In otherwords, we find the (s, lambda) satisfying:
    //     E : s = lambda s
    // for greatest lambda.
    // Should only be used on major-symmetric tensors.
    std::tuple<SMatrix, Real> maxEigenstrain() const {
        auto eigs = computeEigenstrains();
        // Eigenvalues sorted in increasing order
        constexpr size_t largestIdx = flatLen(_Dim) - 1;
        return std::make_tuple(SMatrix(eigs.strains.col(largestIdx)), eigs.lambdas[largestIdx]);
    }

    // Same as above, but also approximate the algebraic multiplicity of the
    // maximum eigenvalue (within the specified tolerance) and also return the
    // second largest eigenvalue.
    std::tuple<SMatrix, Real, int, Real> maxEigenstrainMultiplicity(Real tol = 1e-3) const {
        auto eigs = computeEigenstrains();

        // Eigenvalues sorted in increasing order
        constexpr size_t largestIdx = flatLen(_Dim) - 1;
        Real lambda = eigs.lambdas[largestIdx];

        // Approximate multiplicity
        int multiplicity = 1;
        for (size_t i = largestIdx; i > 0; --i) {
            if ((lambda - eigs.lambdas[i - 1]) < lambda * tol) ++multiplicity;
            else break;
        }
        return std::make_tuple(SMatrix(eigs.strains.col(largestIdx)), lambda, multiplicity,
                               eigs.lambdas[largestIdx - 1]);
    }

    // Write unflattened tensor in Mathematica array syntax.
    void writeUnflattened(std::ostream &os) const {
        os << "{";
        for (size_t i = 0; i < _Dim; ++i) {
            os << "{";
            for (size_t j = 0; j < _Dim; ++j) {
                os << "{";
                for (size_t k = 0; k < _Dim; ++k) {
                    os << "{";
                    for (size_t l = 0; l < _Dim; ++l) {
                        os << (*this)(i, j, k, l);
                        if (l < _Dim - 1) os << ", ";
                    }
                    os << ((k < _Dim - 1) ? "}, " : "}");
                }
                os << ((j < _Dim - 1) ? "}, " : "}");
            }
            os << ((i < _Dim - 1) ? "}, " : "}");
        }
        os << "}";
    }

    // For serialization into a json file
    std::vector<typename DType::Scalar> getCoefficients() const {
        std::vector<typename DType::Scalar> coeffs;
        if (_MajorSymmetry) {
            auto M = (DType) m_d.template selfadjointView<Eigen::Upper>();
            for (int i = 0; i < M.rows(); ++i) {
                for (int j = 0; j < M.cols(); ++j) {
                    coeffs.push_back(M(i, j));
                }
            }
        } else {
            auto M = m_d;
            for (int i = 0; i < M.rows(); ++i) {
                for (int j = 0; j < M.cols(); ++j) {
                    coeffs.push_back(M(i, j));
                }
            }
        }
        return coeffs;
    }

    // For debug purposes only
    void writeD(std::ostream &os) const {
        os << m_d;
    }

    void readD(std::istream &is) const {
        is >> m_d;
    }

    void setD(const DType &d) {
        if (_MajorSymmetry) {
            Real diffSq = (d.transpose() - d).squaredNorm();
            if ((diffSq > 1e-9) && (diffSq > 1e-9 * d.squaredNorm())) {
                std::cout << d << std::endl;
                throw std::runtime_error("Attempted to assign a non-major-symmetric value to a major-symmetric ElasticityTensor.");
            }
        }
        m_d = d;
    }

private:
    DType m_d;

    friend std::ostream &operator<<(std::ostream &os, const ElasticityTensor &E) {
        if (_MajorSymmetry) os << (DType) E.m_d.template selfadjointView<Eigen::Upper>();
        else                os << E.m_d;
        return os;
    }
};

#include "LinearIndexer.hh"

// Index the distinct components (after accounting for symmetries).
// Only major symmetric tensors are currently supported
template<typename Real, size_t N>
struct LinearIndexerImpl<ElasticityTensor<Real, N, true>> {
    using tensor_type = ElasticityTensor<Real, N, true>;
    using scalar_type = Real;

    static Real &index(      tensor_type &val, size_t i) { size_t ij, kl; linearIndexTo2D(i, ij, kl); return val.D(ij, kl); }
    static Real  index(const tensor_type &val, size_t i) { size_t ij, kl; linearIndexTo2D(i, ij, kl); return val.D(ij, kl); }
    static constexpr size_t size() { return (flatLen(N) * (flatLen(N) + 1)) / 2; }

    // Compute the flattened tensor row and column index corresponding to a
    // 1D index. This is the inverse of the following map:
    //    r(ij, kl) = kl + flatlen * ij - (ij * (ij + 1)) / 2
    // Not a closed form inverse, but likely faster than sqrt version anyway
    static void linearIndexTo2D(size_t idx, size_t &ij, size_t &kl) {
        assert(idx < size());
        kl = flatLen(N) + 1; // invalid
        for (ij = 0; ij < flatLen(N); ++ij) {
            size_t rowSize = flatLen(N) - ij;
            if (idx < rowSize) { kl = ij + idx; break;}
            idx -= rowSize;
        }
        assert((ij < flatLen(N)) && (kl < flatLen(N)));
    }
};

#endif /* end of include guard: ELASTICITYTENSOR_HH */
