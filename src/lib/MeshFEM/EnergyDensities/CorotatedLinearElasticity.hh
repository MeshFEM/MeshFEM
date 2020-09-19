////////////////////////////////////////////////////////////////////////////////
// CorotatedLinearElasticity.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A general linear elastic material model with geometric nonlinearities for
//  small strain, large deformation applications.
//
//  We use a (right) polar decomposition F = RS to remove the rigid rotation
//  component R from the deformation gradient F. The resulting large deformation
//  strain is S - I, also known as the Biot strain.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/19/2020 12:15:32
////////////////////////////////////////////////////////////////////////////////
#ifndef COROTATEDLINEARELASTICITY_HH
#define COROTATEDLINEARELASTICITY_HH

#include <Eigen/Dense>
#include "EnergyTraits.hh"
#include "Tensor.hh"

// Dimension-specific calculations
template<typename _Real, size_t N>
struct CRQuantities;

template<typename _Real>
struct CRQuantities<_Real, 3> {
    using GType    = Eigen::Matrix<_Real, 3, 3>;
    using IRotType = Eigen::Matrix<_Real, 3, 1>; // infinitesimal rotation representation
    using Mat      = Eigen::Matrix<_Real, 3, 3>;

    template<typename Derived>
    static GType getG(const Eigen::MatrixBase<Derived> &S) { return S.trace() * GType::Identity() - S; }

    template<typename Derived>
    static GType getGinv(const Eigen::MatrixBase<Derived> &G) { return G.inverse(); }

    // Extract a vector representing the skew symmetric part 0.5 (A - A^T)
    //     0 -c  b      [a]
    //     c  0 -a  ==> [b]
    //    -b  a  0      [c]
    // (This is the vector `w` whose cross product `w x v` equals `0.5 (A - A^T) v`.)
    template<typename Derived>
    static IRotType sk_inv(const Eigen::MatrixBase<Derived> &A) {
        return IRotType(0.5 * (A(2, 1) - A(1, 2)),
                        0.5 * (A(0, 2) - A(2, 0)),
                        0.5 * (A(1, 0) - A(0, 1)));
    }

    // B * sk(w)
    template<typename Derived>
    static Mat right_mul_sk(const Eigen::MatrixBase<Derived> &B, const IRotType &w) {
        return B.rowwise().cross(w);
    }
};

template<typename _Real>
struct CRQuantities<_Real, 2> {
    using GType    = _Real;
    using IRotType = _Real;
    using Mat      = Eigen::Matrix<_Real, 2, 2>;

    template<typename Derived>
    static GType getG(const Eigen::MatrixBase<Derived> &S) { return S.trace(); }

    static GType getGinv(_Real G) { return 1.0 / G; }

    // Extract a scalar representing the skew symmetric part 0.5 (A - A^T)
    //   0 -a
    //   a  0
    // (This scalar represents the counterclockwise infinitesimal rotation
    //  applied by `0.5 (A - A^T)`
    template<typename Derived>
    static IRotType sk_inv(const Eigen::MatrixBase<Derived> &A) {
        return 0.5 * (A(1, 0) - A(0, 1));
    }

    // B * sk(w)
    template<typename Derived>
    static Mat right_mul_sk(const Eigen::MatrixBase<Derived> &B, const IRotType &w) {
        Mat result;
        result << w * B.col(1),
                 -w * B.col(0);
        return result;
    }
};

template <typename _Real, size_t _Dimension>
struct CorotatedLinearElasticity : public Concepts::CRLinearElaticEnergy {
    using SMatrix = SymmetricMatrixValue<_Real, _Dimension>;
    static constexpr EDensityType EDType = EDensityType::FBased;

    static constexpr size_t Dimension = _Dimension;
    static constexpr size_t N         = Dimension;
    using Real = _Real;
    using Matrix  = Eigen::Matrix<_Real, _Dimension, _Dimension>;
    using ETensor = ElasticityTensor<_Real, _Dimension>;
    using CRQ     = CRQuantities<_Real, _Dimension>;

    // We can use simplified formulas if we know `elasticity_tensor` is isotropic. This is
    // specified with the `isotropic` argument.
    CorotatedLinearElasticity(const ETensor& elasticity_tensor, bool isotropic = false) :
        m_elasticity_tensor(elasticity_tensor), m_isotropic(isotropic) {
        setDeformationGradient(Matrix::Identity());
    }

    // Constructor copying material properties only, not the current deformation
    CorotatedLinearElasticity(const CorotatedLinearElasticity &other, UninitializedDeformationTag &&)
        : m_elasticity_tensor(other.m_elasticity_tensor), m_isotropic(other.m_isotropic) { }

    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        m_F = F;
        Eigen::JacobiSVD<Matrix> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV );
        m_R = svd.matrixU() * svd.matrixV().transpose();
        if (m_R.determinant() < 0) {
            Matrix W = svd.matrixV();
            W.col(svd.matrixV().cols() - 1) *= -1;
            m_R = svd.matrixU() * W.transpose();
        }
        m_S = m_R.transpose() * F;

        // Analog to infinitesimal strain for linear elasticity.
        m_biotStrain = m_S - Matrix::Identity();

        // Analog to Cauchy stress for linear elasticity.
        m_biotStress = m_elasticity_tensor.doubleContract(SMatrix(m_biotStrain)).toMatrix();

        if (elevel == EvalLevel::EnergyOnly) return;

        m_G    = CRQ::getG(m_S);
        m_Ginv = CRQ::getGinv(m_G);
        if (!m_isotropic) {
            // 2 sk(G^{-1} sk^{-1}(sigma * S)) := sk(g)
            m_g = 2 * m_Ginv * CRQ::sk_inv(m_biotStress * m_S);
            m_pk1_stress = m_R * m_biotStress - CRQ::right_mul_sk(m_R, m_g);
        }
        else {
            // If the elasticity tensor is isotropic, then sigma and S commute
            // (Biot stress and stretch factor share eigenvectors) and all
            // "rotational stress" terms involving sk(g) vanish.
            m_pk1_stress = m_R * m_biotStress;
        }
    }

    const Matrix &getDeformationGradient() const { return m_F; }

    _Real energy() const {
        return 0.5 * doubleContract(m_biotStress, m_biotStrain);
    }

    // PK1 stress
    _Real denergy(const Matrix& dF) const { return doubleContract(denergy(), dF); }

    // Asymmetric!
    Matrix denergy() const {
        return m_pk1_stress;
    }

    // Symmetric!
    Matrix PK2Stress() const { return m_F.inverse() * denergy(); }

    const Matrix &R() const { return m_R; }
    const Matrix &S() const { return m_S; }
    const Matrix &biotStress() const { return m_biotStress; }

    template<class Mat_>
    Matrix delta_R(const Mat_ &dF) const {
        typename CRQ::IRotType w = 2 * m_Ginv * CRQ::sk_inv(m_R.transpose() * dF);
        return CRQ::right_mul_sk(m_R, w);
    }

    template<class Mat_>
    Matrix delta_S(const Mat_ &dF, const Matrix &dR) const {
        return dR.transpose() * m_F + m_R.transpose() * dF;
    }

    Matrix delta_sigma(const Matrix &dS) const {
        return m_elasticity_tensor.doubleContract(SMatrix(dS)).toMatrix();
    }

    template<class Mat_>
    Matrix delta_denergy(const Mat_ &dF) const {
        Matrix dR     = delta_R(dF);
        Matrix dS     = delta_S(dF, dR);
        Matrix dsigma = delta_sigma(dS);

        Matrix result = dR * m_biotStress + m_R * dsigma;

        if (!m_isotropic) {
            typename CRQ::IRotType dg = m_Ginv * (2 * CRQ::sk_inv(dsigma * m_S + m_biotStress * dS) - CRQ::getG(dS) * m_g);
            result -= CRQ::right_mul_sk(dR, m_g) + CRQ::right_mul_sk(m_R, dg);
        }
        return result;
    }

    _Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        return doubleContract(delta_denergy(dF_lhs), dF_rhs);
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &dF_a, const Mat2_ &dF_b) const {
        typename CRQ::IRotType w_a = 2 * m_Ginv * CRQ::sk_inv(m_R.transpose() * dF_a);
        Matrix dR_a = CRQ::right_mul_sk(m_R, w_a),
               dR_b = delta_R(dF_b);
        Matrix dS_a = delta_S(dF_a, dR_a),
               dS_b = delta_S(dF_b, dR_b);
        Matrix dsigma_a = delta_sigma(dS_a);
        Matrix dsigma_b = delta_sigma(dS_b);

        Matrix d2R, d2S, d2sigma;
        d2R = CRQ::right_mul_sk(dR_b, w_a)
            + CRQ::right_mul_sk(m_R, m_Ginv * (2 * CRQ::sk_inv(dR_b.transpose() * dF_a) - CRQ::getG(dS_b) * w_a));

        d2S = d2R.transpose() * m_F + dR_a.transpose() * dF_b + dR_b.transpose() * dF_a;
        d2sigma = delta_sigma(d2S);

        Matrix result = d2R * m_biotStress + dR_a * dsigma_b + dR_b * dsigma_a + m_R * d2sigma;

        if (!m_isotropic) {
            typename CRQ::IRotType dg_a = m_Ginv * (2 * CRQ::sk_inv(dsigma_a * m_S + m_biotStress * dS_a) - CRQ::getG(dS_a) * m_g),
                                   dg_b = m_Ginv * (2 * CRQ::sk_inv(dsigma_b * m_S + m_biotStress * dS_b) - CRQ::getG(dS_b) * m_g),
                                   d2g  = m_Ginv * (2 * CRQ::sk_inv(d2sigma  * m_S + dsigma_a * dS_b + dsigma_b * dS_a + m_biotStress * d2S) - CRQ::getG(d2S) * m_g - CRQ::getG(dS_a) * dg_b - CRQ::getG(dS_b) * dg_a);

            result -= CRQ::right_mul_sk(d2R, m_g) + CRQ::right_mul_sk(dR_a, dg_b) + CRQ::right_mul_sk(dR_b, dg_a)
                    + CRQ::right_mul_sk(m_R, d2g);
        }
        return result;
    }

    bool isIsotropic() const { return m_isotropic; }

private:
    ETensor m_elasticity_tensor;
    Matrix m_F,
           m_R, m_S, // Polar decomposition
           m_biotStrain, m_biotStress, m_pk1_stress,
           m_anisotropicRotationalStress; // Skew symmetric contribution to the PK1 stress; only nonzero for anisotropic material model.
    typename CRQ::GType m_G, m_Ginv;
    typename CRQ::IRotType m_g;
    bool m_isotropic;
};

#endif /* end of include guard: COROTATEDLINEARELASTICITY_HH */
