////////////////////////////////////////////////////////////////////////////////
// SymmetricDirichlet.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  SymmetricDirichlet(SD) energy for 3D triangular mesh parameterization 
//  D = (1/2)*(J_f : J_f + J_f^(-1) : J_f^(-1))
//  (from (x,y,z) --> (u,v)) --> original case. But here we generalize to various mapping dimensions
*/
//  Author:  Xinzhuo (Johnson) Hu
//  Created:  06/29/2023 17:07 PM
////////////////////////////////////////////////////////////////////////////////

#ifndef SYMMETRICDIRICHLET_HH
#define SYMMETRICDIRICHLET_HH

#include <Eigen/Dense>
#include "Tensor.hh"

template<typename _Real, size_t _Dim>
struct SymmetricDirichlet
{
    static constexpr size_t Dimension = _Dim;
    static constexpr size_t N         = _Dim;
    static constexpr EDensityType EDType = EDensityType::FBased;

    static constexpr const char *name() {return "SymmetricDirichlet";}

    using Real      = _Real;
    using Matrix    = Eigen::Matrix<_Real, N, N>;
    using Vector    = Eigen::Matrix<_Real, N, 1>;
    // generalize it N * N
    using VN2_T     = Eigen::Matrix<_Real, N * N, 1>;
    using MN2_T     = Eigen::Matrix<_Real, N * N, N * N>;

    SymmetricDirichlet(){
        setDeformationGradient(Matrix::Identity());
    }

    SymmetricDirichlet(const SymmetricDirichlet &other, UninitializedDeformationTag &&){ }

    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        m_F = F;
        m_Finv = F.inverse();
        m_J = F.determinant();

        if (elevel < EvalLevel::Hessian) return;
        m_projectionMask = (elevel != EvalLevel::HessianWithDisabledProjection);

        m_Finv_FinvT = m_Finv*m_Finv.transpose();
        m_FinvT_Finv = m_Finv.transpose()*m_Finv;
        if (!m_projectionMask) {
            Matrix dF = Matrix::Zero();
            for (int j = 0; j < dF.cols(); ++j) {
                for (int i = 0; i < dF.rows(); ++i) {
                    dF(i, j) = 1;
                    Eigen::Map<Matrix>(m_d2psi.col(N * j + i).data()) = delta_denergy(dF);
                    dF(i, j) = 0;
                }
            }
            return;
        }

        if (N == 3) throw std::runtime_error("Analytical Hessian Projection in N=3 Unimplemented!");

        // Analytical Hessian Projection in 2D
        Eigen::JacobiSVD<Matrix> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Matrix &U = svd.matrixU();
        const Matrix &V = svd.matrixV();
        const Vector &sigma = svd.singularValues();

        Real I1 = m_F.trace();
        Real I2 = m_F.squaredNorm();
        Real I3 = m_J;

        Real I3Sq = I3*I3;
        Real I3Cu = I3Sq*I3;

        Real lambda_1 = 1.0 + (3.0/pow(sigma[0],4));
        Real lambda_2 = 1.0 + (3.0/pow(sigma[1],4));
        Real lambda_3 = 1.0 + (1.0/I3Sq) + (I2/I3Cu);
        Real lambda_4 = 1.0 + (1.0/I3Sq) - (I2/I3Cu);

        VN2_T T;
        Eigen::Map<Matrix>(T.data()) = U.col(1)*V.col(0).transpose() - U.col(0)*V.col(1).transpose();
        
        VN2_T L;
        Eigen::Map<Matrix>(L.data()) = U.col(0)*V.col(1).transpose() + U.col(1)*V.col(0).transpose();

        VN2_T D1;
        Eigen::Map<Matrix>(D1.data()) = U.col(0)*V.col(0).transpose();

        VN2_T D2;
        Eigen::Map<Matrix>(D2.data()) = U.col(1)*V.col(1).transpose();

        lambda_4 = std::max(lambda_4, 0.0); // clamp potentially negative eigenvalue.
        m_d2psi = lambda_1*D1*D1.transpose() + lambda_2*D2*D2.transpose() + 0.5*lambda_3*L*L.transpose() + 0.5*lambda_4*T*T.transpose();
    }

    const Matrix &getDeformationGradient() const {return m_F; }

    _Real energy() const{
        if (m_J < 0) return std::numeric_limits<_Real>::infinity();
        else         return 0.5*(m_F.squaredNorm() + m_Finv.squaredNorm());
    }

    _Real denergy(const Matrix& dF) const { return doubleContract(denergy(), dF); }

    Matrix denergy() const{
        return m_F - m_Finv.transpose()*m_Finv*m_Finv.transpose();
    }

    // Symmetric!
    Matrix PK2Stress() const { return m_F.inverse() * denergy(); }

    template<class Mat_>
    Matrix delta_denergy(const Mat_ &dF) const{
        Matrix dF_mat = dF.matrix();
        Matrix tmp = (m_Finv.transpose() * dF_mat.transpose() * m_Finv.transpose());
        return dF_mat + (tmp + m_FinvT_Finv * dF_mat) * m_Finv_FinvT 
                      +  m_FinvT_Finv * tmp;
    }

    _Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        return doubleContract(delta_denergy(dF_lhs), dF_rhs);
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const {
        throw std::runtime_error("Unimplemented.");
    }

    using Hessian = Eigen::Matrix<Real, N * N, N * N>;
    const Hessian &d2energy() const { return m_d2psi; }

private:
    bool m_projectionMask = true; // when set to false, we disable projection regardless of `projectionEnabled` flag.

    Matrix m_F, m_Finv, m_Finv_FinvT, m_FinvT_Finv;
    Real m_J;

    Hessian m_d2psi;
};


#endif /* end of include guard: SYMMETRICDIRICHLET_HH */
