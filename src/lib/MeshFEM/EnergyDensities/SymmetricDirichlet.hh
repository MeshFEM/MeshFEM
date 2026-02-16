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
#include <MeshFEM/Utilities/fast_2x2_decompositions.hh>
#include <MeshFEM/Utilities/fast_3x3_decompositions.hh>

template<typename _Real, size_t _Dim>
struct SymmetricDirichlet {
    static constexpr size_t Dimension = _Dim;
    static constexpr size_t N         = _Dim;
    static constexpr EDensityType EDType = EDensityType::FBased;

    static constexpr const char *name() { return "SymmetricDirichlet"; }

    using Real      = _Real;
    using Matrix    = Eigen::Matrix<_Real, N, N>;
    using Vector    = Eigen::Matrix<_Real, N, 1>;
    // generalize it N * N
    using VN2_T     = Eigen::Matrix<_Real, N * N, 1>;
    using MN2_T     = Eigen::Matrix<_Real, N * N, N * N>;

    SymmetricDirichlet() { setDeformationGradient(Matrix::Identity()); }

    template<typename Real2>
    SymmetricDirichlet(const SymmetricDirichlet<Real2, _Dim> &other, UninitializedDeformationTag &&)
        : useAbsProjection(other.useAbsProjection), smoothingEpsilon(other.smoothingEpsilon) { }

    // template<bool Verbose = false>
    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        m_F = F;
        m_Finv = F.inverse();
        m_J = F.determinant();

        if (elevel < EvalLevel::Hessian) return;
        bool projecting = (elevel != EvalLevel::HessianWithDisabledProjection);

        using MMap = Eigen::Map<Matrix>;
        Hessian &H = m_d2psi;

        Matrix FinvT = m_Finv.transpose(); // copy to ensure contiguous memory access in loop below
        Matrix FinvT_Finv = FinvT * m_Finv;
        Matrix Finv_FinvT = m_Finv * FinvT;
        Matrix FinvT_Finv_FinvT = FinvT_Finv * FinvT;

        // ||F^-1||^2 term
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                MMap(H.col(N * j + i).data()) = FinvT.col(j) * FinvT_Finv_FinvT.row(i)
                                   +       FinvT_Finv.col(i) *       Finv_FinvT.col(j).transpose()
                                   + FinvT_Finv_FinvT.col(j) *           m_Finv.col(i).transpose();
            }
        }
        H.diagonal().array() += 1; // Dirichlet term

        if (!projecting) return; // projection disabled.

        if (N == 3) throw std::runtime_error("Analytical Hessian Projection in N=3 Unimplemented!");

        // Analytical Hessian Projection in 2D

#if 0 // Complete analytical eigendecomposition
#if 0
        Eigen::JacobiSVD<Matrix> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Matrix &U = svd.matrixU();
        const Matrix &V = svd.matrixV();
        const Vector &sigma = svd.singularValues();
#else
        Matrix U, V;
        Vector sigma;
        fast_decompositions::svd(F, U, sigma, V);
#endif
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
        MMap(T.data()) = U.col(1)*V.col(0).transpose() - U.col(0)*V.col(1).transpose();

        VN2_T L;
        MMap(L.data()) = U.col(0)*V.col(1).transpose() + U.col(1)*V.col(0).transpose();

        VN2_T D1;
        MMap(D1.data()) = U.col(0)*V.col(0).transpose();

        VN2_T D2;
        MMap(D2.data()) = U.col(1)*V.col(1).transpose();

        if (useAbsProjection)
            lambda_4 = std::abs(lambda_4);
        else
            lambda_4 = std::max(lambda_4, 0.0); // clamp potentially negative eigenvalue.
        H = lambda_1*D1*D1.transpose() + lambda_2*D2*D2.transpose() + 0.5*lambda_3*L*L.transpose() + 0.5*lambda_4*T*T.transpose();
#else // Modify only the potentially negative eigencomponent (the "Twist" mode)
        Real I2 = m_F.squaredNorm();
        Real I3 = m_J;
        Real I3Sq = I3*I3;
        Real I3Cu = I3Sq*I3;

        // Real lambda_4 = I3Cu + I3 - I2;
        Real lambda_4 = 1.0 + (1.0/I3Sq) - (I2/I3Cu);

        Real lambda_4_proj;
        if (usingSmoothProjection()) {
            // We use a smooth approximation to the max operation to avoid
            // discontinuities in higher-order AD derivatives. This adds a small
            // amount of additional stiffness to the "Twist" mode and clamps to a
            // slightly positive value that tends to zero as lambda_4 -> -Inf.
            // Specifically,
            // when `lambda_4 == 0`, `lambda_4_proj = sqrt(smooth_max_eps_sq) / 2`, and
            // when `lambda_4 << 0`, `lambda_4_proj ~ smooth_max_eps_sq / (4 * abs(lambda_4))`.
            double offset = 0; // smoothingEpsilon;
            lambda_4 += offset;
            lambda_4_proj = (0.5 * (lambda_4 + sqrt(lambda_4 * lambda_4 + smoothingEpsilon * smoothingEpsilon)));
            lambda_4_proj -= offset;
        }
        else lambda_4_proj = std::max(lambda_4, Real(0.0));

        // Real proj_dist = (lambda_4_proj - lambda_4) / I3Cu;
        Real proj_dist = lambda_4_proj - lambda_4;
        if (useAbsProjection)
            proj_dist *= 2.0; // adding twice the projection distance gets to the absolute value

        if (proj_dist != 0.0) {
            VN2_T T_vec;
            // The twist eigenmatrix can be rewritten in terms of the polar
            // decomposition `F = R S` as `R [0 -1; 1 0]`.
            // This is both more efficient and avoids numerical singularities in
            // autodiff that arise when using the standard `U` and `V` formulas.
            auto R = fast_decompositions::closest_rotation(m_F);
            MMap(T_vec.data()) << R.col(1), -R.col(0);
            H += (0.5 * proj_dist) * T_vec * T_vec.transpose();

            // if constexpr (Verbose) {
            //     std::cout << "lambda_4: " << lambda_4 << "\t" << lambda_4.c[1] << std::endl;
            //     std::cout << "lambda_4_proj: " << lambda_4_proj << "\t" << lambda_4_proj.c[1] << std::endl;
            //     std::cout << "smoothingEpsilon: " << smoothingEpsilon << std::endl;
            //     std::cout << "proj_dist: " << proj_dist << "\t" << proj_dist.c[1] << std::endl;
            //     std::cout << "R:\n" << R << std::endl << extractTaylorCoefficient(R, 1) << std::endl;
            // }
        }
        else {
            // if constexpr (Verbose) {
            //     std::cout << "lambda_4: " << lambda_4 << "\t" << lambda_4.c[1] << std::endl;
            //     std::cout << "lambda_4_proj: " << lambda_4_proj << "\t" << lambda_4_proj.c[1] << std::endl;
            //     std::cout << "smoothingEpsilon: " << smoothingEpsilon << std::endl;
            //     std::cout << "proj_dist: " << proj_dist << "\t" << proj_dist.c[1] << std::endl;
            // }
        }
#endif

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
        return applyFlattened4thOrderTensor(d2energy(), dF);
    }

    _Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        return doubleContract(delta_denergy(dF_lhs), dF_rhs);
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const {
        throw std::runtime_error("Unimplemented.");
    }

    bool usingSmoothProjection() const { return smoothingEpsilon > 0; }

    using Hessian = Eigen::Matrix<Real, N * N, N * N>;
    const Hessian &d2energy() const { return m_d2psi; }

    bool useAbsProjection = false;
    double smoothingEpsilon = 0; // set this to a small positive value to enable
                                 // the smooth version of the eigenvalue clamping.

private:
    Matrix m_F, m_Finv;
    Real m_J;

    Hessian m_d2psi;
};

#endif /* end of include guard: SYMMETRICDIRICHLET_HH */
