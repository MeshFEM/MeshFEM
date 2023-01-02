////////////////////////////////////////////////////////////////////////////////
// IsoCRLEFixed.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements the "fixed" version of corotated isotropic elasticity introduced
//  in [Stomakhin et al. 2012: Energetically Consistent Invertible Elasticity].
//  This is the standard corotated isotropic elasticity with the "lambda" term
//      lambda / 2 (tr(S-I))^2
//  replaced with an analogous term based on the exact determinant rather than
//  its small-strain linearization:
//      lambda / 2 (det(F) - 1)^2
//  This leads to more accurate volume preservation for large deformations and
//  stronger resistance to inversion.
//
//  An analytic Hessian projection is also provided.
*/
//  Author:  Zhan Zhang
//  Created:  05/29/2022 14:20:15
////////////////////////////////////////////////////////////////////////////////
#ifndef ISOCRLEFIXED_HH
#define ISOCRLEFIXED_HH

template <typename _Real, size_t _Dim>
struct IsoCRLEFixed {
    static constexpr size_t Dimension = _Dim;
    static constexpr size_t N         = _Dim;
    static constexpr EDensityType EDType = EDensityType::FBased;

    static constexpr size_t NumTwistEigenmodes = _Dim * (_Dim - 1) / 2;

    using Real    = _Real;
    using Matrix  = Eigen::Matrix<_Real, N, N>;
    using Vector  = Eigen::Matrix<_Real, N, 1>;
    using ETensor = ElasticityTensor<_Real, N>;
    using SMatrix = SymmetricMatrixValue<_Real, N>;

    static constexpr const char *name() { return "IsoCRLEFixed"; }

    IsoCRLEFixed(Real lambda, Real mu)
        : m_lambda(lambda), m_mu(mu) {
        setDeformationGradient(Matrix::Identity());
    }

    // Constructor copying material properties and settings only, not the current deformation
    IsoCRLEFixed(const IsoCRLEFixed &other, UninitializedDeformationTag &&)
        : projectionEnabled(other.projectionEnabled), m_lambda(other.m_lambda), m_mu(other.m_mu)
    { }

    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        m_F = F;
        m_J = m_F.determinant();

        Eigen::JacobiSVD<Matrix> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV );
        const auto &U = svd.matrixU();
        Matrix V = svd.matrixV();
        Vector sigma = svd.singularValues();

        // Use the "sign-flipped SVD" for inverted elements so that `R` is
        // always the closest *rotation* (rather than closest element of SU(N)).
        // The derivatives of the sign-flipped SVD are identical to the
        // derivatives of the SVD with the standard sign convention! We just
        // need to make sure to use the consistent sign-flipped quantities
        // throughout.
        if (m_J < 0) {
            V.rightCols(1) *= -1;
            sigma.tail(1) *= -1;
        }
        
        m_R = U * V.transpose();
        if (m_J < 0) {
            Matrix W = svd.matrixV();
            W.col(svd.matrixV().cols() - 1) *= -1;
            m_R = svd.matrixU() * W.transpose();
        }
        m_S = m_R.transpose() * F;
        Matrix SinvT = m_S.inverse();

        m_biotStrain = m_S - Matrix::Identity();
        m_biotStress = (m_lambda * (m_J - 1) * m_J) * SinvT + (2 * m_mu) * m_biotStrain; //lambda (J-1)J S^-T + 2mu biotStrain
        m_pk1_stress = m_R * m_biotStress;

        if (elevel < EvalLevel::Hessian) return;
        m_projectionMask = (elevel != EvalLevel::HessianWithDisabledProjection);

        if (N == 3) {
            m_flipEigenvalueCoeffs = sigma;
            m_twistEigenvalueDenominators = m_S.trace() - sigma.array();
        }
        else {
            m_flipEigenvalueCoeffs[0] = 1.0;
            m_twistEigenvalueDenominators[0] = m_S.trace();
        }

        Matrix A;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                A(i, j) = ((i == j) ? m_J : (2 * m_J - 1)) / (sigma[i] * sigma[j]);
            }
        }

        Eigen::SelfAdjointEigenSolver<Matrix> es;
        es.compute(A);
        m_AEigenvalue = m_J * es.eigenvalues();

        for (size_t i = 0; i < N; ++i) {
            m_AEigenvector[i].setZero();
            for (size_t j = 0; j < N; ++j)
                m_AEigenvector[i] += (es.eigenvectors()(j, i) * U.col(j)) * V.col(j).transpose();
        }

        // Flip and twist eigenvectors scaled by sqrt(2)
        for (size_t i = 0; i < NumTwistEigenmodes; ++i) {
            size_t j = (i + 1) % N, k = (i + 2) % N;
            m_Tsqrt2[i] = U.col(j) * V.col(k).transpose() - U.col(k) * V.col(j).transpose();
            m_Lsqrt2[i] = U.col(j) * V.col(k).transpose() + U.col(k) * V.col(j).transpose();
        }
    }

    const Matrix &getDeformationGradient() const { return m_F; }

    _Real energy() const {
        return m_mu * doubleContract(m_biotStrain, m_biotStrain) + m_lambda * (m_J -1) * (m_J -1) / 2.0;
    }

    // PK1 stress
    _Real denergy(const Matrix& dF) const { return doubleContract(denergy(), dF); }

    // Asymmetric!
    Matrix denergy() const { return m_pk1_stress; }

    // Symmetric!
    Matrix PK2Stress() const { return m_F.inverse() * denergy(); }

    const Matrix &R() const { return m_R; }
    const Matrix &S() const { return m_S; }
    const Matrix &biotStress() const { return m_biotStress; }

    template<class Mat_>
    Matrix delta_denergy(const Mat_ &dF) const { //ZZ SimpleDefinedDeltaP
        Matrix result = ((2 * m_mu) * dF).matrix();
        for (size_t i = 0; i < NumTwistEigenmodes; ++i) {
            Real coeff_t = m_lambda * m_flipEigenvalueCoeffs[i] * (m_J - 1.0) / 2.0;
            Real coeff_l = -coeff_t;
            coeff_t -= 2.0 * m_mu / m_twistEigenvalueDenominators[i];
            // Full eigenvalue (2 * mu + 2 * coeff) > 0 ==> coeff > -mu
            if (usingProjection()) {
                coeff_t = std::max(coeff_t, -m_mu);
                coeff_l = std::max(coeff_l, -m_mu);
            }
            result += m_Tsqrt2[i] * (doubleContract(m_Tsqrt2[i], dF) *  coeff_t);
            result += m_Lsqrt2[i] * (doubleContract(m_Lsqrt2[i], dF) *  coeff_l);
        }
        for (size_t i = 0; i < N; ++i) {
            Real coeff = m_AEigenvalue[i] * m_lambda;
            if (usingProjection()) coeff = std::max(coeff, -2 * m_mu);
            result += m_AEigenvector[i] * (doubleContract(m_AEigenvector[i], dF) * coeff);
        }
        return result;
    }

    _Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        return doubleContract(delta_denergy(dF_lhs), dF_rhs);
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const {
        throw std::runtime_error("Unimplemented.");
    }

    bool usingProjection() const { return projectionEnabled && m_projectionMask; }

    bool projectionEnabled = false;

private:
    Real m_lambda = 0.0;   // Lame's first parameter
    Real m_mu = 0.0;       // Shear modulus

    bool m_projectionMask = true; // when set to false, we disable projection regardless of `projectionEnabled` flag.

    ////////////////////////////////////////////////////////////////////////////
    // Deformed state quantities
    ////////////////////////////////////////////////////////////////////////////
    Matrix m_F,
           m_R, m_S, // Polar decomposition
           m_biotStrain, m_biotStress, m_pk1_stress;
    Real m_J;
    Vector m_AEigenvalue;
    Vector m_twistEigenvalueDenominators, m_flipEigenvalueCoeffs; // Only first NumTwistEigenmodes entries are used...

    std::array<Matrix, N> m_AEigenvector;
    std::array<Matrix, NumTwistEigenmodes> m_Tsqrt2, m_Lsqrt2;
};

#endif /* end of include guard: ISOCRLEFIXED_HH */
