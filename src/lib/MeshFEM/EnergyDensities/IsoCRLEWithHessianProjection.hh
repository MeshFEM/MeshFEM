////////////////////////////////////////////////////////////////////////////////
// IsoCRLEWithHessianProjection.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements an isotropic corotated linear elasticity model (both volumetric
//  and 2D) with an optional Hessian projection based on an analytic
//  eigendecomposition. Note, for 2D problems, the user must be careful to
//  to set the moduli properly, e.g., for plane stress or plane strain
//  according to their application. This differs from `NeoHookeanEnergy`,
//  which always expects moduli for the 3D volumetric material and implements
//  plane stress assumptions internally to implement the 2D energy density.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/14/2020 17:20:15
////////////////////////////////////////////////////////////////////////////////
#ifndef ISOCRLEWITHHESSIANPROJECTION_HH
#define ISOCRLEWITHHESSIANPROJECTION_HH

template <typename _Real, size_t _Dim>
struct IsoCRLEWithHessianProjection {
    static constexpr size_t Dimension = _Dim;
    static constexpr size_t N         = _Dim;
    static constexpr EDensityType EDType = EDensityType::FBased;
    using Real    = _Real;
    using Matrix  = Eigen::Matrix<_Real, N, N>;
    using Vector  = Eigen::Matrix<_Real, N, 1>;
    using ETensor = ElasticityTensor<_Real, N>;
    using SMatrix = SymmetricMatrixValue<_Real, N>;

    static constexpr const char *name() { return "IsoCRLEWithHessianProjection"; }

    IsoCRLEWithHessianProjection(Real lambda, Real mu)
        : m_lambda(lambda), m_mu(mu) {
        setDeformationGradient(Matrix::Identity());
    }

    // Constructor copying material properties and settings only, not the current deformation
    IsoCRLEWithHessianProjection(const IsoCRLEWithHessianProjection &other, UninitializedDeformationTag &&)
        : projectionEnabled(other.projectionEnabled), m_lambda(other.m_lambda), m_mu(other.m_mu)
    { }

    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        m_F = F;
        Eigen::JacobiSVD<Matrix> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV );
        const auto &U =svd.matrixU(),
                   &V =svd.matrixV();
        m_R = U * V.transpose();
        m_S = m_R.transpose() * F;
        m_traceSigma = m_S.trace();

        // Analog to infinitesimal strain for linear elasticity.
        m_biotStrain = m_S - Matrix::Identity();

        m_biotStress = m_lambda * (m_traceSigma - N) * Matrix::Identity() + 2 * m_mu * m_biotStrain;
        m_pk1_stress = m_R * m_biotStress;

        if (elevel < EvalLevel::Hessian) return;
        m_projectionMask = (elevel != EvalLevel::HessianWithDisabledProjection);

        if (N == 3) {
            m_twistEigenvalueDenominators = m_traceSigma - svd.singularValues().array();
            // Construct eigenmatrices needed for Hessian evaluation (scaled by sqrt(2), not unit)
            for (size_t i = 0; i < N; ++i) {
                m_Tsqrt2[i] = U.col((i + 1) % N) * V.col((i + 2) % N).transpose()
                            - U.col((i + 2) % N) * V.col((i + 1) % N).transpose();
            }
        }
        else {
            m_twistEigenvalueDenominators[0] = m_traceSigma;
            m_Tsqrt2[0] = U.col(0) * V.col(1).transpose()
                        - U.col(1) * V.col(0).transpose();
        }
    }

    const Matrix &getDeformationGradient() const { return m_F; }

    _Real energy() const { return 0.5 * doubleContract(m_biotStress, m_biotStrain); }

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
    Matrix delta_denergy(const Mat_ &dF) const {
        Matrix result = m_R * (m_lambda * doubleContract(m_R, dF)) + (2 * m_mu) * dF;
        constexpr size_t numTwistEigenmatrices = (N == 3 ? 3 : 1);
        for (size_t i = 0; i < numTwistEigenmatrices; ++i) {
            Real coeff = (m_lambda * (m_traceSigma - N) - 2 * m_mu) / m_twistEigenvalueDenominators[i];
            // Full eigenvalue (2 * mu + 2 * coeff) > 0 ==> coeff > -mu
            if (usingProjection()) coeff = std::max(coeff, -m_mu);
            result += m_Tsqrt2[i] * (doubleContract(m_Tsqrt2[i], dF) *  coeff);
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

    bool projectionEnabled = true;

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
    Real m_traceSigma;
    Vector m_twistEigenvalueDenominators;
    std::array<Matrix, N> m_Tsqrt2;
};

#endif /* end of include guard: ISOCRLEWITHHESSIANPROJECTION_HH */
