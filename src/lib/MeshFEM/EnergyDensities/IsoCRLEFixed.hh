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
        Eigen::JacobiSVD<Matrix> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV );
        const auto &U =svd.matrixU(),
                   &V =svd.matrixV();
        m_J = m_F.determinant();
        m_R = U * V.transpose();
        if (m_J < 0) {
            Matrix W = svd.matrixV();
            W.col(svd.matrixV().cols() - 1) *= -1;
            m_R = svd.matrixU() * W.transpose();
        }
        m_S = m_R.transpose() * F;
        m_traceSigma = m_S.trace();
        Matrix m_SinvT = m_S.inverse();
        m_FinvT = m_R * m_SinvT;
        m_sigma = svd.singularValues().array();
        // m_simpleFinvTEigenvalueDenominators = 0;
        // for (size_t i = 0; i < N; ++i) {
        //     m_simpleFinvTEigenvalueDenominators += m_J/m_sigma[i];
        // }
        

        // Analog to infinitesimal strain for linear elasticity.
        m_biotStrain = m_S - Matrix::Identity();

        m_biotStress = m_lambda * (m_J - 1) * m_J * m_SinvT + 2 * m_mu * m_biotStrain; //lambda (J-1)J S^-T + 2mu biotStrain
        m_pk1_stress = m_R * m_biotStress;
        m_pk1_stress = m_lambda * (m_J - 1) * m_J * m_F.inverse().transpose() + 2 * m_mu * (m_F - m_R);

        if (elevel < EvalLevel::Hessian) return;
        m_projectionMask = (elevel != EvalLevel::HessianWithDisabledProjection);

        if (N == 3) {
            m_flipEigenvalueDenominators = m_sigma;
            m_twistEigenvalueDenominators = m_traceSigma - svd.singularValues().array();
            // Construct eigenmatrices needed for Hessian evaluation (scaled by sqrt(2), not unit)
            Matrix A;
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    if (i==j) A(i,j) = m_J / (m_sigma[i] * m_sigma[j]);
                    else A(i,j) = (2 * m_J - 1) / (m_sigma[i] * m_sigma[j]);
                }
            }
            // A.triangularView<Eigen::Upper> = A.transpose();
            Eigen::SelfAdjointEigenSolver<Matrix> es;
            es.compute(A);
            Matrix vect = es.eigenvectors();
            m_AEigenvalue = m_J * es.eigenvalues().array();

            std::array<Matrix, N> temp;
            for (size_t i = 0; i < N; ++i) {
                temp[i] = U.col(i) * V.col(i).transpose();
            }

            for (size_t i = 0; i < N; ++i) {
                m_Tsqrt2[i] = U.col((i + 1) % N) * V.col((i + 2) % N).transpose()
                            - U.col((i + 2) % N) * V.col((i + 1) % N).transpose();
                m_Lsqrt2[i] = U.col((i + 1) % N) * V.col((i + 2) % N).transpose()
                            + U.col((i + 2) % N) * V.col((i + 1) % N).transpose();
                m_AEigenvector[i] = temp[0] * vect(0,i)
                                  + temp[1] * vect(1,i)
                                  + temp[2] * vect(2,i);
            }
        }
        else {
            m_flipEigenvalueDenominators[0] = 1.0;
            m_twistEigenvalueDenominators[0] = m_traceSigma;
            Matrix A;
            A << m_sigma[1] * m_sigma[1], 2 * m_J - 1, 2 * m_J - 1,  m_sigma[0] * m_sigma[0];
            Eigen::SelfAdjointEigenSolver<Matrix> es;
            es.compute(A);
            Matrix vect = es.eigenvectors();
            m_AEigenvalue = es.eigenvalues().array();

            std::array<Matrix, N> temp;
            for (size_t i = 0; i < N; ++i) {
                temp[i] = U.col(i) * V.col(i).transpose();
            }
            m_Tsqrt2[0] = U.col(0) * V.col(1).transpose()
                        - U.col(1) * V.col(0).transpose();
            m_Lsqrt2[0] = U.col(0) * V.col(1).transpose()
                        + U.col(1) * V.col(0).transpose();
            for (size_t i = 0; i < N; ++i) {
                m_AEigenvector[i] = temp[0] * vect(0,i)
                                  + temp[1] * vect(1,i);
            }
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
        Matrix result;
        if (usingProjection()){
            result = (2 * m_mu) * dF;
            constexpr size_t numTwistEigenmatrices = (N == 3 ? 3 : 1);
            for (size_t i = 0; i < numTwistEigenmatrices; ++i) {
                Real temp = m_lambda * m_flipEigenvalueDenominators[i] * (m_J - 1.0) / 2.0;
                Real coeff_t = (- 2.0 * m_mu) / m_twistEigenvalueDenominators[i] + temp;
                Real coeff_l = -temp;
                //ZZ `Real coeff = (- 2 * m_mu) / m_twistEigenvalueDenominators[i];`
                // Full eigenvalue (2 * mu + 2 * coeff) > 0 ==> coeff > -mu
                // coeff_t = std::max(coeff_t, -m_mu);
                // coeff_l = std::max(coeff_l, -m_mu);
                result += m_Tsqrt2[i] * (doubleContract(m_Tsqrt2[i], dF) *  coeff_t);
                result += m_Lsqrt2[i] * (doubleContract(m_Lsqrt2[i], dF) *  coeff_l);
            }
            // Simple
            //ZZ In Joey's SimpleDefinedDeltaP code: 
            // result += m_FinvT * (m_lambda * doubleContract(m_FinvT, dF));
            // Full
            for (size_t i = 0; i < N; ++i) {
                Real coeff = m_AEigenvalue[i] * m_lambda;
                // coeff = std::max(coeff, -2 * m_mu);
                result += m_AEigenvector[i] * (doubleContract(m_AEigenvector[i], dF) * coeff);
            }
        }
        else {
            if(N == 3){
                Matrix JFinvT,dR,dJFinvT,Dinv;
                Matrix D = m_traceSigma*Matrix::Identity() - m_S;
                Dinv=D.inverse();
                
                JFinvT(0, 0) = m_F(1, 1) * m_F(2, 2) - m_F(2, 1) * m_F(1, 2);
                JFinvT(0, 1) = m_F(2, 0) * m_F(1, 2) - m_F(1, 0) * m_F(2, 2);
                JFinvT(0, 2) = m_F(1, 0) * m_F(2, 1) - m_F(2, 0) * m_F(1, 1);
                JFinvT(1, 0) = m_F(2, 1) * m_F(0, 2) - m_F(0, 1) * m_F(2, 2);
                JFinvT(1, 1) = m_F(0, 0) * m_F(2, 2) - m_F(2, 0) * m_F(0, 2);
                JFinvT(1, 2) = m_F(2, 0) * m_F(0, 1) - m_F(0, 0) * m_F(2, 1);
                JFinvT(2, 0) = m_F(0, 1) * m_F(1, 2) - m_F(1, 1) * m_F(0, 2);
                JFinvT(2, 1) = m_F(1, 0) * m_F(0, 2) - m_F(0, 0) * m_F(1, 2);
                JFinvT(2, 2) = m_F(0, 0) * m_F(1, 1) - m_F(1, 0) * m_F(0, 1);
                
                Real dJ = 0;
                for(size_t alpha=0;alpha<3;alpha++){
                    for(size_t beta=0;beta<3;beta++){
                        dJ+=JFinvT(alpha,beta)*dF(alpha,beta);
                    }
                }

                //dR
                Matrix A,B;
                Vector a,b;
                A = m_R.transpose()*dF;
                b(0)=A(1,2)-A(2,1);b(1)=A(2,0)-A(0,2);b(2)=A(0,1)-A(1,0);
                a=Dinv*b;
                B(0,0)=0;B(0,1)=a(2);B(0,2)=-a(1);
                B(1,0)=-a(2);B(1,1)=0;B(1,2)=a(0);
                B(2,0)=a(1);B(2,1)=-a(0);B(2,2)=0;
                dR=m_R*B;

                dJFinvT(0, 0) = m_F(2, 2) * dF(1, 1) - m_F(2, 1) * dF(1, 2) - m_F(1, 2) * dF(2, 1) + m_F(1, 1) * dF(2, 2);
                dJFinvT(0, 1) = -m_F(2, 2) * dF(1, 0) + m_F(2, 0) * dF(1, 2) + m_F(1, 2) * dF(2, 0) - m_F(1, 0) * dF(2, 2);
                dJFinvT(0, 2) = m_F(2, 1) * dF(1, 0) - m_F(2, 0) * dF(1, 1) - m_F(1, 1) * dF(2, 0) + m_F(1, 0) * dF(2, 1);
                dJFinvT(1, 0) = -m_F(2, 2) * dF(0, 1) + m_F(2, 1) * dF(0, 2) + m_F(0, 2) * dF(2, 1) - m_F(0, 1) * dF(2, 2);
                dJFinvT(1, 1) = m_F(2, 2) * dF(0, 0) - m_F(2, 0) * dF(0, 2) - m_F(0, 2) * dF(2, 0) + m_F(0, 0) * dF(2, 2);
                dJFinvT(1, 2) = -m_F(2, 1) * dF(0, 0) + m_F(2, 0) * dF(0, 1) + m_F(0, 1) * dF(2, 0) - m_F(0, 0) * dF(2, 1);
                dJFinvT(2, 0) = m_F(1, 2) * dF(0, 1) - m_F(1, 1) * dF(0, 2) - m_F(0, 2) * dF(1, 1) + m_F(0, 1) * dF(1, 2);
                dJFinvT(2, 1) = -m_F(1, 2) * dF(0, 0) + m_F(1, 0) * dF(0, 2) + m_F(0, 2) * dF(1, 0) - m_F(0, 0) * dF(1, 2);
                dJFinvT(2, 2) = m_F(1, 1) * dF(0, 0) - m_F(1, 0) * dF(0, 1) - m_F(0, 1) * dF(1, 0) + m_F(0, 0) * dF(1, 1);

                result = 2*m_mu*(dF - dR) + m_lambda*dJ*JFinvT + m_lambda*(m_J-1)*dJFinvT;
            }
            else{
                Matrix JFinvT,dR,dJFinvT,C,A;
                JFinvT<<m_F(1,1),-m_F(1,0),-m_F(0,1),m_F(0,0);
                dJFinvT<<dF(1,1),-dF(1,0),-dF(0,1),dF(0,0);
                C<<0.0,-1.0,1.0,0.0;

                Real dJ=doubleContract(JFinvT, dF);
                A = m_R.transpose()*dF;
                Real a = doubleContract(C, A)/m_traceSigma;
                dR=a*m_R*C;

                result = 2 * m_mu*(dF - dR) + m_lambda*dJ*JFinvT + m_lambda*(m_J-1)*dJFinvT;
            }
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
    Matrix m_F, m_FinvT,
           m_R, m_S, // Polar decomposition
           m_biotStrain, m_biotStress, m_pk1_stress;
    Real m_traceSigma, m_J; //, m_simpleFinvTEigenvalueDenominators;
    Vector m_twistEigenvalueDenominators, m_flipEigenvalueDenominators, m_AEigenvalue, m_sigma;
    std::array<Matrix, N> m_Tsqrt2, m_Lsqrt2, m_AEigenvector;
};

#endif /* end of include guard: ISOCRLEFIXED_HH */