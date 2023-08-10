////////////////////////////////////////////////////////////////////////////////
// NeoHookeanEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Volumetric and plane stress Neo-Hookean energy.
*///////////////////////////////////////////////////////////////////////////////
#ifndef NEOHOOKEANENERGY_HH
#define NEOHOOKEANENERGY_HH

#include <Eigen/Dense>
// #include <unsupported/Eigen/KroneckerProduct>
#include <cstdlib>

#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>

// Implements the Neo-Hookean Energy described in the Neo-Hookean Energy section of doc/doc.pdf
template<typename _Real, size_t _Dim, template<typename, size_t> class _Derived_T>
struct NeoHookeanEnergyBase : public Concepts::NeoHookeanEnergy
{
    static constexpr size_t Dimension = _Dim;
    static constexpr size_t N         = _Dim;
    static constexpr EDensityType EDType = EDensityType::FBased;
    using Real = _Real;
    using Derived = _Derived_T<Real, N>;
    using Matrix = Eigen::Matrix<Real, N, N>;

    NeoHookeanEnergyBase(const NeoHookeanEnergyBase &/* other */) = default;
    NeoHookeanEnergyBase &operator=(const NeoHookeanEnergyBase &/* other */) = default;

    // Constructor copying material properties only, not the current deformation
    NeoHookeanEnergyBase(const NeoHookeanEnergyBase &other, UninitializedDeformationTag &&)
        : m_lambda(other.m_lambda), m_mu(other.m_mu)
    { }

    // Construct from Lame's first parameter (lambda) and shear modulus (mu).
    NeoHookeanEnergyBase(Real lambda, Real mu)
        : m_lambda(lambda), m_mu(mu)
    {
        setDeformationGradient(Matrix::Identity());
    }

    void setDeformationGradient(const Matrix& deformation_gradient, const EvalLevel elevel = EvalLevel::Full) {
        m_F = deformation_gradient;
        m_detF = deformation_gradient.determinant();
        if (elevel < EvalLevel::Gradient) return;
        m_Finv = m_F.inverse();
        m_Finv_T = m_Finv.transpose();
    }

    const Matrix &getDeformationGradient() const { return m_F; }

    Real energy() const {
        // Standard behavior: return inf for inverted elements
        if (m_detF < 0)
            return std::numeric_limits<Real>::max();


        const Real I3 = getI3();
        const Real I1 = getI1();

        return (m_mu / 2) * (I1 - 3) + (m_lambda / 4) * (I3 - 1) - std::log(I3) * (m_mu / 2 + m_lambda / 4);
    }

    Matrix denergy() const {
        return d_psi_d_I1() * d_I1_d_F() + d_psi_d_I3() * d_I3_d_F();
    }

    Real denergy(const Matrix& dF) const { return doubleContract(dF, denergy()); }

    Real d2energy(const Matrix& dF_a, const Matrix& dF_b) const {
        return doubleContract(dF_a, delta_denergy(dF_b));
    }

    // Directional derivative of "denergy" along dF:
    //      (d^2 psi / dF^2) : dF
    template<class Mat_>
    Matrix delta_denergy(const Mat_ &dF) const {
        Matrix dI3 = d_I3_d_F();
        Real delta_I3 = doubleContract(dI3, dF);
        return d_psi_d_I1() * delta_d_I1_d_F(dF) + (d2_psi_d2_I3() * delta_I3) * dI3 + d_psi_d_I3() * delta_d_I3_d_F(dF);
    }

    // (d^3 psi / dF^3) :: (dF_a \otimes dF_b)
    // Second variation of "denergy" along (dF_a, dF_b)
    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &dF_a, const Mat2_ &dF_b) const {
        Matrix dI3 = d_I3_d_F();
        Real delta_I3_a = doubleContract(dI3, dF_a),
             delta_I3_b = doubleContract(dI3, dF_b);
        Matrix delta_dI3_a = delta_d_I3_d_F(dF_a),
               delta_dI3_b = delta_d_I3_d_F(dF_b);
        return // Derivative of (d_psi_d_I1() * delta_d_I1_d_F(dF):                      (Note d2_psi_d_I1 = 0)
               d_psi_d_I1() * delta2_d_I1_d_F(dF_a, dF_b)                                // Symmetric
               // Derivative of (d2_psi_d2_I3() * delta_I3) * dI3:
             + (d3_psi_d3_I3() * delta_I3_b * delta_I3_a) * dI3                          // Symmetric
             + (d2_psi_d2_I3() *              doubleContract(delta_dI3_b, dF_a)) * dI3   // Symmetric
             + (d2_psi_d2_I3() *              delta_I3_a) * delta_dI3_b                  // Symmetric pair (*)
               // Derivative of d_psi_d_I3() * delta_d_I3_d_F(dF):
             + (d2_psi_d2_I3() *              delta_I3_b) * delta_dI3_a                  // Symmetric pair (*)
             + (  d_psi_d_I3()                          ) * delta2_d_I3_d_F(dF_a, dF_b); // Symmetric
    }

    ////////////////////////////////////////////////////////////////////////////
    // Invariants of the Cauchy-Green deformation tensor and their derivatives
    // appearing in the energy density expressions.
    // These must be provided by the derived class since they differ for the
    // 2D plane stress and 3D volumetric cases.
    ////////////////////////////////////////////////////////////////////////////
    // Trace of Cauchy-Green deformation tensor.
    Real getI1() const { return derived().getI1(); }

    // Determinant of Cauchy-Green deformation tensor.
    Real getI3() const { return derived().getI3(); }

    // dI1/dF
    Matrix d_I1_d_F() const { return derived().d_I1_d_F(); }

    // dI3/dF
    Matrix d_I3_d_F() const { return derived().d_I3_d_F(); }

    // (d^2 I1 / dF^2) : dF

    template<class Mat_>
    Matrix delta_d_I1_d_F(const Mat_ &dF) const { return derived().delta_d_I1_d_F(dF); }

    // (d^2 I1 / dF^2) : dF
    template<class Mat_>
    Matrix delta_d_I3_d_F(const Mat_ &dF) const { return derived().delta_d_I3_d_F(dF); }

    // (d^3 I1 / dF^3) :: (dF_a \otimes dF_b)
    Matrix delta2_d_I1_d_F(const Matrix &dF_a, const Matrix &dF_b) const { return derived().delta2_d_I1_d_F(dF_a, dF_b); }

    // (d^3 I3 / dF^3) :: (dF_a \otimes dF_b)
    Matrix delta2_d_I3_d_F(const Matrix &dF_a, const Matrix &dF_b) const { return derived().delta2_d_I3_d_F(dF_a, dF_b); }

    Matrix PK2Stress() const { return m_Finv_T.transpose() * denergy(); }

    const Derived &derived() const { return *static_cast<const Derived *>(this); }
protected:
    ////////////////////////////////////////////////////////////////////////////
    // Derivatives of the energy density with respect to the tensor invariants.
    ////////////////////////////////////////////////////////////////////////////
    // Derivative of the energy density with respect to I1
    Real d_psi_d_I1() const { return m_mu / 2; }

    // Derivative of the energy density with respect to I3
    Real d_psi_d_I3() const { return (m_lambda - (2 * m_mu + m_lambda) / getI3()) / 4; }

    // Second derivative of the energy density with respect to I3
    Real d2_psi_d2_I3() const {
        Real I3 = getI3();
        return (2 * m_mu + m_lambda) / (4 * I3 * I3);
    }

    // Third derivative of the energy density with respect to I3
    Real d3_psi_d3_I3() const {
        Real I3 = getI3();
        return - (m_mu + 0.5 * m_lambda) / (I3 * I3 * I3);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Derivatives of the "unpadded" invariants
    // (i.e., the 2x2 invariants for 2D, not including the C33 component)
    ////////////////////////////////////////////////////////////////////////////
    Real           unpaddedI3()     const { return m_detF * m_detF; }
    Matrix       d_unpaddedI3_d_F() const { return (2 * unpaddedI3()) * m_Finv_T; }
    template<typename Mat_>
    Real delta_unpaddedI3(const Mat_ &dF) const { return (2 * unpaddedI3()) * doubleContract(m_Finv_T, dF); }
    template<typename Mat_>
    Matrix delta_d_unpaddedI3_d_F(const Mat_ &dF) const {
        return (2 * delta_unpaddedI3(dF)) *  m_Finv_T
             - (2 *       unpaddedI3()  ) * (m_Finv * dF * m_Finv).transpose();
    }

    Matrix delta2_d_unpaddedI3_d_F(const Matrix &dF_a, const Matrix &dF_b) const {
        Real delta2_unpaddedI3_ab = doubleContract(delta_d_unpaddedI3_d_F(dF_a), dF_b);
        Matrix delta_Finv_a = -(m_Finv * dF_a * m_Finv),
               delta_Finv_b = -(m_Finv * dF_b * m_Finv);

        return (2 * delta2_unpaddedI3_ab)   *       m_Finv_T
             + (2 * delta_unpaddedI3(dF_a)) * delta_Finv_b.transpose()
             + (2 * delta_unpaddedI3(dF_b)) * delta_Finv_a.transpose()
             - (2 *       unpaddedI3()  ) * (delta_Finv_b * dF_a * m_Finv).transpose()
             - (2 *       unpaddedI3()  ) * (m_Finv * dF_a * delta_Finv_b).transpose();
    }

    Real m_lambda = 0.0; // Lame's first parameter
    Real m_mu = 0.0;     // Shear modulus

    // Cached deformation quantities.
    Matrix m_F, m_Finv, m_Finv_T;
    Real m_detF = 1.0;
};

template<typename _Real, size_t _Dim>
struct NeoHookeanEnergy;

template<typename _Real>
struct NeoHookeanEnergy<_Real, 2> : public NeoHookeanEnergyBase<_Real, 2, NeoHookeanEnergy>
{
    using Base = NeoHookeanEnergyBase<_Real, 2, ::NeoHookeanEnergy>;
    using Real = _Real;
    using Matrix = typename Base::Matrix;

    using Base::Base;

    NeoHookeanEnergy(const NeoHookeanEnergy &other)
        : Base(other), m_C33(other.m_C33) { }

    NeoHookeanEnergy &operator=(const NeoHookeanEnergy &other) = default; // Silence deprecation warning.

    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        Base::setDeformationGradient(F, elevel);
        if (m_lambda + m_mu == 0.0) m_C33 = 1.0; // Avoid `NaN` for zero-stiffness material.
        else m_C33 = (m_lambda + 2 * m_mu) / (m_lambda * unpaddedI3() + 2 * m_mu);
    }

    // Trace of full (padded) Cauchy-Green deformation tensor.
    Real getI1() const { return m_F.squaredNorm() + m_C33; }

    // Determinant of full (padded) Cauchy-Green deformation tensor.
    Real getI3() const { return unpaddedI3() * m_C33; }

    // dI1/dF
    Matrix d_I1_d_F() const { return 2 * m_F + d_C33_d_F(); }

    // dI3/dF
    Matrix d_I3_d_F() const { return d_unpaddedI3_d_F() * m_C33 + unpaddedI3() * d_C33_d_F(); }

    // (d^2 I1 / dF^2) : dF
    template<class Mat_>
    Matrix delta_d_I1_d_F(const Mat_ &dF) const { return 2 * dF.matrix() + delta_d_C33_d_F(dF); }

    // (d^2 I&1 / dF^2) : dF
    template<typename Mat_>
    Matrix delta_d_I3_d_F(const Mat_ &dF) const {
        Matrix dC33 = d_C33_d_F();
        Matrix d_unpaddedI3 = d_unpaddedI3_d_F();
        Real delta_unpaddedI3_val = doubleContract(d_unpaddedI3, dF);

        return delta_d_unpaddedI3_d_F(dF) * m_C33                    +
               d_unpaddedI3               * doubleContract(dC33, dF) +
               delta_unpaddedI3_val       * dC33                     +
               unpaddedI3()               * delta_d_C33_d_F(dF);
    }

    using Base::denergy; // prevent shadowing
    // Inlined and accelerated gradient calculation for 2D
    Matrix denergy() const {
        Real unpaddedI3_eval = unpaddedI3();
        Real d_psi_d_I3_eval = Base::d_psi_d_I3();

        Real FinvTCoeff = (d_C33_d_unpaddedI3() * unpaddedI3_eval);
        FinvTCoeff = (2 * unpaddedI3_eval) * d_psi_d_I3_eval * (FinvTCoeff + m_C33) + m_mu * FinvTCoeff;
        return m_mu * m_F + FinvTCoeff * m_Finv_T;
    }

    static constexpr size_t N = Base::N;
    using Hessian = Eigen::Matrix<Real, N * N, N * N>;
    using Base::d2energy; // prevent shadowing
    Hessian d2energy() const {
        // Inlined and deduplicated accelerated calculations...
        Hessian H;
        const Real d_psi_d_I1_eval         = Base::d_psi_d_I1();
        const Real d_psi_d_I3_eval         = Base::d_psi_d_I3();
        const Real d_C33_d_unpaddedI3_eval = d_C33_d_unpaddedI3();
        const Real unpaddedI3_eval         = m_detF * m_detF;
        const Real coeff_1 = (d_psi_d_I1_eval + d_psi_d_I3_eval * unpaddedI3_eval);
        const Real delta_d_unpaddedI3_coeff = (coeff_1 * d_C33_d_unpaddedI3_eval + d_psi_d_I3_eval * m_C33) * unpaddedI3_eval;
              Real coeff_2 = (m_C33 + (unpaddedI3_eval * d_C33_d_unpaddedI3_eval));
                   coeff_2 *= coeff_2 * Base::d2_psi_d2_I3();
                   coeff_2 += (2 * d_C33_d_unpaddedI3_eval) * (d_psi_d_I3_eval - m_C33 * coeff_1 * (m_lambda / (m_lambda + 2 * m_mu)));
                   coeff_2  = 2 * (coeff_2 * unpaddedI3_eval * unpaddedI3_eval + delta_d_unpaddedI3_coeff);

        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                // delta_F = e_i \otimes e_j
                Matrix delta_de = (coeff_2 * m_Finv_T(i, j)) * m_Finv_T
                                - delta_d_unpaddedI3_coeff * (m_Finv_T.col(j) * m_Finv.col(i).transpose());
                delta_de(i, j) += d_psi_d_I1_eval;
                delta_de *= 2;
                H.col(i + j * N) = Eigen::Map<const Eigen::Matrix<Real, N * N, 1>>(delta_de.data());
            }
        }

        return H;
    }

    // (d^3 I1 / dF^3) :: (dF_a \otimes dF_b)
    Matrix delta2_d_I1_d_F(const Matrix &dF_a, const Matrix &dF_b) const { return delta2_d_C33_d_F(dF_a, dF_b); }

    // (d^3 I3 / dF^3) :: (dF_a \otimes dF_b)
    Matrix delta2_d_I3_d_F(const Matrix &dF_a, const Matrix &dF_b) const {
        Matrix dC33 = d_C33_d_F();
        Real delta_C33_a   = doubleContract(dC33, dF_a),
             delta_C33_b   = doubleContract(dC33, dF_b);
        Real delta2_C33_ab = doubleContract(delta_d_C33_d_F(dF_a), dF_b);
        Matrix delta_d_unpaddedI3_a  = delta_d_unpaddedI3_d_F(dF_a),
               delta_d_unpaddedI3_b  = delta_d_unpaddedI3_d_F(dF_b);
        Real    delta2_unpaddedI3_ab = doubleContract(delta_d_unpaddedI3_a, dF_b);

        Matrix d_unpaddedI3 = d_unpaddedI3_d_F();
        return // Derivative of delta_d_unpaddedI3_d_F(dF) * m_C33:
               delta2_d_unpaddedI3_d_F(dF_a, dF_b) * m_C33 + delta_d_unpaddedI3_a * delta_C33_b
               // Derivative of d_unpaddedI3 * doubleContract(dC33, dF):
             + delta_d_unpaddedI3_b * delta_C33_a + d_unpaddedI3 * delta2_C33_ab
               // Derivative of delta_unpaddedI3 * dC33:
             + delta2_unpaddedI3_ab * dC33 + doubleContract(d_unpaddedI3, dF_a) * delta_d_C33_d_F(dF_b)
               // Derivative of unpaddedI3() * delta_d_C33_d_F(dF):
             + doubleContract(d_unpaddedI3, dF_b) * delta_d_C33_d_F(dF_a) + unpaddedI3() * delta2_d_C33_d_F(dF_a, dF_b);
    }

protected:
    // Derivative of normal component C33 with respect to the 2D deformation gradient.
    Matrix d_C33_d_F() const { return d_C33_d_unpaddedI3() * d_unpaddedI3_d_F(); }

    template<class Mat_>
    Matrix delta_d_C33_d_F(const Mat_ &dF) const {
        return delta_d_C33_d_unpaddedI3(dF) * d_unpaddedI3_d_F() +
               d_C33_d_unpaddedI3()         * delta_d_unpaddedI3_d_F(dF);
    }

    Matrix delta2_d_C33_d_F(const Matrix &dF_a, const Matrix &dF_b) const {
        Matrix dC33 = d_C33_d_F();
        Real delta_C33_a   = doubleContract(dC33, dF_a),
             delta_C33_b   = doubleContract(dC33, dF_b);
        Real delta2_C33_ab = doubleContract(delta_d_C33_d_F(dF_a), dF_b);

        Real coeff = -2 * m_lambda / (m_lambda + 2 * m_mu);
        if (m_lambda + m_mu == 0.0) coeff = 0.0; // Avoid `NaN` for zero-stiffness material.
        // Second variation of d_C33_d_unpaddedI3 along (dF_a, dF_b)
        Real delta2_d_C33_d_unpaddedI3_ab = coeff * (delta_C33_a * delta_C33_b + m_C33 * delta2_C33_ab);
        Real delta_d_C33_d_unpaddedI3_a   = coeff * m_C33 * delta_C33_a;
        Real delta_d_C33_d_unpaddedI3_b   = coeff * m_C33 * delta_C33_b;

        return delta2_d_C33_d_unpaddedI3_ab *        d_unpaddedI3_d_F()
             + delta_d_C33_d_unpaddedI3_a   *  delta_d_unpaddedI3_d_F(dF_b)
             + delta_d_C33_d_unpaddedI3_b   *  delta_d_unpaddedI3_d_F(dF_a)
             + d_C33_d_unpaddedI3()         * delta2_d_unpaddedI3_d_F(dF_a, dF_b);
    }

    // Derivative of normal component C33 with respect to the unpadded I3 invariant.
    Real d_C33_d_unpaddedI3() const {
        if (m_lambda + m_mu == 0.0) return 0.0; // Avoid `NaN` for zero-stiffness material.
        return -m_C33 * m_C33 * (m_lambda / (m_lambda + 2 * m_mu));
    }

    // Directional derivative of d_C33_d_unpaddedI3 along dF
    template<class Mat_>
    Real delta_d_C33_d_unpaddedI3(const Mat_ &dF) const {
        if (m_lambda + m_mu == 0.0) return 0.0; // Avoid `NaN` for zero-stiffness material.
        Real delta_C33 = doubleContract(d_C33_d_F(), dF);
        return -2 * m_C33 * delta_C33 * (m_lambda / (m_lambda + 2 * m_mu));
    }

private:
    using Base::m_F;
    using Base::m_Finv;
    using Base::m_Finv_T;
    using Base::m_detF;
    using Base::m_lambda;
    using Base::m_mu;
    using Base::unpaddedI3;
    using Base::d_unpaddedI3_d_F;
    using Base::delta_d_unpaddedI3_d_F;
    using Base::delta2_d_unpaddedI3_d_F;
    Real m_C33 = 1.0;
};

template<typename _Real>
struct NeoHookeanEnergy<_Real, 3> : public NeoHookeanEnergyBase<_Real, 3, NeoHookeanEnergy>
{
    using Base = NeoHookeanEnergyBase<_Real, 3, ::NeoHookeanEnergy>;
    using Real = _Real;
    using Matrix = typename Base::Matrix;
    using Base::Base;

    Real getI3() const { return m_detF * m_detF; }
    Real getI1() const { return m_F.squaredNorm(); }

    Matrix d_I1_d_F() const { return 2 * m_F; }
    Matrix d_I3_d_F() const { return this->d_unpaddedI3_d_F(); }

    template<typename Mat_> Matrix delta_d_I1_d_F(const Mat_ &dF) const { return 2 * dF.matrix(); }
    template<typename Mat_> Matrix delta_d_I3_d_F(const Mat_ &dF) const { return this->delta_d_unpaddedI3_d_F(dF); }

    Matrix delta2_d_I1_d_F(const Matrix &/* dF_a */, const Matrix /* &dF_b */) const { return Matrix::Zero(); }
    Matrix delta2_d_I3_d_F(const Matrix    &dF_a   , const Matrix    &dF_b   ) const { return this->delta2_d_unpaddedI3_d_F(dF_a, dF_b); }

    using Base::denergy;
    // Inlined and accelerated gradient calculation.
    Matrix denergy() const {
        return m_mu * m_F + ((0.5 * m_lambda) * (getI3() - 1) - m_mu) * m_Finv_T;
    }

    using Base::d2energy;
    static constexpr size_t N = 3;
    using Hessian = Eigen::Matrix<Real, N * N, N * N>;
    Hessian d2energy() const {
        Real I3_cache = getI3();
        Real coeff = ((0.5 * m_lambda) * (I3_cache - 1) - m_mu);
        Real coeff2 = m_lambda * I3_cache;

        Hessian H;
#if 1
        auto vectorize = [](Matrix A) { return Eigen::Map<const Eigen::Matrix<double, N * N, 1>>(A.data()); };
        H = m_mu * Hessian::Identity() + (coeff2 * vectorize(m_Finv_T)) * vectorize(m_Finv_T).transpose();
        for (size_t j = 0; j < N; ++j)
            for (size_t i = 0; i < N; ++i)
                Eigen::Map<Matrix>(H.col(i + j * N).data()) -= (coeff * m_Finv_T.col(j)) * m_Finv.col(i).transpose();

        // H -= coeff * Eigen::kroneckerProduct(m_Finv, m_Finv_T); // is this really a Kronecker product operation with the transposes going on?
#else
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                // delta_F = e_i \otimes e_j
                Matrix delta_de = (coeff2 * m_Finv_T(i, j)) * m_Finv_T
                                - coeff * (m_Finv_T.col(j) * m_Finv.col(i).transpose());
                delta_de(i, j) += m_mu;
                H.col(i + j * N) = Eigen::Map<const Eigen::Matrix<double, N * N, 1>>(delta_de.data());
            }
        }
#endif
        return H;
    }

private:
    using Base::m_F;
    using Base::m_Finv;
    using Base::m_Finv_T;
    using Base::m_detF;
    using Base::m_lambda;
    using Base::m_mu;
};

// Simulate a Neo-Hookean sheet made of a material with Poisson's ratio 0.5.
// This would cause the volumetric NeoHookeanEnergy above to blow up, but we
// can obtain a membrane energy by imposing incompressibility as a hard
// constraint.
template<typename _Real>
struct IncompressibleNeoHookeanEnergyCBased {
    static constexpr EDensityType EDType = EDensityType::CBased;
    static constexpr size_t Dimension = 2;
    static constexpr size_t N         = 2;
    using Real   = _Real;
    using Matrix = Eigen::Matrix<_Real, 2, 2>;
    using M2d    = Matrix;

    static constexpr const char *name() { return "IncompressibleNeoHookean"; }

    IncompressibleNeoHookeanEnergyCBased(const IncompressibleNeoHookeanEnergyCBased &other, UninitializedDeformationTag &&) {
        setYoungModulus(other.youngModulus());
    }

    IncompressibleNeoHookeanEnergyCBased(Real E = 6 /* corresponds to "stiffness"  of 1 */) {
        setYoungModulus(E);
    }

    void setC(const M2d &C) {
        Real a = C(0, 0),
             b = C(0, 1),
             c = C(1, 1);
        if (std::abs(b - C(1, 0)) > 1e-15) throw std::runtime_error("Asymmetric matrix");

        m_C = C;
        m_trace_C = C.trace();
        m_det_C = a * c - b * b;
        m_grad_det_C <<  c, -b,
                        -b,  a;
    }

    Real energy() const {
        return stiffness * (m_trace_C + 1.0 / m_det_C - 3.0);
    }

    // dpsi / dE = 2 * dpsi/dC
    M2d PK2Stress() const {
        return 2 * stiffness * (M2d::Identity() - (1.0 / (m_det_C * m_det_C)) * m_grad_det_C);
    }

    template<class Mat_>
    M2d delta_PK2Stress(const Mat_ &dC) const {
        M2d adj_dC;
        adj_dC << dC(1, 1), -dC(0, 1),
                 -dC(1, 0),  dC(0, 0);
        return 2 * (((2.0 * stiffness / (m_det_C * m_det_C * m_det_C)) * doubleContract(m_grad_det_C, dC)) * m_grad_det_C
                         - (stiffness / (m_det_C * m_det_C))                                               * adj_dC);
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_PK2Stress(const Mat_ &/* dC_a */, const Mat2_ &/* dC_b */) const {
        throw std::runtime_error("Unimplemented");
    }

    // The stiffness parameter is mu / 2 = E / (4 * (1 + nu)) = E / 6
    void setYoungModulus(Real E) {
        stiffness = E / 6;
    }

    Real youngModulus() const {
        return stiffness * 6;
    }

    void copyMaterialProperties(const IncompressibleNeoHookeanEnergyCBased &other) { setYoungModulus(other.youngModulus()); }

    Real stiffness = 1.0; // mu / 2

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Real m_trace_C, m_det_C;
    M2d m_grad_det_C;
    M2d m_C;
};

template <typename _Real>
using IncompressibleNeoHookeanEnergy = EnergyDensityFBasedFromCBased<IncompressibleNeoHookeanEnergyCBased<_Real>>;

#endif
