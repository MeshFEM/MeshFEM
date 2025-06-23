////////////////////////////////////////////////////////////////////////////////
// CommonNeoHookean.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  The variant of the neo-Hookean material model used in PolyFEM:
//      psi(F) = mu/2 (I1 - d - 2 ln(J)) + lambda/2 (ln(J))^2
//  It's also the primary version analyzed in [Smith et al. 2018: Stable Neo-Hookean].
//  Note that while this model is common in graphics, it does *not* properly
//  model 2D materials since it does not solve for the relaxation in the
//  transverse direction.
//  It corresponds to the UJOption = 2 setting discussed here:
//     https://osupdocs.forestry.oregonstate.edu/index.php/Neo-Hookean_Material
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/18/2025 12:18:22
*///////////////////////////////////////////////////////////////////////////////
#ifndef COMMONNEOHOOKEAN_HH
#define COMMONNEOHOOKEAN_HH

#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

template<typename _Real, size_t _Dim>
struct CommonNeoHookeanEnergy : public Concepts::NeoHookeanEnergy {
    static constexpr size_t Dimension = _Dim;
    static constexpr size_t N         = _Dim;
    static constexpr EDensityType EDType = EDensityType::FBased;
    using Real = _Real;
    using Matrix = Eigen::Matrix<Real, N, N>;

    static constexpr const char *name() { return "CommonNeoHookean"; }

    CommonNeoHookeanEnergy(const CommonNeoHookeanEnergy &/* other */) = default;
    CommonNeoHookeanEnergy() { setDeformationGradient(Matrix::Identity()); }
    CommonNeoHookeanEnergy &operator=(const CommonNeoHookeanEnergy &/* other */) = default;

    CommonNeoHookeanEnergy(const CommonNeoHookeanEnergy &other, UninitializedDeformationTag &&)
        : m_lambda(other.m_lambda), m_mu(other.m_mu) { }

    CommonNeoHookeanEnergy(Real lambda = 0, Real mu = 0.5)
        : m_lambda(lambda), m_mu(mu) { setDeformationGradient(Matrix::Identity()); }

    void setDeformationGradient(const Matrix& deformation_gradient, const EvalLevel elevel = EvalLevel::Full) {
        m_F = deformation_gradient;
        m_detF = deformation_gradient.determinant();
        m_logDetF = std::log(m_detF);

        if (elevel < EvalLevel::Gradient) return;
        m_Finv = m_F.inverse();
        m_Finv_T = m_Finv.transpose();
    }

    const Matrix &getDeformationGradient() const { return m_F; }

    Real energy() const {
        // Standard behavior: return inf for inverted elements
        if (m_detF < 0) return std::numeric_limits<Real>::infinity();
        return m_mu / 2.0 * (m_F.squaredNorm() - Dimension - 2.0 * m_logDetF) + m_lambda / 2.0 * m_logDetF * m_logDetF;
    }

    Matrix denergy() const { return m_mu * m_F + (m_lambda * m_logDetF - m_mu) * m_Finv_T; }
    Real denergy(const Matrix& dF) const { return doubleContract(dF, denergy()); }

    template<typename Mat_>
    Matrix delta_denergy(const Mat_ &dF) const {
        Matrix dF_mat = dF.matrix();
        return m_mu * dF_mat
            - (m_lambda * m_logDetF - m_mu) * m_Finv_T * dF_mat.transpose() * m_Finv_T
            + m_lambda * doubleContract(m_Finv_T, dF) * m_Finv_T;
        // return m_mu * (dF_mat + m_Finv_T * dF_mat.transpose() * m_Finv_T)
        //          - m_lambda * m_logDetF * m_Finv_T * dF_mat.transpose() * m_Finv_T
        //          + m_lambda * doubleContract(m_Finv_T, dF) * m_Finv_T;
    }

    Real d2energy(const Matrix& dF_a, const Matrix& dF_b) const {
        return doubleContract(dF_a, delta_denergy(dF_b));
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const {
        throw std::runtime_error("Unimplemented.");
    }

    // using Hessian = Eigen::Matrix<Real, N * N, N * N>;
    // Hessian d2energy() const {
    //     Hessian H;
    //     H.setZero();
    //     return H;
    // }

    Matrix PK2Stress() const { return m_Finv_T.transpose() * denergy(); }

private:
    Real m_lambda = 0.0; // Lame's first parameter
    Real m_mu = 0.0;     // Shear modulus

    // Cached deformation quantities.
    Matrix m_F, m_Finv, m_Finv_T;
    Real m_detF = 1.0, m_logDetF = 0.0;
};

#endif // COMMONNEOHOOKEAN_HH
