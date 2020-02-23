#ifndef NEOHOOKEANENERGY_HH
#define NEOHOOKEANENERGY_HH

#include <Eigen/Dense>
#include <cstdlib>

#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

/**
 *  Implements the Neo-Hookean Energy described in the Neo-Hookean Energy section of doc/doc.pdf
 */
template<typename _Real, size_t _Dim, template<typename, size_t> class _Derived_T>
struct NeoHookeanEnergyBase : public NeoHookeanEnergyConcept
{
    static constexpr size_t Dim = _Dim;
    using Real = _Real;
    using Derived = _Derived_T<Real, Dim>;
    using Matrix = Eigen::Matrix<Real, Dim, Dim>;

    NeoHookeanEnergyBase(const NeoHookeanEnergyBase& other) = default;
    NeoHookeanEnergyBase(Real lame_first_parameter, Real shear_modulus,
                         Real finite_continuation_start = -1)
        : m_lame_first_parameter(lame_first_parameter), m_shear_modulus(shear_modulus),
          m_finite_continuation_start(finite_continuation_start)
    { };

    void setDeformationGradient(const Matrix& deformation_gradient) {
        m_F = deformation_gradient;
        m_detF = deformation_gradient.determinant();
        m_Finv = m_F.inverse();
    }

    Real energy() const {
        // Standard behavior: return inf for inverted elements
        if (m_finite_continuation_start <= 0 && m_detF < 0) {
            return std::numeric_limits<Real>::max();
        }

        const Real I3 = getI3();
        const Real I1 = getI1();

        // Modified behavior to support inverted elements:
        // if det F < eps, we replace the log(I3) term by a constant + exp(- (det (F) - eps) )
        // where the constant is chosen such that the energy remains continuous
        if (m_finite_continuation_start > 0 && m_detF < m_finite_continuation_start) {
            Derived tmp(m_lame_first_parameter, m_shear_modulus, m_finite_continuation_start);
            Matrix tmp_F = Matrix::Identity();
            tmp_F(0, 0) = m_finite_continuation_start;
            tmp.setDeformationGradient(tmp_F);
            Real continuation_constant = - std::log(tmp.getI3()) * (m_lame_first_parameter / 2 + m_shear_modulus) / 2;

            return m_lame_first_parameter * (I3 - 1) / 4 + m_shear_modulus * (I1 - 3) / 2
                + continuation_constant + std::exp(-(m_detF - m_finite_continuation_start)) - 1;
        }

        const Real mu = m_shear_modulus, lambda = m_lame_first_parameter;
        return (mu / 2) * (I1 - 3)
             + (lambda / 4) * (I3 - 1)
             - std::log(I3) * (mu / 2 + lambda / 4);
    }

    Matrix denergy() const {
        if (m_finite_continuation_start > 0 && m_detF < m_finite_continuation_start) {
            Real dPsi3 = m_lame_first_parameter / 4;
            return (-std::exp(-(m_detF - m_finite_continuation_start))) * m_detF * m_Finv.transpose()
                + dPsi3 * d_I3_d_F()
                + d_psi_d_I1() * d_I1_d_F();
        }

        return d_psi_d_I1() * d_I1_d_F() + d_psi_d_I3() * d_I3_d_F();
    }

    Real denergy(const Matrix& dF) const { return doubleContract(dF, denergy()); }

    Real d2energy(const Matrix& dF_a, const Matrix& dF_b) const {
        return doubleContract(dF_a, delta_denergy(dF_b));
    }

    // (H : dF) where H is the hessian of the energy density with respect to F.
    // (directional derivative of "denergy" along dF)
    Matrix delta_denergy(const Matrix& dF) const {
        if (m_finite_continuation_start > 0 && m_detF < m_finite_continuation_start) {
            // ln I3 term is constant, but exp(-(detF)) got added
            Real dPsi3 = m_lame_first_parameter / 4;
            Real exp_term = -std::exp(-(m_detF - m_finite_continuation_start));

            Matrix d_det_dF = m_detF * m_Finv.transpose();
            Matrix delta_d_det_dF = doubleContract(d_det_dF, dF) * m_Finv.transpose()
                                  - m_detF * (m_Finv * dF * m_Finv).transpose();

            return exp_term * d_det_dF * d_det_dF
                + exp_term * delta_d_det_dF
                + dPsi3 * delta_d_I3_d_F(dF)
                + d_psi_d_I1() * delta_d_I1_d_F(dF);
        }

        Matrix dI3 = d_I3_d_F();
        Real delta_I3 = doubleContract(dI3, dF);
        return d_psi_d_I1() * delta_d_I1_d_F(dF) + (d2_psi_d2_I3() * delta_I3) * dI3 + d_psi_d_I3() * delta_d_I3_d_F(dF);
    }

    Real delta_d2energy(const Matrix &dF_a, const Matrix &dF_b, const Matrix &dF_c) {
        if (m_finite_continuation_start > 0) throw std::runtime_error("Finite continuation energy variant is not supported");
    }

    ////////////////////////////////////////////////////////////////////////////
    // Invariants of the Cauchy-Green deformation tensor and their derivatives
    // appearing in the energy density expressions.
    // These must be provided by the derived class since they differ for the
    // 2D plane stress and 3D volumetric cases.
    ////////////////////////////////////////////////////////////////////////////
    // Trace of Cauchy-Green deformation tensor.
    Real getI1() const { return static_cast<const Derived*>(this)->getI1(); }

    // Determinant of Cauchy-Green deformation tensor.
    Real getI3() const { return static_cast<const Derived*>(this)->getI3(); }

    // dI1/dF
    Matrix d_I1_d_F() const { return static_cast<const Derived*>(this)->d_I1_d_F(); }

    // dI3/dF
    Matrix d_I3_d_F() const { return static_cast<const Derived*>(this)->d_I3_d_F(); }

    // (d^2 I1 / dF^2) : dF
    Matrix delta_d_I1_d_F(const Matrix& dF) const { return static_cast<const Derived*>(this)->delta_d_I1_d_F(dF); }

    // (d^2 I1 / dF^2) : dF
    Matrix delta_d_I3_d_F(const Matrix& dF) const { return static_cast<const Derived*>(this)->delta_d_I3_d_F(dF); }

protected:
    ////////////////////////////////////////////////////////////////////////////
    // Derivatives of the energy density with respect to the tensor invariants.
    ////////////////////////////////////////////////////////////////////////////
    // Derivative of the energy density with respect to I1
    Real d_psi_d_I1() const { return m_shear_modulus / 2; }

    // Derivative of the energy density with respect to I3
    Real d_psi_d_I3() const { return (m_lame_first_parameter - (2 * m_shear_modulus + m_lame_first_parameter) / getI3()) / 4; }

    // Second derivative of the energy density with respect to I3
    Real d2_psi_d2_I3() const {
        Real I3 = getI3();
        return (2 * m_shear_modulus + m_lame_first_parameter) / (4 * I3 * I3);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Derivatives of the "unpadded" invariants
    // (i.e., the 2x2 invariants for 2D, not including the C33 component)
    ////////////////////////////////////////////////////////////////////////////
    Real unpaddedI3() const { return m_detF * m_detF; }
    // d_I3_d_F for the "unpadded" I3 invariant
    Matrix d_unpaddedI3_d_F() const { return (2 * m_detF * m_detF) * m_Finv.transpose(); }

    // delta_d_I3_d_F for the "unpadded" I3 invariant
    Matrix delta_d_unpaddedI3_d_F(const Matrix& dF) const {
        return (4 * m_detF * m_detF * doubleContract(m_Finv.transpose(), dF)) * m_Finv.transpose()
             - (2 * m_detF * m_detF                                         ) * (m_Finv * dF * m_Finv).transpose();
    }

    Matrix m_F, m_Finv;
    Real m_lame_first_parameter, m_shear_modulus;
    Real m_finite_continuation_start;
    Real m_detF;
};

template<typename _Real, size_t _Dim>
class NeoHookeanEnergy;

template<typename _Real>
struct NeoHookeanEnergy<_Real, 2> : public NeoHookeanEnergyBase<_Real, 2, NeoHookeanEnergy>
{
    using Base = NeoHookeanEnergyBase<_Real, 2, ::NeoHookeanEnergy>;
    using Real = _Real;
    using Matrix = typename Base::Matrix;
    using Base::Base;

    void setDeformationGradient(const Matrix &F) {
        Base::setDeformationGradient(F);
        m_C33 = (m_lame_first_parameter + 2 * m_shear_modulus) /
            (m_lame_first_parameter * m_detF * m_detF + 2 * m_shear_modulus);
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
    Matrix delta_d_I1_d_F(const Matrix& dF) const { return 2 * dF + delta_d_C33_d_F(dF); }

    // (d^2 I1 / dF^2) : dF
    Matrix delta_d_I3_d_F(const Matrix& dF) const {
        Matrix dC33 = d_C33_d_F();
        Matrix d_unpaddedI3 = d_unpaddedI3_d_F();
        Real delta_unpaddedI3 = doubleContract(d_unpaddedI3, dF);

        return delta_d_unpaddedI3_d_F(dF) * m_C33                    +
               d_unpaddedI3               * doubleContract(dC33, dF) +
               delta_unpaddedI3           * dC33                     +
               unpaddedI3()               * delta_d_C33_d_F(dF);
    }

protected:
    // Derivative of normal component C33 with respect to the 2D deformation gradient.
    Matrix d_C33_d_F() const { return d_C33_d_unpaddedI3() * d_unpaddedI3_d_F(); }

    Matrix delta_d_C33_d_F(const Matrix &dF) const {
        return delta_d_C33_d_unpaddedI3(dF) * d_unpaddedI3_d_F() +
               d_C33_d_unpaddedI3()         * delta_d_unpaddedI3_d_F(dF);
    }

    // Derivative of normal component C33 with respect to the unpadded I3 invariant.
    Real d_C33_d_unpaddedI3() const {
        return - m_C33 * m_C33 * (m_lame_first_parameter / (m_lame_first_parameter + 2 * m_shear_modulus));
    }

    // Directional derivative of d_C33_d_unpaddedI3 along dF
    Real delta_d_C33_d_unpaddedI3(const Matrix& dF) const {
        Real delta_C33 = doubleContract(d_C33_d_F(), dF);
        return - 2 * m_C33 * delta_C33 * (m_lame_first_parameter / (m_lame_first_parameter + 2 * m_shear_modulus));
    }

private:
    using Base::m_F;
    using Base::m_detF;
    using Base::m_lame_first_parameter;
    using Base::m_shear_modulus;
    using Base::unpaddedI3;
    using Base::d_unpaddedI3_d_F;
    using Base::delta_d_unpaddedI3_d_F;
    Real m_C33;
};

template<typename _Real>
struct NeoHookeanEnergy<_Real, 3> : public NeoHookeanEnergyBase<_Real, 3, NeoHookeanEnergy>
{
    using Base = NeoHookeanEnergyBase<_Real, 3, ::NeoHookeanEnergy>;
    using Real = _Real;
    using Matrix = typename Base::Matrix;
    using Base::Base;

    Real    getI3() const { return m_detF * m_detF; }
    Real    getI1() const { return m_F.squaredNorm(); }

    Matrix d_I1_d_F() const { return 2 * m_F; }
    Matrix d_I3_d_F() const { return this->d_unpaddedI3_d_F(); }

    Matrix delta_d_I1_d_F(const Matrix& dF) const { return 2 * dF; }
    Matrix delta_d_I3_d_F(const Matrix& dF) const { return this->delta_d_unpaddedI3_d_F(dF); }
private:
    using Base::m_F;
    using Base::m_detF;
    using Base::m_lame_first_parameter;
    using Base::m_shear_modulus;
};

#endif
