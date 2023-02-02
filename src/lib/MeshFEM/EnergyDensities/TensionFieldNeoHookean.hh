////////////////////////////////////////////////////////////////////////////////
// TensionNeoHookean.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  The tension-field-theory-based strain energy density used in Skouras 2014:
//  Designing Inflatable Structures.
//
//  This energy is implemented as a function of the right Green-Green
//  deformation tensor.
//
//  Note: the same energy density can also be achieved by wrapping the
//  `IncompressibleNeoHookeanEnergyCBased` energy in an
//  `EnergyDensityFBasedFromCBased`.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/04/2019 18:21:53
////////////////////////////////////////////////////////////////////////////////
#ifndef TENSIONFIELDENERGY_HH
#define TENSIONFIELDENERGY_HH

#include <Eigen/Dense>
#include <array>
#include "EigSensitivity.hh"
#include "Tensor.hh"

template<typename Real>
struct IncompressibleBalloonEnergy {
    using V2d  = Eigen::Matrix<Real, 2, 1>;
    using M2d  = Eigen::Matrix<Real, 2, 2>;

    IncompressibleBalloonEnergy() { }

    template<typename Derived>
    IncompressibleBalloonEnergy(const Eigen::MatrixBase<Derived> &C) { setMatrix(C); }

    template<typename Derived>
    void setMatrix(const Eigen::MatrixBase<Derived> &C) {
        static_assert((Derived::RowsAtCompileTime == 2) && (Derived::ColsAtCompileTime == 2), "Only 2x2 supported for now");

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

    Real denergy_dC(const M2d &dC) const {
        return stiffness * (dC.trace() - (1.0 / (m_det_C * m_det_C)) * doubleContract(m_grad_det_C, dC));
    }

    M2d denergy_dC() const {
        return stiffness * (M2d::Identity() - (1.0 / (m_det_C * m_det_C)) * m_grad_det_C);
    }

    Real d2energy_dC(const M2d &dC_a, const M2d &dC_b) const {
        return stiffness * ((2.0 / (m_det_C * m_det_C * m_det_C)) * doubleContract(m_grad_det_C, dC_a) * doubleContract(m_grad_det_C, dC_b)
                          - (1.0 / (m_det_C * m_det_C)) * (dC_a(1, 1) * dC_b(0, 0) + dC_a(0, 0) * dC_b(1, 1) - 2 * dC_a(0, 1) * dC_b(0, 1)));
    }

    M2d delta_denergy_dC(const M2d &dC) const {
        M2d adj_dC;
        adj_dC << dC(1, 1), -dC(0, 1),
                 -dC(1, 0),  dC(0, 0);
        return (((2.0 * stiffness / (m_det_C * m_det_C * m_det_C)) * doubleContract(m_grad_det_C, dC)) * m_grad_det_C
                     - (stiffness / (m_det_C * m_det_C))                                               * adj_dC);
    }

    // Second derivatives evaluated at the reference configuration
    M2d delta_denergy_dC_undeformed(const M2d &dC) const {
        return stiffness * (dC.trace() * M2d::Identity() + dC);
    }

    Real stiffness = 1.0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Real m_trace_C = 2, m_det_C = 1;
    M2d m_grad_det_C;
    M2d m_C;
};

template<typename Real>
struct TensionFieldEnergy {
    using V2d  = Eigen::Matrix<Real, 2, 1>;
    using M2d  = Eigen::Matrix<Real, 2, 2>;

    TensionFieldEnergy() { }

    template<typename Derived>
    TensionFieldEnergy(const Eigen::MatrixBase<Derived> &C) { setMatrix(C); }

    template<typename Derived>
    void setMatrix(const Eigen::MatrixBase<Derived> &C) {
        m_balloonEnergy   .setMatrix(C);
        m_eigSensitivities.setMatrix(C);
        setEigs(m_eigSensitivities.lambda(0), m_eigSensitivities.lambda(1));
    }
    Real energy() const {
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return 0.0;
        if (m_l2 < m_l2_tilde)            return m_balloonEnergy.stiffness * (m_l1 + 2.0 * m_l2_tilde - 3.0);
        return m_balloonEnergy.energy();
    }

    Real denergy_dC(const M2d &dC) const {
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return 0.0;
        if (m_l2 < m_l2_tilde)            return m_balloonEnergy.stiffness * ((1.0 - m_l2_tilde * m_l2_tilde * m_l2_tilde) * m_eigSensitivities.dLambda(dC)[0]);
        return m_balloonEnergy.denergy_dC(dC);
    }

    M2d denergy_dC() const {
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return M2d::Zero();
        if (m_l2 < m_l2_tilde)            return (m_balloonEnergy.stiffness * ((1.0 - m_l2_tilde * m_l2_tilde * m_l2_tilde))) * m_eigSensitivities.dLambda(0);
        return m_balloonEnergy.denergy_dC();
    }

    Real d2energy_dC(const M2d &dC_a, const M2d &dC_b) const {
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return 0.0;
        if (m_l2 < m_l2_tilde) {
            Real inv_l1 = 1.0 / m_l1;
            return m_balloonEnergy.stiffness * ((1.0 - inv_l1 * m_l2_tilde)         * m_eigSensitivities.d2Lambda(dC_a, dC_b)[0]
                                               + 1.5 * inv_l1 * inv_l1 * m_l2_tilde * m_eigSensitivities. dLambda(0, dC_a)
                                                                                    * m_eigSensitivities. dLambda(0, dC_b));
        }
        return m_balloonEnergy.d2energy_dC(dC_a, dC_b);
    }

    M2d delta_denergy_dC(const M2d &dC) const {
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return m_relaxedStiffnessEps * m_balloonEnergy.delta_denergy_dC_undeformed(dC); // add a small artificial stiffness to avoid rank-deficient Hessian in fully compressed regions
        if (m_l2 < m_l2_tilde) {
            Real inv_l1 = 1.0 / m_l1;
            return m_balloonEnergy.stiffness * ((1.0 - inv_l1 * m_l2_tilde)         * m_eigSensitivities.delta_dLambda(0, dC)
                                              + (1.5 * inv_l1 * inv_l1 * m_l2_tilde * m_eigSensitivities.      dLambda(0, dC))
                                                                                    * m_eigSensitivities.      dLambda(0)
                                              + (m_relaxedStiffnessEps * m_eigSensitivities.dLambda(1, dC)) // add a small artificial stiffness to avoid rank-deficient Hessian in regions of partial tension
                                                                       * m_eigSensitivities.dLambda(1));
        }
        return m_balloonEnergy.delta_denergy_dC(dC);
    }

    size_t tensionState() const {
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return 0; // compression in both directions
        if (m_l2 < m_l2_tilde)            return 1; //     tension in one  direction
        return 2;                                   //     tension in both directions
    }

    Real stiffness() const { return m_balloonEnergy.stiffness; }
    void setStiffness(Real val ) { m_balloonEnergy.stiffness = val; }

    // Note: the stiffness added in the relaxed case is also proportional to stiffness()!
    void setRelaxedStiffnessEpsilon(Real val) { m_relaxedStiffnessEps = val; }
    Real getRelaxedStiffnessEpsilon() const { return m_relaxedStiffnessEps; }

    const EigSensitivity<Real> &eigSensitivities() const { return m_eigSensitivities; }

    // Useful for visualizing/debugging the energy and its derivatives at
    // arbitrary (l1, l2)...
    void setEigs(Real l1, Real l2) {
        if (l1 < l2) std::swap(l1, l2);
        m_l1 = l1;
        m_l2 = l2;
        m_l2_tilde = 1.0 / std::sqrt(l1);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Expressions in terms of eigenvalues for debugging/developing smoothed
    // energy density.
    ////////////////////////////////////////////////////////////////////////////
    // Energy expressed in terms of eigenvalues.
    Real psi() const {
        const Real k = m_balloonEnergy.stiffness;
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return 0.0;
        if (m_l2 < m_l2_tilde)            return k * (m_l1 + 2.0 * m_l2_tilde - 3.0);
        // full neo-Hookean energy in terms of eigenvalues
        return k * (m_l1 + m_l2 + 1.0 / (m_l1 * m_l2) - 3.0);
    }

    // Derivatives of energy with respect to eigenvalues l1, l2 (Note: these formulas can
    // not be used for computing denergy(dC) when l1 == l2 since the eigenvalues become
    // non-smooth functions of C in this case).
    V2d dpsi_dl() const {
        const Real k = m_balloonEnergy.stiffness;
        V2d result(V2d::Zero());
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return V2d::Zero();
        if (m_l2 < m_l2_tilde) return k * V2d(1.0 - m_l2_tilde * m_l2_tilde * m_l2_tilde, 0.0);
        return k * V2d(1.0 - 1.0 / (m_l1 * m_l1 * m_l2),
                       1.0 - 1.0 / (m_l1 * m_l2 * m_l2));
    }

    M2d d2psi_dl2() const {
        if ((m_l1 < 1.0) && (m_l2 < 1.0)) return M2d::Zero();
        M2d result;
        const Real k = m_balloonEnergy.stiffness;
        const Real inv_l1 = 1.0 / m_l1;
        if (m_l2 < m_l2_tilde) {
            result << k * 1.5 * inv_l1 * inv_l1 * m_l2_tilde, 0.0, 0.0, 0.0;
        }
        else {
            result << 2.0 / (m_l1 * m_l1 * m_l1 * m_l2), 1.0 / (m_l1 * m_l1 * m_l2 * m_l2),
                      1.0 / (m_l1 * m_l1 * m_l2 * m_l2), 2.0 / (m_l1 * m_l2 * m_l2 * m_l2);
            result *= k;
        }
        return result;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
protected:
    IncompressibleBalloonEnergy<Real> m_balloonEnergy;
    EigSensitivity<Real>              m_eigSensitivities;
    Real m_l1 = 0.0, m_l2 = 0.0, m_l2_tilde = 0.0;
    bool useSmoothedTensionField = false;
    Real m_relaxedStiffnessEps = 1e-8;
};

// Optionally use either TensionFieldEnergy or IncompressibleBalloonEnergy
template<typename _Real>
struct OptionalTensionFieldEnergy : public TensionFieldEnergy<Real> {
    using Real = _Real;
    using TFE = TensionFieldEnergy<Real>;
    using V2d = typename TFE::V2d;
    using M2d = typename TFE::M2d;

    OptionalTensionFieldEnergy() { }

    template<typename Derived>
    OptionalTensionFieldEnergy(const Eigen::MatrixBase<Derived> &C) : TFE(C) { }

    Real energy() const {
        if (useTensionField) return TFE::energy();
        return TFE::m_balloonEnergy.energy();
    }

    Real denergy_dC(const M2d &dC) const {
        if (useTensionField) return TFE::denergy_dC(dC);
        return TFE::m_balloonEnergy.denergy_dC(dC);
    }

    M2d denergy_dC() const {
        if (useTensionField) return TFE::denergy_dC();
        return TFE::m_balloonEnergy.denergy_dC();
    }

    M2d delta_denergy_dC(const M2d &dC) const {
        if (useTensionField) return TFE::delta_denergy_dC(dC);
        return TFE::m_balloonEnergy.delta_denergy_dC(dC);
    }

    Real d2energy_dC(const M2d &dC_a, const M2d &dC_b) const {
        if (useTensionField) return TFE::d2energy_dC(dC_a, dC_b);
        return TFE::m_balloonEnergy.d2energy_dC(dC_a, dC_b);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Conform to the C-based energy interface
    // (So we can use the membrane energy density wrapper)
    ////////////////////////////////////////////////////////////////////////////
    static std::string name() { return std::string("OptionalTensionFieldEnergy"); }

    OptionalTensionFieldEnergy(const OptionalTensionFieldEnergy &other, UninitializedDeformationTag &&) {
        copyMaterialProperties(other);
    }

    static constexpr EDensityType EDType = EDensityType::CBased;
    static constexpr size_t Dimension = 2;
    static constexpr size_t N         = 2;
    using Matrix = M2d;
    void setC(Eigen::Ref<const M2d> C) { setMatrix(C); }
    M2d PK2Stress() const { return 2.0 * denergy_dC(); }

    template<class Mat_>
    M2d delta_PK2Stress(const Mat_ &dC) const { return 2.0 * delta_denergy_dC(dC.matrix()); }

    template<class Mat_, class Mat2_>
    M2d delta2_PK2Stress(const Mat_ &/* dC_a */, const Mat2_ &/* dC_b */) const {
        throw std::runtime_error("Unimplemented");
    }

    bool useTensionField = true;

    void copyMaterialProperties(const OptionalTensionFieldEnergy &b) {
        setStiffness(b.stiffness());
        setRelaxationEnabled(b.getRelaxationEnabled());
        setRelaxedStiffnessEpsilon(b.getRelaxedStiffnessEpsilon());
    }
    bool getRelaxationEnabled() const { return useTensionField; }
    void setRelaxationEnabled(bool enable) { useTensionField = enable; }
};

#endif /* end of include guard: TENSIONFIELDENERGY_HH */
