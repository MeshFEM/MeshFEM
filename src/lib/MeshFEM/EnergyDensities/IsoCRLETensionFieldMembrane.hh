////////////////////////////////////////////////////////////////////////////////
// IsoCRLETensionFieldMembrane.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements an isotropic F-based corotated linear elasticity 2D membrane model
//  with an optional tension field theory relaxation, and optional
//  smoothing for this relaxation.
//  Note, for 2D problems, the user must be careful to to set the moduli
//  properly, e.g., for plane stress or plane strain according to their
//  application. This differs from `NeoHookeanEnergy`, which always expects
//  moduli for the 3D volumetric material and implements plane stress
//  assumptions internally to implement the 2D energy density.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  10/23/2020 15:45:32
////////////////////////////////////////////////////////////////////////////////
#ifndef ISOCRLETENSIONFIELD_HH
#define ISOCRLETENSIONFIELD_HH

template <typename _Real>
struct IsoCRLETensionFieldMembrane {
    static constexpr size_t Dimension          = 2;
    static constexpr size_t EmbeddingDimension = 3;
    static constexpr size_t N                  = Dimension;
    static constexpr size_t M                  = EmbeddingDimension;
    using Real = _Real;
    using M32d = Eigen::Matrix<Real, 3, 2>;
    using M3d  = Eigen::Matrix<Real, 3, 3>;
    using M2d  = Eigen::Matrix<Real, 2, 2>;
    using V3d  = Eigen::Matrix<Real, 3, 1>;
    using V2d  = Eigen::Matrix<Real, 2, 1>;
    using Matrix = M32d;

    static constexpr const char *name() { return "IsoCRLETensionFieldMembrane"; }

    // Construct from Young's modulus, Poisson's ratio, defaulting
    // to values that correspond to a "stiffness" of 1 (in Skouras 2014's
    // incompressible neo-Hookean model).
    IsoCRLETensionFieldMembrane(Real E = 6, Real nu = 0.5) {
        setYoungPoisson(E, nu, false);
        setDeformationGradient(M32d::Identity());
    }

    // Constructor copying material properties and settings only, not the current deformation
    IsoCRLETensionFieldMembrane(const IsoCRLETensionFieldMembrane &other, UninitializedDeformationTag &&)
    {
        copyMaterialProperties(other);
    }

    void setYoungPoisson(Real E, Real nu, bool updateCache = true) {
        m_E  = E;
        m_nu = nu;
        m_lambda_div_nu = m_E / (1 - m_nu * m_nu);
        if (updateCache) setDeformationGradient(getDeformationGradient());
    }

    void copyMaterialProperties(const IsoCRLETensionFieldMembrane &other) {
        relaxationEnabled        = other.relaxationEnabled;
        smoothingEnabled         = other.smoothingEnabled;
        hessianProjectionEnabled = other.hessianProjectionEnabled;
        smoothingEps             = other.smoothingEps;
        relaxedStiffnessEps      = other.relaxedStiffnessEps;
        setYoungPoisson(other.m_E, other.m_nu, false);
    }

    void setDeformationGradient(const M32d &F, const EvalLevel /* elevel */ = EvalLevel::Full) {
        m_F = F;
        Eigen::JacobiSVD<M32d> svd;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV );
        m_U = svd.matrixU();
        m_V = svd.matrixV();

        m_sigma  = svd.singularValues();
        m_strain = m_sigma.array() - 1.0;
        m_excessStrain = m_strain[1] + m_nu * m_strain[0];
    }

    Real eps() const {
        if (!strainDependentSmoothing) return smoothingEps;
        return smoothingEps * (m_sigma[0] - m_sigma[1]);
    }

    // We avoid a near divide by zero when the smoothing amount becomes extremely small
    bool smoothingActive() const {
        return smoothingEnabled && eps() > 1e-10;
    }

    Real c(Real x) const {
        const Real e = eps();
        if (!relaxationEnabled || (x > e)) return 0.5 * x * x;
        if (!smoothingActive()) return 0.5 * std::pow(std::max<Real>(x, 0), 2);
        if (x < -e) return -std::pow(e, 2) / 6.0;
        return std::pow(e, 2) / 12.0 * (std::pow(x / e + 1, 3) - 2);
    }

    Real dc_dx(Real x) const {
        const Real e = eps();
        if (!relaxationEnabled || (x > e)) return x;
        if (!smoothingActive()) return std::max<Real>(x, 0);
        if (x < -e) return 0.0;
        return e / 4.0 * std::pow(x / e + 1, 2);
    }

    Real d2c_dx2(Real x) const {
        const Real e = eps();
        if (!relaxationEnabled || (x > e)) return 1;
        if (!smoothingActive()) return (x >= 0) ? 1 : 0;
        if (x < -e) return 0.0;
        return 1.0 / 2.0 * (x / e + 1);
    }

    Real dc_de(Real x) const {
        const Real e = eps();
        if (!relaxationEnabled || !smoothingActive()) return 0.0;
        if (x < -e) return -e / 3;
        if (x <  e) return -e * std::pow(x / e - 1, 2) * (x / e + 2) / 12;
        return 0.0;
    }

    Real d2c_dxde(Real x) const {
        const Real e = eps();
        if (!relaxationEnabled || !smoothingActive()) return 0.0;
        if (x < -e) return 0.0;
        if (x <  e) return 0.25 * (1 - std::pow(x / e, 2));
        return 0.0;
    }

    Real d2c_de2(Real x) const {
        const Real e = eps();
        if (!relaxationEnabled || !smoothingActive()) return 0.0;
        if (x < -e) return -1 / 3;
        if (x <  e) return (1 / 6) * (std::pow(x / e, 3) - 1);
        return 0.0;
    }

    const M32d &getDeformationGradient() const { return m_F; }

    V2d dPsi_dSigma() const {
        Real dPsi_dExcessStrain = m_lambda_div_nu * dc_dx(m_excessStrain);
        V2d result(m_E * dc_dx(m_strain[0]) + m_nu * dPsi_dExcessStrain, dPsi_dExcessStrain);
        if (smoothingActive() && strainDependentSmoothing) {
            Real dPsi_deps = m_E * dc_de(m_strain[0]) + m_lambda_div_nu * dc_de(m_excessStrain);
            result += V2d(dPsi_deps, -dPsi_deps);
        }
        return result;
    }

    Real energy() const { return m_E * c(m_strain[0]) + m_lambda_div_nu * c(m_excessStrain); }

    // PK1 stress
    _Real denergy(const M32d& dF) const { return doubleContract(denergy(), dF); }
    M32d denergy() const {
        return  m_U.leftCols(2) * dPsi_dSigma().asDiagonal() * m_V.transpose();
    }

    M2d PK2Stress() const { throw std::runtime_error("Unimplemented"); }

    template<class Mat_>
    M32d delta_denergy(const Mat_ &dF) const {
        const M32d UtdFV = m_U.transpose() * dF * m_V; // dF in the singular vector basis
        M32d result; // Result ***in the singular vector basis***

        Real e = eps();

        const bool isRelaxed      = relaxationEnabled && (m_excessStrain < (smoothingActive() ?  e : 0));
        const bool isFullyRelaxed = relaxationEnabled && (   m_strain[0] < (smoothingActive() ? -e : 0));

        if (isFullyRelaxed) {
            // Add a small artificial stiffness to avoid a singular Hessian in regions of full compression
            return relaxedStiffnessEps * unrelaxed_delta_denergy_undeformed(dF);
        }

        // G (Eigenvalue always nonnegative)
        {
            const Real excessStrainStiffness = m_lambda_div_nu * d2c_dx2(m_excessStrain);
            const Real dexcessStrainCoeff    = excessStrainStiffness * (UtdFV(1, 1) + m_nu * UtdFV(0, 0));
            result(0, 0) = m_E * d2c_dx2(m_strain[0]) * UtdFV(0, 0) + m_nu * dexcessStrainCoeff;
            result(1, 1) = dexcessStrainCoeff;
            if (excessStrainStiffness < relaxedStiffnessEps) {
                // Add a small artificial stiffness to avoid a singular Hessian in regions of partial compression
                result(1, 1) += (relaxedStiffnessEps - excessStrainStiffness) * UtdFV(1, 1);
            }
        }

        const Real dc_e0_term = m_E * dc_dx(m_strain(0)),
               dc_excess_term = m_lambda_div_nu * dc_dx(m_strain[1] + m_nu * m_strain[0]);
        // L (Eigenvalue always nonnegative)
        Real Lcoeff;
        if (isRelaxed) {
            Lcoeff = dc_e0_term + (m_nu - 1) * dc_excess_term;
            Real den = m_sigma[0] - m_sigma[1];
            if (den < 1e-16) den = 1e-16;
            Lcoeff = std::min<Real>(std::abs(Lcoeff / den), 1e6); // clamp to moderate finite value. Largish values should not be problematic since we only use the inverse Hessian...
        }
        else {
            Lcoeff = m_E / (1 + m_nu); // Use robust formula avoiding 0/0 in unrelaxed case
        }
        result(0, 1) = result(1, 0) = 0.5 * Lcoeff * (UtdFV(1, 0) + UtdFV(0, 1)); // 0.5 is from the 1/sqrt(2) normalization of each L

        // T
        Real Tcoeff = 0.5 * (dc_e0_term + (m_nu + 1) * dc_excess_term) / (m_sigma[0] + m_sigma[1]);
        if (hessianProjectionEnabled && Tcoeff < 0.0) Tcoeff = 0.0; // In-plane twisting instability only happens under full compression
        Tcoeff *= (UtdFV(1, 0) - UtdFV(0, 1));
        result(1, 0) +=  Tcoeff;
        result(0, 1) += -Tcoeff;

        // Omega_y
        Real Omega_y_coeff = (dc_e0_term + m_nu * dc_excess_term) / m_sigma[0];
        if (hessianProjectionEnabled && Omega_y_coeff < 0.0) Omega_y_coeff = 0.0; // Rotational instability around y happens when element experiences compression in the x (0th) direction
        result(2, 0) = UtdFV(2, 0) * Omega_y_coeff;

        // Omega_x
        Real Omega_x_coeff = dc_excess_term / m_sigma[1];
        if (hessianProjectionEnabled && Omega_x_coeff < 0.0) Omega_x_coeff = 0.0; // Rotational instability around x happens when element experiences compression in the y (1st) direction
        result(2, 1) = UtdFV(2, 1) * Omega_x_coeff;

        return m_U * result * m_V.transpose(); // Change back to the standard basis
    }

    _Real d2energy(const M32d &dF_lhs, const M32d &dF_rhs) const {
        return doubleContract(delta_denergy(dF_lhs), dF_rhs);
    }

    // Second derivatives evaluated at the corotated reference configuration
    template<class Mat_>
    M32d unrelaxed_delta_denergy_undeformed(const Mat_ &dF) const {
        const M32d UtdFV = m_U.transpose() * dF * m_V; // dF in the singular vector basis
        const Real PLcoeff = 0.5 * m_E / (1 + m_nu);
        M32d result; // Result ***in the singular vector basis***
        Real Pcoeff = PLcoeff * (UtdFV(0, 0) - UtdFV(1, 1));
        Real Lcoeff = PLcoeff * (UtdFV(1, 0) + UtdFV(0, 1));
        Real Rcoeff = 0.5 * (m_E / (1 - m_nu)) * (UtdFV(0, 0) + UtdFV(1, 1));
        result(0, 0) =  Pcoeff + Rcoeff;
        result(0, 1) =  Lcoeff;
        result(1, 0) =  Lcoeff;
        result(1, 1) = -Pcoeff + Rcoeff;
        result.row(2).setZero();
        return m_U * result * m_V.transpose(); // Change back to the standard basis
    }

    template<class Mat_, class Mat2_>
    M32d delta2_denergy(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const {
        throw std::runtime_error("Unimplemented.");
    }

    bool relaxationEnabled = true,
         smoothingEnabled = true,
         strainDependentSmoothing = false,
         hessianProjectionEnabled = false;
    Real smoothingEps = 1 / 512.0;
    Real relaxedStiffnessEps = 1e-8;

    const M3d &U() const { return m_U; }
    const M2d &V() const { return m_V; }
    const V2d &principalStrains() const { return m_strain; }

    size_t tensionState() const {
        if (m_strain[0] < 0)     return 0; // compression in both directions
        if (m_excessStrain < 0)  return 1; //     tension in one  direction
        return 2;                          //     tension in both directions
    }

    // Legacy interface for compatibility with incompressible neo-Hookean material
    // from Skouras 2014 (which uses a Poisson's ratio of 0.5 and has just has
    // a single "stiffness" parameter corresponding to mu / 2)
    void setStiffness(Real val) {
        this->m_E = 6.0 * val;
        this->m_nu = 0.5;
    }
    Real stiffness() const { return this->m_E / 6.0; }
    void setRelaxedStiffnessEpsilon(Real val) { relaxedStiffnessEps = val; }
    void setRelaxationEnabled(bool enabled)       { relaxationEnabled = enabled; }
    bool getRelaxationEnabled()             const { return relaxationEnabled; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
protected:
    Real m_E = 1.0;   // Young's modulus
    Real m_nu = 0.5;  // Poisson's ratio
    Real m_lambda_div_nu = 0.0;

    ////////////////////////////////////////////////////////////////////////////
    // Deformed state quantities
    ////////////////////////////////////////////////////////////////////////////
    M3d m_U;
    M2d m_V;
    V2d m_sigma, m_strain; // principal stretches (sigma) and strains
    Real m_excessStrain;
    M32d m_F;
};

#endif /* end of include guard: ISOCRLETENSIONFIELD_HH */
