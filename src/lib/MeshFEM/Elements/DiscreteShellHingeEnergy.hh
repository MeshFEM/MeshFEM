////////////////////////////////////////////////////////////////////////////////
// DiscreteShellHingeEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements the discrete shell hinge energy from [Grinspun 2003]
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  11/05/2023 23:00:24
*///////////////////////////////////////////////////////////////////////////////
#ifndef DISCRETESHELLHINGEENERGY_HH
#define DISCRETESHELLHINGEENERGY_HH

#include <MeshFEM/Elements/HingeElement.hh>

namespace MeshFEM {

template<class _Real>
struct DiscreteShellHingeEnergy {
    using Real = _Real;
    struct RestState {
        RestState(const elements::DihedralAngle<Real> &da) {
            theta = da.value();
            e_len = da.hingeEdgeLen();
            h_bar = da.avgHeight() / 3; // see [Grinspun 2003], below (2)
        }
        Real theta, e_len, h_bar;
    };

    static constexpr const char *name() { return "DiscreteShell"; }

    struct MaterialProperties : public MaterialBase {
        void setYoungPoisson(Real E, Real nu) { throw std::runtime_error("TODO"); }
        Real stiffness = 1;
        bool include_inv_hbar_weight = true;

    };

    Real weightedStiffness(const MaterialProperties &m) const {
        Real result = m.stiffness * m_e_len;
        if (m.include_inv_hbar_weight) result /= m_h_bar;
        return result;
    }

    void configure(const RestState &X, Real theta, EvalLevel elevel = EvalLevel::Full) {
        m_theta = theta;
        m_theta_bar = X.theta;
        m_h_bar = X.h_bar;
        m_e_len = X.e_len;
    }

    Real   energy(const MaterialProperties &m) const { return 0.5 * (m_theta - m_theta_bar) * (m_theta - m_theta_bar) * weightedStiffness(m); }
    Real gradient(const MaterialProperties &m) const { return (m_theta - m_theta_bar) * weightedStiffness(m); }
    Real  hessian(const MaterialProperties &m) const { return weightedStiffness(m); }

private:
    Real m_theta, m_theta_bar, m_e_len, m_h_bar, m_stiffness;
};

template<class _Real>
using DiscreteShellHingeMeshEnergy = HingeMeshEnergy<DiscreteShellHingeEnergy<_Real>>;

} // namespace MeshFEM

#endif /* end of include guard: DISCRETESHELLHINGEENERGY_HH */
