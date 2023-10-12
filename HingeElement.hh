////////////////////////////////////////////////////////////////////////////////
// HingeElement.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Implements a generic hinge element whose energy is a nonlinear function
// of a dihedral angle specified via the HingeElement
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
////////////////////////////////////////////////////////////////////////////////
#ifndef HINGEELEMENT_HH
#define HINGEELEMENT_HH

#include <cmath>
#include <Eigen/Dense>
#include <array>
#include <MeshFEM/Elements/DihedralAngle.hh>
#include <MeshFEM/EnergyDensities//EnergyTraits.hh>

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
    using MaterialProperties = Real; // Bending stiffness

    void configure(const RestState &X, Real theta, MaterialProperties k, EvalLevel elevel = EvalLevel::Full) {
        m_theta = theta;
        m_theta_bar = X.theta;
        m_weight = k * X.e_len / X.h_bar;
    }

    Real   energy() const { return 0.5 * (m_theta - m_theta_bar) * (m_theta - m_theta_bar) * m_weight; }
    Real gradient() const { return (m_theta - m_theta_bar) * m_weight; }
    Real  hessian() const { return m_weight; }

private:
    Real m_theta, m_theta_bar, m_weight;
};

template<class _Real>
struct PanelizationHingeEnergy {
    using Real = _Real;
    struct RestState {
        RestState(const elements::DihedralAngle<Real> &/* da */) { }
    };
    struct MaterialProperties { Real delta = 0.01, stiffness = 1; };
    void configure(const RestState /* X */, Real theta, const MaterialProperties &m, EvalLevel elevel = EvalLevel::Full) {
        m_theta = theta;
        m_k = m.stiffness;
        m_delta = m.delta;
    }

    Real   sigmoid(Real x) const { Real x2 = x*x; return x2 / (x2 + m_delta); }
    Real  dsigmoid(Real x) const { Real x2 = x*x; return 2 * m_delta * x / ((x2 + m_delta) * (x2 + m_delta)); }
    Real ddsigmoid(Real x) const { Real x2 = x*x; return 2 * m_delta * (m_delta - 3 * x2) / ((x2 + m_delta) * (x2 + m_delta) * (x2 + m_delta)); }

    Real   energy() const { return m_k *   sigmoid(m_theta); }
    Real gradient() const { return m_k *  dsigmoid(m_theta); }
    Real  hessian() const { return m_k * ddsigmoid(m_theta); }

private:
    Real m_k, m_delta, m_theta;
};

template<class HingeEnergy>
struct HingeElement {
    using Real      = typename HingeEnergy::Real;
    using DA        = elements::DihedralAngle<Real>;
    using RestState = typename HingeEnergy::RestState;
    using Vars      = typename DA::StencilPoints;

    using Gradient = typename DA::Gradient;
    using Hessian  = typename DA::Hessian;
    using MProps   = typename HingeEnergy::MaterialProperties;

    HingeElement(const Vars &X, const MProps &m = MProps())
        : m_theta(X), m_restState(m_theta) {
        material = m;
    }

    void setDeformedConfiguration(const Vars &x, EvalLevel elevel = EvalLevel::Full) {
        m_theta.configure(x);
        m_he.configure(m_restState, m_theta.value(), material, elevel); 
    }

    void setRestConfiguration(const Vars &X) {
        m_theta.configure(X);
        m_restState = m_restState(m_theta);
    }

    Real       energy() const { return m_he.energy(); }
    Gradient gradient() const { return m_he.gradient() * m_theta.gradient(); }
    Hessian   hessian() const {
        Gradient gradTheta = m_theta.gradient();
        return gradTheta * m_he.hessian() * gradTheta.transpose() + m_he.gradient() * m_theta.hessian();
    }

    MProps material;

private:
    HingeEnergy m_he;
    elements::DihedralAngle<Real> m_theta;
    RestState m_restState;
};

#endif /* end of include guard: HINGEELEMENT_HH */
