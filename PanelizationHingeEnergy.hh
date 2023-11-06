#ifndef PANELIZATIONHINGEENERGY_HH
#define PANELIZATIONHINGEENERGY_HH

#include <MeshFEM/Elements/HingeElement.hh>

template<class _Real>
struct PanelizationHingeEnergy {
    using Real = _Real;

    struct RestState {
        RestState(const elements::DihedralAngle<Real> &/* da */) { }
    };

    struct MaterialProperties : public MaterialBase {
        Real delta = 0.01, stiffness = 1;
    };

    static constexpr const char *name() { return "Panelization"; }

    void configure(const RestState /* X */, Real theta, EvalLevel elevel = EvalLevel::Full) { m_theta = theta; }

    Real   sigmoid(Real x, Real d) const { Real x2 = x*x; return x2 / (x2 + d); }
    Real  dsigmoid(Real x, Real d) const { Real x2 = x*x; return 2 * d * x / ((x2 + d) * (x2 + d)); }
    Real ddsigmoid(Real x, Real d) const { Real x2 = x*x; return 2 * d * (d - 3 * x2) / ((x2 + d) * (x2 + d) * (x2 + d)); }

    Real   energy(const MaterialProperties &m) const { return m.stiffness *   sigmoid(m_theta, m.delta); }
    Real gradient(const MaterialProperties &m) const { return m.stiffness *  dsigmoid(m_theta, m.delta); }
    Real  hessian(const MaterialProperties &m) const { return m.stiffness * ddsigmoid(m_theta, m.delta); }

private:
    Real m_theta;
};

#endif /* end of include guard: PANELIZATIONHINGEENERGY_HH */
