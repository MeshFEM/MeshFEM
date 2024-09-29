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

#include <Eigen/Dense>
#include <MeshFEM/Elements/DihedralAngle.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>
#include <MeshFEM/MeshEnergy.hh>

#include "ElementBase.hh"

template<class HingeEnergy>
struct HingeElement;

template<class HingeEnergy>
struct ElementTraits<HingeElement<HingeEnergy>> {
    using Material = typename HingeEnergy::MaterialProperties;
};

template<class HingeEnergy>
struct HingeElement : public ElementBase<HingeElement<HingeEnergy>> {
    using Base      = ElementBase<HingeElement<HingeEnergy>>;
    using Real      = typename HingeEnergy::Real;
    using DA        = elements::DihedralAngle<Real>;
    using RestState = typename HingeEnergy::RestState;
    using LocalVars = typename DA::StencilPoints;
    using Gradient  = typename DA::Gradient;
    using Hessian   = typename DA::Hessian;
    using Material  = typename Base::Material;

    static std::string name() { return std::string("HingeElement<") + HingeEnergy::name() + ">"; }

    static constexpr bool CachesDeformedQuantities = true;

    template<class Mesh>
    HingeElement(size_t ei, const Mesh &/* m */, const LocalVars &x, MaterialAssignment<Material> &materials)
        : Base(ei, materials), m_theta(x), m_restState(m_theta) {
        m_he.configure(m_restState, m_theta.value());
    }

    void setDeformedConfiguration(const LocalVars &x, EvalLevel elevel = EvalLevel::Full) {
        m_theta.configure(x);
        m_he.configure(m_restState, m_theta.value(), elevel);
    }

    void setRestConfiguration(const LocalVars &X) {
        m_theta.configure(X);
        m_restState = m_restState(m_theta);
    }

    Real theta() const { return m_theta.value(); }

    Real       energy(      ) const { return m_he.energy(Base::material()); }
    Gradient gradient(Real w) const { return (m_he.gradient(Base::material()) * w) * m_theta.gradient(); }
    Hessian   hessian(Real w, bool /* projectionMask */ = false) const {
        Gradient gradTheta = m_theta.gradient();
        return gradTheta * (m_he.hessian(Base::material()) * w) * gradTheta.transpose() + (w * m_he.gradient(Base::material())) * m_theta.hessian();
    }

private:
    HingeEnergy m_he;
    elements::DihedralAngle<Real> m_theta;
    RestState m_restState;
};

template<class HingeEnergy>
using HingeMeshEnergy = MeshEnergy<FEMMesh<2, 1, Vector3D>, NodalVars<3>, TriFlapStencil, HingeElement<HingeEnergy>>;

#endif /* end of include guard: HINGEELEMENT_HH */
