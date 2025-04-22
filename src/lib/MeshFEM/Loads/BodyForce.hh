////////////////////////////////////////////////////////////////////////////////
// BodyForce.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A force per unit volume (3D) or area (2D) applied to the interior of an
//  elastic object.
//
//  potential: V(x) = -int_X x . f(X) dX = - int_X x . (f_j phi_j) dX
//  gradient:  g_i  = -int_X phi_i . f(X) dX = - int_X phi_i f_j phi_j dX = -M_ij f_j
//  Hessian:   0
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/14/2025 21:31:17
*///////////////////////////////////////////////////////////////////////////////
#ifndef BODYFORCE_HH
#define BODYFORCE_HH

#include "Load.hh"
#include "../ElasticObject.hh"

namespace Loads {

template<class _Real>
struct BodyForce : public ObjectSpecificLoad<ElasticObject<_Real>> {
    using EO = ElasticObject<_Real>;
    using Base = ObjectSpecificLoad<EO>;

    using Real = typename Base::Real;
    using VXd  = typename Base::VXd;
    using MXd  = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // Row-major for proper flattened order
    using Base::getObj;

    BodyForce(std::weak_ptr<const EO> obj)
        : Base(obj) {
        m_updateCache();
    }

    virtual Real energy() const override {
        return m_grad.dot(getObj().getVars());
    }

    // Derivative with respect to deformed configuration
    virtual VXd grad_x() const override {
        return m_grad;
    }

    // Derivative with respect to rest configuration (for shape optimization)
    virtual VXd grad_X() const override {
        // Do we really want the total force to decrease when the boundary shrinks??
        throw std::runtime_error("Unimplemented");
    }

    // Potential is linear with respect to the deformed state
    virtual void accumulateHessian(Real weight, NewtonHessian & /* H */, bool /* projectionMask */ = true) const override { }
    virtual NewtonHessian hessianSparsityPattern() const override { return NewtonHessian(); }

    void setNodalForceDensity(MXd f /* pass-by-value due to copy inside */) { m_nodalForceDensity = f; m_updateCache(); }
    const MXd &getNodalForceDensity() const { return m_nodalForceDensity; }

    virtual ~BodyForce() { }
private:
    MXd m_nodalForceDensity;
    VXd m_grad;

    virtual void m_stateUpdated(typename Base::VM vmask) override {
        if (vmask == Base::VM::Rest) m_updateCache();
    }

    void m_updateCache() {
        m_grad.setZero(getObj().numVars());
        if (m_nodalForceDensity.size() == 0) return; // Empty force density interpreted as zero force.
        auto M = getObj().massMatrix(/* updatedParametrization */ false);

        Eigen::Map<VXd> f(m_nodalForceDensity.data(), m_nodalForceDensity.size());
        if (M.numVars() != f.rows()) { throw std::runtime_error("Invalid size of force density: " + std::to_string(M.numVars()) + " != " + std::to_string(f.rows())); }

        // Note: mass matrix `M` includes the physical mass density factor,
        // `rho`, which we don't want to include here.
        // (We really just want the standard FEM mass matrix: M_ij = int phi_i * phi_j dX)
        // TODO: support for non-uniform mass densities.
        m_grad = -M.apply(f) / getObj().getMassDensity();
    }
};

}

#endif /* end of include guard: BODYFORCE_HH */
