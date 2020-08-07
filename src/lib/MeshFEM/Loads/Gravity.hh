////////////////////////////////////////////////////////////////////////////////
// Gravity.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements a gravitational potential energy that can be applied to a
//  volumetric ElasticObject or an ElasticSheet.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/05/2020 10:23:12
////////////////////////////////////////////////////////////////////////////////
#ifndef GRAVITY_HH
#define GRAVITY_HH

#include "Load.hh"
#include <MeshFEM/GaussQuadrature.hh>

namespace Loads {

template<class Object>
struct Gravity : public Load<3, typename Object::Real> {
    using Real = typename Object::Real;
    using VXd  = typename Object::VXd;
    using V3d  = Eigen::Matrix<Real, 3, 1>;
    static constexpr size_t N   = 3;
    static constexpr size_t K   = Object::K;
    static constexpr size_t Deg = Object::Deg;

    Gravity(const Object &obj, Real rho, const V3d &g = V3d(0.0, 0.0, 9.80635))
        : m_obj(obj), m_rho(rho), m_g(g) {
        restStateUpdated();
    }

    void set_rho(Real rho) { m_rho = rho; m_updateCache(); }
    Real get_rho(Real rho) { return m_rho; }

    virtual void deformedStateUpdated() override { /* Gravity force is const wrt. x */ }

    virtual void restStateUpdated() override { m_updateCache(); }

    virtual Real energy() const override {
        return m_grad.dot(m_obj.getVars());
    }

    // Gradient with respect to the deformed state
    virtual VXd grad_x() const override {
        return m_grad;
    }

    // Gradient with respect to the rest state
    virtual VXd grad_X() const override {
        throw std::runtime_error("TODO");
    }

    // Gravity is linear ==> Hessian is zero.
    virtual void hessian(SuiteSparseMatrix& /* H */) const override { }

    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const override {
        const size_t nv = m_obj.numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
        return SuiteSparseMatrix(Hsp);
    }

private:
    const Object &m_obj;
    Real m_rho;
    V3d  m_g; // Gravitational acceleration vector

    void m_updateCache() {
        m_grad.setZero(m_obj.numVars());
        const auto &m = m_obj.mesh();
        Interpolant<Real, K, Deg> phi;
        phi = 0.0;
        for (const auto &e : m.elements()) {
            for (const auto &n : e.nodes()) {
                phi[n.localIndex()] = 1.0;
                m_grad.template segment<3>(3 * n.index()) += m_g; //  * phi.integrate(e->volume());
                phi[n.localIndex()] = 0.0;
            }
        }

        m_grad *= -m_rho;
    }

    VXd m_grad;
};

} // namespace Loads

#endif /* end of include guard: GRAVITY_HH */
