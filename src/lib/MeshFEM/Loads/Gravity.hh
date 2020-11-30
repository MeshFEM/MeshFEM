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

    Gravity(std::weak_ptr<const Object> obj, Real rho, const V3d &g = V3d(0.0, 0.0, 9.80635))
        : m_obj(obj), m_rho(rho), m_g(g) {
        m_updateCache();
        m_callbackID = getObj().registerRestConfigUpdateCallback([this]() { m_updateCache(); });
    }

    void set_rho(Real rho) { m_rho = rho; m_updateCache(); }
    Real get_rho()         { return m_rho; }

    virtual Real energy() const override {
        return m_grad.dot(getObj().getVars());
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
    virtual void hessian(SuiteSparseMatrix& /* H */, bool /* projectionMask */ = true) const override { }

    virtual SuiteSparseMatrix hessianSparsityPattern(Real /* val */ = 0.0) const override {
        const size_t nv = getObj().numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
        return SuiteSparseMatrix(Hsp);
    }

    virtual ~Gravity() {
        if (auto o = m_obj.lock())
            o->deregisterRestConfigUpdateCallback(m_callbackID);
    }

private:
    std::weak_ptr<const Object> m_obj;
    Real m_rho;
    V3d  m_g; // Gravitational acceleration vector
    int m_callbackID;

    const Object &getObj() const {
        if (auto o = m_obj.lock()) return *o;
        throw std::runtime_error("Elastic object was destroyed");
    }

    void m_updateCache() {
        m_grad.setZero(getObj().numVars());
        const auto &m = getObj().mesh();
        typename Object::Mesh::ElementData::Phis phiIntegrals;
        auto integratedPhis = integratedShapeFunctions<Deg, K>();
        for (const auto e : m.elements()) {
            for (const auto n : e.nodes()) {
                m_grad.template segment<3>(3 * n.index()) +=
                    m_g * (integratedPhis[n.localIndex()] * e->volume());
            }
        }

        m_grad *= -m_rho;
    }

    VXd m_grad;
};

} // namespace Loads

#endif /* end of include guard: GRAVITY_HH */
