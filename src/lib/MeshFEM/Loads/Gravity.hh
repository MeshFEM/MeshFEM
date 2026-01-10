////////////////////////////////////////////////////////////////////////////////
// Gravity.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements a gravitational potential energy that can be applied to an
//  elastic object.
//
//  This is a simple wrapper around `BodyForce` since a gravity acceleration
//  vector `g` simply induces a body force density `f = rho * g`, where `rho` is
//  the mass density of the object.
//
//  Currently the mass density is assumed to be constant throughout any design
//  optimization; otherwise it must be defined as a *nodal* field
//  (since `BodyForce` works with nodal force densities) and accounted
//  for during sensitivity analysis.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/05/2020 10:23:12
////////////////////////////////////////////////////////////////////////////////
#ifndef GRAVITY_HH
#define GRAVITY_HH

#include "BodyForce.hh"

namespace Loads {
    template<class Object>
    struct Gravity;

    template<class Object>
    struct Gravity : public BodyForce<Object> {
        using Real = typename Object::Real;
        using Base = BodyForce<Object>;
        using ST   = typename Base::EOStorageType;
        static constexpr size_t N = Object::N;
        using VXd  = typename Object::VXd;
        using VNd  = Eigen::Matrix<Real, N, 1>; // ElasticSolid has the information of N

        static constexpr VNd default_gravity() {
            VNd result = VNd::Zero();
            static_assert(N == 2 || N == 3, "Gravity load only implemented in 2D and 3D");
            result[1] = -9.80635; // Negative y direction.
            return result;
        }

        Gravity(const ST &obj, const VNd &g = default_gravity()) : Base(obj) { set_g(g); }

        void set_g(VNd g) {
            m_g = g;
            const auto &o = this->getObj();

            VXd f = o.getMassDensity() * (m_g).replicate(o.numNodes(), 1);

            Base::setNodalForceDensity(f);
        }
        const VNd &get_g() const { return m_g; }

    private:
        VNd  m_g; // Gravitational acceleration vector
    };

} // namespace Loads

#endif /* end of include guard: GRAVITY_HH */
