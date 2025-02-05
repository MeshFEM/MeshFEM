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
    struct Gravity;

    namespace detail {
        template<class Object>
        struct GravityLoadVector {
            using VXd  = typename Object::VXd;
            static constexpr size_t N   = Object::N;
            static constexpr size_t K   = Object::K;
            static constexpr size_t Deg = Object::Deg;
            static VXd compute(const Gravity<Object> &g) {
                const auto &o = g.getObj();
                auto M = o.massMatrix(/* updatedParametrization */ false, /* useLumpedMass */ false);
                VXd neg_g_rep = (-g.get_g()).replicate(M.n / N, 1); // Assumes all variables are nodal displacements... (like ElasticSolid)
                return M.apply(neg_g_rep);
            }
        };
    }

    template<class Object>
    struct Gravity : public ObjectSpecificLoad<Object> {
        using Real = typename Object::Real;
        using Base = ObjectSpecificLoad<Object>;
        using ST   = typename Base::EOStorageType;
        static constexpr size_t N = Object::N;
        using VXd  = typename Object::VXd;
        using VNd  = Eigen::Matrix<Real, N, 1>; // ElasticSolid has the information of N
        using Base::getObj;

        static constexpr VNd default_gravity() {
            VNd result = VNd::Zero();
            static_assert(N == 2 || N == 3, "Gravity load only implemented in 2D and 3D");
            result[1] = -9.80635; // Negative y direction.
            return result;
        }

        Gravity(const ST &obj, const VNd &g = default_gravity())
            : Base(obj), m_g(g) { m_updateCache(); }

        void set_g(VNd g)      { m_g = g; m_updateCache(); }
        VNd  get_g()     const { return m_g; }

        virtual Real energy() const override { return m_grad.dot(Base::getObj().getVars()); }

        // Gradient with respect to the deformed state
        virtual VXd grad_x() const override { return m_grad; }

        // Gradient with respect to the rest state
        virtual VXd grad_X() const override { throw std::runtime_error("TODO"); }

        // Gravity is linear ==> Hessian is zero.
        virtual void accumulateHessian(Real /* weight */, NewtonHessian &/* H */, bool /* projectionMask */ = true) const override { }
        virtual NewtonHessian hessianSparsityPattern() const override { return NewtonHessian(); }

    private:
        virtual void m_stateUpdated(typename Base::VM vmask) override {
            if (vmask == Base::VM::Rest) m_updateCache();
        }

        VNd  m_g; // Gravitational acceleration vector

        void m_updateCache() {
            m_grad = detail::GravityLoadVector<Object>::compute(*this);
        }

        VXd m_grad;
    };

} // namespace Loads

#endif /* end of include guard: GRAVITY_HH */
