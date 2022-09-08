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
    template<class Object, template<typename T> class EOStoragePolicy = EOStoragePolicyWeakPtr>
    struct Gravity;

    namespace detail {
        template<class Object>
        struct GravityLoadVector {
            using VXd  = typename Object::VXd;
            static constexpr size_t N   = 3;
            static constexpr size_t K   = Object::K;
            static constexpr size_t Deg = Object::Deg;
            static VXd compute(const Gravity<Object> &g) {
                VXd result;
                result.setZero(g.numVars());
                const auto &m = g.getObj().mesh();
                typename Object::Mesh::ElementData::Phis phiIntegrals;
                auto integratedPhis = integratedShapeFunctions<Deg, K>();
                for (const auto e : m.elements()) {
                    for (const auto n : e.nodes()) {
                        result.template segment<3>(3 * n.index()) +=
                            g.get_g() * (integratedPhis[n.localIndex()] * e->volume());
                    }
                }
                result *= -g.get_rho();
                return result;
            }
        };
    }

    template<class Object, template<typename T> class EOStoragePolicy>
    struct Gravity : public ObjectSpecificLoad<Object, EOStoragePolicy> {
        using Real = typename Object::Real;
        using Base = ObjectSpecificLoad<Object, EOStoragePolicy>;
        using ST   = typename Base::SP::StorageType;
        using VXd  = typename Object::VXd;
        using V3d  = Eigen::Matrix<Real, 3, 1>;

        Gravity(const ST &obj, Real rho, const V3d &g = V3d(0.0, 0.0, 9.80635))
            : Base(obj), m_rho(rho), m_g(g) {
            m_updateCache();
        }

        void set_rho(Real rho) { m_rho = rho; m_updateCache(); }
        Real get_rho()   const { return m_rho; }

        void set_g(V3d g)      { m_g = g; m_updateCache(); }
        V3d  get_g()     const { return m_g; }

        size_t numVars() const { return Base::getObj().numVars(); }
        virtual Real energy() const override {
            return m_grad.dot(Base::getObj().getVars());
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
            const size_t nv = numVars();
            TripletMatrix<> Hsp(nv, nv);
            Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
            return SuiteSparseMatrix(Hsp);
        }

    private:
        virtual void m_stateUpdated(typename Base::VM vmask) override {
            if (vmask == Base::VM::Rest) m_updateCache();
        }

        Real m_rho;
        V3d  m_g; // Gravitational acceleration vector

        void m_updateCache() {
            m_grad = detail::GravityLoadVector<Object>::compute(*this);
        }

        VXd m_grad;
    };

} // namespace Loads

#endif /* end of include guard: GRAVITY_HH */
