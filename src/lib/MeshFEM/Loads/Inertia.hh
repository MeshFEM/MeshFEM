////////////////////////////////////////////////////////////////////////////////
// InertiaLoad.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements an inertia load that can be applied to a dynamic simulation using
//  both Backward Euler, and implicit newmark integrators.
*/
//  Author:  Haleh Mohammadian (halehm), haleh.mohammadian@gmail.com
//  Created:  07/11/2023 13:23:12
////////////////////////////////////////////////////////////////////////////////
#ifndef INERTIALOAD_HH
#define INERTIALOAD_HH

#include "Load.hh"
#include <MeshFEMCore/GlobalBenchmark.hh>

namespace MeshFEM {

namespace Loads {
    template<class Object>
    struct Inertia : public ObjectSpecificLoad<Object> {
        using Real = typename Object::Real;
        using Base = ObjectSpecificLoad<Object>;
        using ST   = std::weak_ptr<const Object>;
        using VXd  = typename Object::VXd;
        using Base::getObj;

        Inertia(const ST &obj, const bool lumpedMass)
            : Base(obj), m_useLumpedMass(lumpedMass)
        {
            m_updateCache();
            xhat = getObj().getVars(); // Needed for evaluating the initial barrier stiffness...

            // The sparsity pattern of the inertia term Hessian (i.e., the mass matrix)
            // is known to be a subset of the elastic object's Hessian sparsity pattern,
            // enabling an optimization of the full sparsity pattern construction.
            this->suppressSparsity = true;
        }

        VXd disp_x() const { return getObj().getVars() - xhat; }

        void setXhat(const VXd &x) { xhat = x;}

        virtual Real energy() const override {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Loads.Inertia.energy");
            return 0.5 * weight * evalQuadraticForm(disp_x());
        }

        // Gradient with respect to the deformed state
        virtual VXd grad_x() const override {
            if (m_useLumpedMass) return weight * M_lumped.asDiagonal() * disp_x();
            return weight * M_full.apply(disp_x());
        }

        // Gradient with respect to the rest state
        virtual VXd grad_X() const override { throw std::runtime_error("TODO"); }

        virtual void accumulateHessian(Real w, NewtonHessian &H , bool /* projectionMask */ = true) const override {
            if (m_useLumpedMass) H.H_ss->addDiag((w * weight * M_lumped).eval());
            else                 H.H_ss->addWithSubSparsityFast(*(M_full.H_ss), w * weight);
        }

        // The Hessian (mass matrix) has an identical sparsity pattern to the
        // elastic object's Hessian.
        // Note that technically `Loads::hessianSparsityPattern` need only
        // report *additional* non-zeros beyond those present in the elastic object's
        // Hessian sparsity pattern (none in this case).
        // However, to support user code that might call `this->hessian()` for debugging,
        // we return the full mass matrix sparsity pattern here.
        // This won't slow down the NewtonMultiobjectiveProblem sparsity pattern
        // since we've set `suppressSparsity = true` in the constructor.
        virtual NewtonHessian hessianSparsityPattern() const override {
            return getObj().hessianSparsityPattern();
        }

        size_t getMassMatrixID() const { return m_massMatrixID; }

        Real evalQuadraticForm(Eigen::Ref<const VXd> x) const {
            if (m_useLumpedMass) return x.dot(M_lumped.asDiagonal() * x);
            return M_full.H_ss->evalQuadraticForm(x);
        }

        bool usingLumpedMass() const { return m_useLumpedMass; }

        Real weight = 1.0;
        VXd  xhat;

        NewtonHessian M_full;
        VXd M_lumped;

      private:
        virtual void m_stateUpdated(typename Base::VM vmask) override {
            if ((vmask == Base::VM::Defo && getObj().hasVariableMassMatrix()) ||
                (vmask == Base::VM::Rest)) {
                m_updateCache();
            }
        }

        void m_updateCache() {
            if (m_useLumpedMass) M_lumped = getObj().lumpedMass(/* updatedParametrization = */ false);
            else                 M_full   = getObj().massMatrix(/* updatedParametrization = */ false);
            ++m_massMatrixID;
        }

        const bool m_useLumpedMass;

        size_t m_massMatrixID = 0; // monotonically increasing identity tag for the current mass
                                   // matrix (used to invalidate factorization in DynamicSimulator).
    };

} // namespace Loads

} // namespace MeshFEM

#endif /* end of include guard: INERTIALOAD_HH */
