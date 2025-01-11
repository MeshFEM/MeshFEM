////////////////////////////////////////////////////////////////////////////////
// InertiaLoad.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements a inertia load that can be applied to a
//  dynamic simulation using both Backward Euler, and implicit newmark integrators
*/
//  Author:  Haleh Mohammadian (halehm), haleh.mohammadian@gmail.com
//  Created:  07/11/2023 13:23:12
////////////////////////////////////////////////////////////////////////////////
#ifndef INERTIALOAD_HH
#define INERTIALOAD_HH

#include "Load.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include <memory>


// TO DO
// Needs to check for newtonObjectiveTerm update in Load inheritance
namespace Loads {
    template<class Object, template<typename T> class EOStoragePolicy = EOStoragePolicyWeakPtr>
    struct Inertia;

    template<class Object, template<typename T> class EOStoragePolicy>
    struct Inertia : public ObjectSpecificLoad<Object, EOStoragePolicy> {
        using Real = typename Object::Real;
        using Base = ObjectSpecificLoad<Object, EOStoragePolicy>;
        using ST   = typename Base::SP::StorageType;
        using VXd  = typename Object::VXd;
        using Base::getObj;

        Inertia(const ST &obj, const bool lumpedMass)
            : Base(obj), m_useLumpedMass(lumpedMass)
        {
            m_updateCache();
            xhat = getObj().getVars(); // Needed for evaluating the initial barrier stiffness...
        }

        size_t numVars() const { return getObj().numVars(); }

        VXd disp_x() const { return getObj().getVars() - xhat; }

        void setXhat(const VXd &x) { xhat = x;}

        virtual Real energy() const override {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Loads.Inertia.energy");
            return 0.5 * weight * evalQuadraticForm(disp_x());
        }

        // Gradient with respect to the deformed state
        virtual VXd grad_x() const override {
            if (m_useLumpedMass) return weight * M.data().asDiagonal() * disp_x();
            return weight * M_full.applyTransposeParallel(disp_x());
        }

        // Gradient with respect to the rest state
        virtual VXd grad_X() const override { throw std::runtime_error("TODO"); }

        virtual void accumulateHessian(Real w, SuiteSparseMatrix & H , bool /* projectionMask */ = true) const override {
            if (m_useLumpedMass) H.addDiag(w * weight * M.data());
            else H.addWithSubSparsityFast(M, w * weight);
        }

        virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const override {
            SuiteSparseMatrix Hsp = M;
            Hsp.fill(val);
            return Hsp;
        }

        size_t getMassMatrixID() const { return m_massMatrixID; }

        Real evalQuadraticForm(Eigen::Ref<const VXd> x) const {
            if (m_useLumpedMass) return x.dot(M.data().asDiagonal() * x);
            return M.evalQuadraticForm(x);
        }

        Real weight = 1.0;
        VXd  xhat;

        SuiteSparseMatrix M, M_full;

      private:
        virtual void m_stateUpdated(typename Base::VM vmask) override {
            if ((vmask == Base::VM::Defo && getObj().hasVariableMassMatrix()) ||
                (vmask == Base::VM::Rest)) {
                m_updateCache();
            }
        }

        void m_updateCache() {
            if (m_useLumpedMass) {
                M.m = M.n = numVars();
                M.setIdentity(/* preserveSparsity = */ false);
                M.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
                getObj().massMatrix(M, /* updatedParametrization = */ false, m_useLumpedMass);
            }
            else {
                M = getObj().massMatrix(/* updatedParametrization = */ false, m_useLumpedMass);
                M_full = M.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::NONE);
            }
            ++m_massMatrixID;
        }

        const bool m_useLumpedMass;

        size_t m_massMatrixID = 0; // monotonically increasing identity tag for the current mass
                                   // matrix (used to invalidate factorization in DynamicSimulator).
    };

} // namespace Loads

#endif /* end of include guard: INERTIALOAD_HH */
