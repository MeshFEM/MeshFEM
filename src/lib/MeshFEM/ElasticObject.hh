////////////////////////////////////////////////////////////////////////////////
// ElasticObject.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Interface for a generic ElasticObject along with a primitive system for
//  notifying other objects when the deformed or rest configuration of the
//  ElasticObject updates.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/12/2020 09:59:34
////////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICOBJECT_HH
#define ELASTICOBJECT_HH
#include "Types.hh"
#include <map>
#include <cstdlib>
#include <functional>

#include "FieldSampler.hh"
#include "SparseMatrices.hh"
#include "newton_optimizer/MultiobjectiveProblem.hh"

////////////////////////////////////////////////////////////////////////////
// Generic variables:
// An optimization can include variables parametrizing the deformed
// configuration, the rest configuration, or both. The generic interface
// here allows the user to specify which variables why wish to access via
// a `VariableMask`.
////////////////////////////////////////////////////////////////////////////
enum class VariableMask { Defo, Rest, All };

template<class _Real>
struct ElasticObject : public NewtonObjectiveTerm, public NewtonVarsBase {
    using Real = _Real;
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using CSCMat = CSCMatrix<SuiteSparse_long, _Real>;
    using NotificationCB = std::function<void()>;
    using VariableMask = VariableMask;

    ////////////////////////////////////////////////////////////////////////////
    // Wrapper methods implementing the NewtonVarsManager interface.
    ////////////////////////////////////////////////////////////////////////////
    size_t numVars() const override { return numVars(VariableMask::Defo); }
    void setVars(const VXd &vars) override { setVars(vars, VariableMask::Defo); }
    VXd getVars() const override { return getVars(VariableMask::Defo); }

    ////////////////////////////////////////////////////////////////////////////
    // Wrapper methods implementing the NewtonObjectiveTerm interface
    ////////////////////////////////////////////////////////////////////////////
    Real objective() const override { return energy(); }

    virtual void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        return accumulateGradient(weight, g, freshIterate, VariableMask::Defo);
    }

    using NewtonObjectiveTerm::hessian; // Don't shadow the `hessian` convenience method
    virtual void accumulateHessian(Real weight, SuiteSparseMatrix &H, bool projectionMask) const override {
        return accumulateHessian(weight, H, projectionMask, VariableMask::Defo);
    }

    virtual CSCMat hessianSparsityPattern(Real val = 0.0) const override {
        return hessianSparsityPattern(0.0, VariableMask::Defo);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Custom interface
    ////////////////////////////////////////////////////////////////////////////
    size_t numVars(VariableMask vmask) const {
        if (vmask == VariableMask::Defo) return numDefoVars();
        if (vmask == VariableMask::Rest) return numRestVars();
        if (vmask == VariableMask::All ) return numDefoVars() + numRestVars();
        throw std::runtime_error("Unknown variable type");
    }

    void setVars(const Eigen::Ref<const VXd> &vars, VariableMask vmask) {
        if (size_t(vars.size()) != numVars(vmask)) throw std::runtime_error("Input vars size doesn't match vmask");
        if ((vmask == VariableMask::Defo) || (vmask == VariableMask::All))
            setDefoVars(vars.head(numDefoVars()));
        if ((vmask == VariableMask::Rest) || (vmask == VariableMask::All))
            setRestVars(vars.tail(numRestVars()));
    }

    VXd getVars(VariableMask vmask) const {
        if (vmask == VariableMask::Defo) return getDefoVars();
        if (vmask == VariableMask::Rest) return getRestVars();
        if (vmask == VariableMask::All ) {
            VXd result(numDefoVars() + numRestVars());
            result.head(numDefoVars()) = getDefoVars();
            result.tail(numRestVars()) = getRestVars();
            return result;
        }
        throw std::runtime_error("Unknown variable type");
    }

    ////////////////////////////////////////////////////////////////////////////
    // Deformation and rest state variables
    ////////////////////////////////////////////////////////////////////////////
    virtual size_t numDefoVars() const = 0;
    virtual size_t numRestVars() const = 0;

    virtual VXd getDefoVars() const = 0;
    virtual VXd getRestVars() const = 0;

    void setDefoVars(const Eigen::Ref<const VXd> &vars) { if (size_t(vars.size()) != numDefoVars()) throw std::runtime_error("Size of input vars doesn't match numDefoVars!");
        m_setDefoVars(vars); m_defoConfigUpdated(); 
    }
    void setRestVars(const Eigen::Ref<const VXd> &vars) { if (size_t(vars.size()) != numRestVars()) throw std::runtime_error("Size of input vars doesn't match numRestVars!");
        m_setRestVars(vars); m_restConfigUpdated(); 
    }

    ////////////////////////////////////////////////////////////////////////////
    // Energy and derivatives
    ////////////////////////////////////////////////////////////////////////////
    virtual Real  energy() const = 0;
    virtual void accumulateGradient(Real weight, VXd &g, bool updatedParametrization, VariableMask vmask) const = 0;
    virtual void accumulateHessian(Real weight, CSCMat &Hout, bool projectionMask, VariableMask vmask) const = 0;
    virtual CSCMat hessianSparsityPattern(Real val, VariableMask vmask) const = 0;

    // Convenience method
    VXd gradient(bool updatedParametrization, VariableMask vmask = VariableMask::Defo) const {
        VXd g = VXd::Zero(numVars());
        accumulateGradient(1.0, g, updatedParametrization, vmask);
        return g;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Optional parts of the interface
    ////////////////////////////////////////////////////////////////////////////
    // Update parametrization of the system's DoFs. For `ElasticSheet`, this
    // means updating the source frame used for parallel transport.
    virtual CSCMat sobolevInnerProductMatrix(Real /* Mscale */ = 1.0) const { throw std::runtime_error("Unimplemented"); }

    virtual void massMatrix(CSCMat &M, bool /* updatedParametrization */, bool /* lumped */) const { M.setIdentity(true); }

    // Get a FieldSampler for sampling FEM fields defined on the reference configuration mesh.
    virtual std::unique_ptr<FieldSampler> referenceConfigSampler()                     const { throw std::runtime_error("Unimplemented"); }
    virtual CSCMat deformationSamplerMatrix(Eigen::Ref<const Eigen::MatrixXd> /* P */) const { throw std::runtime_error("Unimplemented"); }

    virtual void setIdentityDeformation() {  throw std::runtime_error("Unimplemented"); }

    ////////////////////////////////////////////////////////////////////////////
    // Update notification infrastructure.
    ////////////////////////////////////////////////////////////////////////////
    // The callback interface is not considered part of the elastic object's
    // state and therefore the register/deregister methods are marked const.
    int registerUpdateCallback(VariableMask type, const NotificationCB &cb) const {
        if ((type != VariableMask::Defo) && (type != VariableMask::Rest))
            throw std::runtime_error("`type` must be VariableMask::Defo or VariableMask::Rest");
        int id;
        while (m_updateCBs.count(id = rand()));
        m_updateCBs.emplace(id, CBRecord{type, cb});
        return id;
    }

    void deregisterUpdateCallback(int id) const {
        auto it = m_updateCBs.find(id);
        if (it == m_updateCBs.end()) throw std::runtime_error("Attempted to deregister nonexistent callback");
        m_updateCBs.erase(it);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Convenience methods
    ////////////////////////////////////////////////////////////////////////////
    CSCMat massMatrix(bool updatedParametrization, bool lumped = false) const {
        CSCMat M(hessianSparsityPattern());
        massMatrix(M, updatedParametrization, lumped);
        return M;
    }

    virtual ~ElasticObject() { }
private:
    // The following two methods must be implemented by the derived class to
    // update the deformed/rest states.
    virtual void m_setDefoVars(const Eigen::Ref<const VXd> &vars) = 0;
    virtual void m_setRestVars(const Eigen::Ref<const VXd> &vars) = 0;

    // Deformed state update notifications
    struct CBRecord {
        VariableMask type;
        NotificationCB cb;
    };
    mutable std::map<int, CBRecord> m_updateCBs;

    void m_issueNotifications(VariableMask type) const {
        for (const auto &it : m_updateCBs) {
            const CBRecord &record = it.second;
            if (record.type == type)
                record.cb();
        }
    }

protected:
    // Subclasses are free to define other interfaces for mutating subsets of
    // variables (in which case, they must manually issue update notifications)
    void m_defoConfigUpdated() const { m_issueNotifications(VariableMask::Defo); }
    void m_restConfigUpdated() const { m_issueNotifications(VariableMask::Rest); }
};

#endif /* end of include guard: ELASTICOBJECT_HH */
