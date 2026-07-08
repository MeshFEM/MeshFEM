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
#include <cstdlib>
#include <functional>


#include "FieldSampler.hh"
#include <MeshFEMSparse/SparseMatrices.hh>
#include "newton_optimizer/MultiobjectiveProblem.hh"
#include "IPCIntegration/CollisionMesh.hh"

namespace MeshFEM {

////////////////////////////////////////////////////////////////////////////
// Generic variables:
// An optimization can include variables parametrizing the deformed
// configuration, the rest configuration, or both. The generic interface
// here allows the user to specify which variables why wish to access via
// a `VariableMask`.
////////////////////////////////////////////////////////////////////////////
enum class VariableMask { Defo, Rest, All };

template<class _Real>
struct MESHFEM_EXPORT ElasticObject : public NewtonObjectiveTermBase, public NewtonVarsBase {
    using Real = _Real;
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using CSCMat = CSCMatrix<SuiteSparse_long, _Real>;
    using NotificationCB = std::function<void()>;

    ////////////////////////////////////////////////////////////////////////////
    // Wrapper methods implementing the NewtonVarsManager interface.
    ////////////////////////////////////////////////////////////////////////////
    size_t numVars() const override { return numVars(VariableMask::Defo); }
    VXd    getVars() const override { return getVars(VariableMask::Defo); }

    size_t numParameters() const override { return numVars(VariableMask::Rest); }
    VXd    getParameters() const override { return getVars(VariableMask::Rest); }

    ////////////////////////////////////////////////////////////////////////////
    // Wrapper methods implementing the NewtonObjectiveTermBase interface
    ////////////////////////////////////////////////////////////////////////////
    Real objective() const override { return energy(); }

    virtual void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        return accumulateGradient(weight, g, freshIterate, VariableMask::Defo);
    }

    using NewtonObjectiveTermBase::hessian; // Don't shadow the `hessian` convenience method
    virtual void accumulateHessian(Real weight, NewtonHessian &H, bool projectionMask) const override {
        return accumulateHessian(weight, H, projectionMask, VariableMask::Defo);
    }

    virtual NewtonHessian hessianSparsityPattern() const override {
        return hessianSparsityPattern(VariableMask::Defo);
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

    using NewtonVarsBase::setVars;
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

    // Elastic objects can have, e.g., angle variables in addition to nodal
    // variables. We sometimes need to know the number of nodes, which cannot
    // be inferred from the number of variables alone for these objects.
    virtual size_t numNodes() const = 0;

    virtual VXd getDefoVars() const = 0;
    virtual VXd getRestVars() const = 0;

    void setDefoVars(const Eigen::Ref<const VXd> &vars) { NewtonVarsBase::setVars(vars); }
    void setRestVars(const Eigen::Ref<const VXd> &vars) { NewtonVarsBase::setParameters(vars); }

    ////////////////////////////////////////////////////////////////////////////
    // Energy and derivatives
    ////////////////////////////////////////////////////////////////////////////
    virtual Real  energy() const = 0;
    virtual void accumulateGradient(Real weight, VXd &g, bool updatedParametrization, VariableMask vmask) const = 0;
    virtual void accumulateHessian(Real weight, NewtonHessian &H, bool projectionMask, VariableMask vmask) const = 0;
    virtual NewtonHessian hessianSparsityPattern(VariableMask vmask) const = 0;
    virtual VXd contract_d2E_dXdx(const VXd &/* y */) const { throw std::runtime_error("Unimplemented!"); }

    using NewtonObjectiveTermBase::gradient; // prevent hiding
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

    // Store the full mass matrix in `M`, preserving its sparsity pattern.
    virtual void massMatrix(NewtonHessian &M, bool /* updatedParametrization */) const { M.setIdentity(true); }
    virtual VXd  lumpedMass(bool /* updatedParametrization */) const { return VXd::Ones(numDefoVars()); }

    // Whether the mass matrix depends on the deformed configuration.
    virtual bool hasVariableMassMatrix() const { return false; }

    // Get a FieldSampler for sampling FEM fields defined on the reference configuration mesh.
    virtual std::unique_ptr<FieldSampler> referenceConfigSampler()                     const { throw std::runtime_error("Unimplemented"); }
    virtual CSCMat deformationSamplerMatrix(Eigen::Ref<const Eigen::MatrixXd> /* P */) const { throw std::runtime_error("Unimplemented"); }

    virtual void setIdentityDeformation() {  throw std::runtime_error("Unimplemented"); }
    void applyTrivialInitialGuess() override { setIdentityDeformation(); }

    ////////////////////////////////////////////////////////////////////////////
    // Update notification infrastructure.
    ////////////////////////////////////////////////////////////////////////////
    // The callback interface is not considered part of the elastic object's
    // state and therefore the register/deregister methods are marked const.
    int registerUpdateCallback(VariableMask type, const NotificationCB &cb) const {
        return NewtonVarsBase::registerUpdateCallback(m_vtypeForVariableMask(type), cb);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Convenience methods
    ////////////////////////////////////////////////////////////////////////////
    NewtonHessian massMatrix(bool updatedParametrization) const {
        NewtonHessian M = hessianSparsityPattern();
        massMatrix(M, updatedParametrization);
        return M;
    }

    // Note: changing the mass density invalidates certain rest-state-cache
    // quantities (like the gravity load vector), so we issue a rest-state
    // update notification below.
    Real getMassDensity() const { return m_rho; }
    void setMassDensity(Real rho) { m_rho = rho; m_restConfigUpdated(); }

    // 3D volume attached to a (potentially codimensional) element.
    // For solid elements this is just the element volume, but for
    // plates it is area * thickness.
    virtual Real element3DVolume(size_t ei) const = 0;

    //////////////////////////////////////////////////////////////////////////
    // Methods needed for IPCEquilibriumSolver
    //////////////////////////////////////////////////////////////////////////
    virtual CollisionMesh getCollisionMesh() const { throw std::runtime_error("Unimplemented"); }
    virtual Real volume()            const { throw std::runtime_error("Unimplemented"); }

    virtual ~ElasticObject() { }
private:
    // Global material density (used for scaling the mass matrix and for gravity loads)
    Real m_rho = 1.0;

    // The following two methods must be implemented by the derived class to
    // update the deformed/rest states.
    virtual void m_setDefoVars(const Eigen::Ref<const VXd> &vars) = 0;
    virtual void m_setRestVars(const Eigen::Ref<const VXd> &vars) = 0;

    static VarType m_vtypeForVariableMask(VariableMask type) {
        if ((type != VariableMask::Defo) && (type != VariableMask::Rest))
            throw std::runtime_error("`type` must be VariableMask::Defo or VariableMask::Rest");
        return (type == VariableMask::Defo) ? VarType::Variable : VarType::Parameter;
    }

    virtual void m_setVarsImpl(const VXd &vars) override { m_setDefoVars(vars); }           // Note: NewtonVarsBase::setVars already dispatches notification!
    virtual void m_setParametersImpl(const VXd &params) override { m_setRestVars(params); } // Note: NewtonVarsBase::setParameters already dispatches notification!

protected:
    // Subclasses are free to define other interfaces for mutating subsets of
    // variables (in which case, they must manually issue update notifications)
    void m_defoConfigUpdated() const { m_issueNotifications(VarType::Variable); }
    void m_restConfigUpdated() const { m_issueNotifications(VarType::Parameter); }
};

} // namespace MeshFEM

#endif /* end of include guard: ELASTICOBJECT_HH */
