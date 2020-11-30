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

template<class _Real>
class ElasticObject {
public:
    using Real = _Real;
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    using NotificationCB = std::function<void()>;

    virtual void setVars(Eigen::Ref<const VXd> vars) = 0;
    virtual Real  energy() const = 0;
    virtual VXd gradient() const = 0;
    virtual void hessian(SuiteSparseMatrix &Hout, bool projectionMask = false) const = 0;
    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const = 0;

    // Optional interface
    virtual SuiteSparseMatrix massMatrix(bool /* lumped */ = false)              const { throw std::runtime_error("Unimplemented"); }
    virtual SuiteSparseMatrix sobolevInnerProductMatrix(Real /* Mscale */ = 1.0) const { throw std::runtime_error("Unimplemented"); }

    // Get a FieldSampler for sampling FEM fields defined on the reference configuration mesh.
    virtual std::unique_ptr<FieldSampler> referenceConfigSampler()                                            const { throw std::runtime_error("Unimplemented"); }
    virtual SuiteSparseMatrix             deformationSamplerMatrix(Eigen::Ref<const Eigen::MatrixXd> /* P */) const { throw std::runtime_error("Unimplemented"); }

    // The callback interface is not considered part of the elastic object's
    // state and therefore the register/deregister methods are marked const.
    int    registerDeformationUpdateCallback(const NotificationCB &cb) const { return registerCallback(cb, m_deformationUpdateCBs); }
    void deregisterDeformationUpdateCallback(int id)                   const {      deregisterCallback(id, m_deformationUpdateCBs); }
    int     registerRestConfigUpdateCallback(const NotificationCB &cb) const { return registerCallback(cb,        m_restUpdateCBs); }
    void  deregisterRestConfigUpdateCallback(int id)                   const {      deregisterCallback(id,        m_restUpdateCBs); }

    virtual ~ElasticObject() { }

protected:
    // Deformed state update notifications
    using NotificationRegistry = std::map<int, NotificationCB>;
    mutable NotificationRegistry m_deformationUpdateCBs, m_restUpdateCBs;
    void m_deformedConfigUpdated() const {
        for (const auto &entry : m_deformationUpdateCBs)
            entry.second();
    }

    static int registerCallback(const NotificationCB &cb, NotificationRegistry &registry) {
        int id;
        while (registry.count(id = rand()));
        registry.emplace(id, cb);
        return id;
    }

    static void deregisterCallback(int id, NotificationRegistry &registry) {
        auto it = registry.find(id);
        if (it == registry.end()) throw std::runtime_error("Attempted to deregister nonexistent callback");
        registry.erase(it);
    }
};

#endif /* end of include guard: ELASTICOBJECT_HH */
