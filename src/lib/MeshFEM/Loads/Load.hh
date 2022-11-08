////////////////////////////////////////////////////////////////////////////////
// Load.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Generic interface for conservative loads originating from a potential
//  energy function (suitable for use in a nonlinear elasticity simulation).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/06/2020 10:30:20
////////////////////////////////////////////////////////////////////////////////
#ifndef LOADS_LOAD_HH
#define LOADS_LOAD_HH

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/ParallelAssembly.hh>

namespace Loads {

template<typename _Real = Real>
struct Load {
    using Real = _Real;
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    virtual Real energy() const = 0;

    // Derivative with respect to deformed configuration
    virtual VXd grad_x() const = 0;

    // Derivative with respect to rest configuration (for shape optimization)
    virtual VXd grad_X() const = 0;

    // Hessian with respect to deformed configuration (H_xx)
    virtual void hessian(SuiteSparseMatrix &H, bool projectionMask = true) const = 0;

    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const = 0;
    SuiteSparseMatrix constructHessian() const {
        auto H = hessianSparsityPattern(0.0);
        hessian(H);
        return H;
    }

    virtual ~Load() { }
};

template<class EO>
struct EOStoragePolicyWeakPtr {
    using StorageType = std::weak_ptr<const EO>;

    EOStoragePolicyWeakPtr(const std::weak_ptr<const EO> &obj) : m_obj(obj) { }

    const EO &getObj() const {
        if (auto o = m_obj.lock()) return *o;
        throw std::runtime_error("Object was destroyed");
    }

    std::shared_ptr<const EO> getObjPtr() const { return m_obj.lock(); }

private:
    StorageType m_obj;
};

template<class EO>
struct EOStoragePolicyReference {
    using StorageType = const EO &;
    EOStoragePolicyReference(const EO &obj) : m_obj(obj) { }
    const EO &getObj() const { return m_obj; }
    const EO *getObjPtr() const { return &m_obj; }
private:
    StorageType m_obj;
};

// Load that depends on the deformed and/or rest state of an ElasticObject.
// This base class manages the elastic object pointers for derived load classes
// along with the callbacks needed for them to leverage `ElasticObject`'s state
// update notification infrastructure.
template<class EO, template<typename T> class EOStoragePolicy = EOStoragePolicyWeakPtr>
struct ObjectSpecificLoad : public Load<Real>, public EOStoragePolicy<EO> {
    using Real = typename EO::Real;
    using Base = Load<Real>;
    using SP   = EOStoragePolicy<EO>;
    using VM   = typename EO::VariableMask;
    using SP::getObj;

    ObjectSpecificLoad(const typename SP::StorageType &obj)
        : SP(obj)
    {
        m_defoUpdateCallbackID = getObj().registerUpdateCallback(VM::Defo, [this]() { m_stateUpdated(VM::Defo); });
        m_restUpdateCallbackID = getObj().registerUpdateCallback(VM::Rest, [this]() { m_stateUpdated(VM::Rest); });
    }

    virtual ~ObjectSpecificLoad() {
        if (auto o = this->getObjPtr()) {
            o->deregisterUpdateCallback(m_defoUpdateCallbackID);
            o->deregisterUpdateCallback(m_restUpdateCallbackID);
        }
    }

private:
    // Called whenever the elastic object rest state or deformed state updates.
    // This mechanism enables caching of quantities that otherwise would need
    // to be recomputed separately by the energy/gradient/hessian methods.
    virtual void m_stateUpdated(VM /* vmask */) { /* NOP */ }

    int m_defoUpdateCallbackID, m_restUpdateCallbackID;
};

}

#endif /* end of include guard: LOADS_LOAD_HH */
