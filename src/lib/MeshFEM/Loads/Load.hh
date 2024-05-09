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
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>

namespace Loads {

template<typename _Real = Real>
struct Load : public NewtonObjectiveTerm {
    using Real = _Real;
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using VM = VariableMask;

    Load(const NewtonObjectiveTerm::NVStorageType &vars)
        : NewtonObjectiveTerm(vars) { }

    ////////////////////////////////////////////////////////////////////////////
    // Wrapers adapting to the NewtonObjectiveTerm interface
    ////////////////////////////////////////////////////////////////////////////
    Real objective() const override { return energy(); }
    virtual void accumulateGradient(Real w, VXd &g, bool freshIterate = false) const override {
        if (w == 1.0) g +=     grad_x();
        else          g += w * grad_x();
    }

    virtual Real energy() const = 0;

    // Derivative with respect to deformed configuration
    virtual VXd grad_x() const = 0;

    // Derivative with respect to rest configuration (for shape optimization)
    virtual VXd grad_X() const = 0;

    // Notification from NewtonMultiobjectiveProblem that (deformation) variables have changed.
    void varsUpdated() override { m_stateUpdated(VM::Defo); }

    virtual void m_stateUpdated(VM /* vmask */) { /* NOP */ }

    virtual ~Load() { }
};

// Load that depends on the deformed and/or rest state of an ElasticObject.
// This base class manages the elastic object pointers for derived load classes
// along with the callbacks needed for them to leverage `ElasticObject`'s state
// update notification infrastructure.
template<class EO>
struct ObjectSpecificLoad : public Load<Real> {
    using Real = typename EO::Real;
    using Base = Load<Real>;
    using VM   = VariableMask;
    using EOStorageType = std::weak_ptr<const EO>; // See note below in `getObj`...

    ObjectSpecificLoad(const std::weak_ptr<const EO> &eo)
        : Load(eo), m_eo(eo) { }

    const EO &getObj() const {
        // The following `dynamic_cast` approach unfortunately breaks
        // (at least when compiled in debug mode), probably because of
        // difficulties casting across shared library boundaries.
        // Even though the RTTI names do match, the cast throws a
        // `std::bad_cast` exception)
        //      std::cout << "Attempting to cast nvars of dynamic type " << typeid(Base::getNVars()).name() << " to " << typeid(EO).name() << std::endl;
        //      return dynamic_cast<const EO &>(Base::getNVars());
        // We therefore resort to maintaining a separate typed pointer from
        // the one maintained by `Load`)
        if (auto eo_ptr = m_eo.lock()) return *eo_ptr;
        throw std::runtime_error("ElasticObject was destroyed");
    }

    virtual ~ObjectSpecificLoad() { }

private:
    // Called whenever the elastic object rest state or deformed state updates.
    // This mechanism enables caching of quantities that otherwise would need
    // to be recomputed separately by the energy/gradient/hessian methods.
    virtual void m_stateUpdated(VM /* vmask */) { /* NOP */ }

    EOStorageType m_eo;
};

}

#endif /* end of include guard: LOADS_LOAD_HH */
