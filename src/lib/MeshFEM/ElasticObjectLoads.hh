#ifndef ELASITCOBJECTLOADS_HH
#define ELASITCOBJECTLOADS_HH

#include <array>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

// Loads currently take the form of Dirichlet constraints.
// If the Dirichlet constraints are not sufficient to pin down
// rigid motion, the user should specify applyRigidMotionPins = true.
template<class EObject>
struct ElasticObjectLoads {
    using Real = typename EObject::Real;
    using VecX = typename EObject::VXd;

    struct DirichletConstraint {
        DirichletConstraint(size_t i, Real v) : varIndex(i), value(v) { }
        size_t varIndex;
        Real value;
    };

    bool applyRigidMotionPins = true;
    std::vector<DirichletConstraint> dirichletConstraints;

    Real energy(const EObject &eo) const {
        return 0;
    }

    VecX gradient(const EObject &eo) const {
        VecX g(VecX::Zero(eo.numVars()));
        return g;
    }
};

#endif /* end of include guard: ELASITCOBJECTLOADS_HH */
