////////////////////////////////////////////////////////////////////////////////
// ConstraintBarrier.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A C2 barrier term for smooth, soft enforcement of an inequality constraint
//      c <= 0
//  it smoothly activates (becomes nonzero) when c falls exceeds
//  `activationThreshold` and shoots off to infinity as c approaches
//  `barrierThreshold`.
*///////////////////////////////////////////////////////////////////////////////
#ifndef CONSTRAINTBARRIER_HH
#define CONSTRAINTBARRIER_HH

#include <MeshFEM/Types.hh>

struct RawBarrierLog {
    template<typename Real_>
    Real_ logTerm(Real_ c) const {
        return log((barrierThreshold - c) / (barrierThreshold - activationThreshold));
    }

    template<typename Real_>
    Real_ b(Real_ c) const {
        if (c < activationThreshold) return 0.0;
        Real_ tmp = logTerm(c);
        return pow(-tmp, 3);
    }

    template<typename Real_>
    Real_ db(Real_ c) const {
        if (c < activationThreshold) return 0.0;
        return 3 * pow(logTerm(c), 2) / (barrierThreshold - c);
    }

    template<typename Real_>
    Real_ d2b(Real_ c) const {
        if (c < activationThreshold) return 0.0;
        Real_ l = logTerm(c);
        return (3 * (l - 2) * l) / ((barrierThreshold - c) * (barrierThreshold - c));
    }

    Real activationThreshold = -0.02; // where the penalty term becomes nonzero
    Real    barrierThreshold = 10.0;  // where it becomes infinite
};

template<class RawBarrier>
struct ConstraintBarrier_T : public RawBarrier {

    using Base = RawBarrier;
    ////////////////////////////////////////////////////////////////////////////
    // Lower/upper bound enforcement using the barrier function.
    ////////////////////////////////////////////////////////////////////////////
    template<typename Real_>
    Real_ eval(Real_ x, Real_ lower, Real_ upper) const {
        return Base::template b<Real_>(x - upper)
             + Base::template b<Real_>(lower - x);
    }

    template<typename Real_>
    Real_ deval(Real_ x, Real_ lower, Real_ upper) const {
        return Base::template db<Real_>(x - upper)
             - Base::template db<Real_>(lower - x);
    }

    template<typename Real_>
    Real_ d2eval(Real_ x, Real_ lower, Real_ upper) const {
        return Base::template d2b<Real_>(x - upper)
             + Base::template d2b<Real_>(lower - x);
    }
};

using ConstraintBarrier = ConstraintBarrier_T<RawBarrierLog>;

#endif /* end of include guard: CONSTRAINTBARRIER_HH */
