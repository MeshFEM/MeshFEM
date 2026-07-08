////////////////////////////////////////////////////////////////////////////////
// CollapsePreventionEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Sheet material energy density that prevents elements from collapsing into
//  degenerate configurations with an infinite energy barrier:
//      (-log((det(C) - activationThreshold) / activationThreshold + 1))^2
//  for det(C) < activationThreshold, 0 otherwise
//
//  This energy term is C1. We could make it C2 (to avoid the single point
//  where the Hessian is undefined) by raising the power from 2 to 3--at the
//  expense of a faster ramp-up (greater nonlinearity).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/30/2019 17:23:18
////////////////////////////////////////////////////////////////////////////////
#ifndef COLLAPSEPREVENTIONENERGY_HH
#define COLLAPSEPREVENTIONENERGY_HH
#include <cmath>

#include <Eigen/Dense>
#include <MeshFEM/EnergyDensities/Tensor.hh>

namespace MeshFEM {

template<typename Real_>
struct BarrierFuncLogSq {
    using Real = Real_;

    constexpr static Real inf = std::numeric_limits<double>::infinity();

    static Real   b(Real x) { if (x <= 0) return inf; if (x >= 1.0) return 0.0; return 0.5 * std::pow(log(x), 2); }
    static Real  db(Real x) { if (x <= 0) return inf; if (x >= 1.0) return 0.0; return log(x) / x; }
    static Real d2b(Real x) { if (x <= 0) return inf; if (x >= 1.0) return 0.0; return (1 - log(x)) / (x * x); }
};

template<class BarrierFunc>
struct NormalizedBarrierFunction {
    using Real = typename BarrierFunc::Real;
    using BF = BarrierFunc;

    NormalizedBarrierFunction(Real a = 1.0) : m_a(a) { }

    void setActivationThreshold(Real val) { m_a = val; }
    Real activationThreshold() const { return m_a; }

    Real   b(Real x) const { return BF::  b(x / m_a); }
    Real  db(Real x) const { return BF:: db(x / m_a) / m_a; }
    Real d2b(Real x) const { return BF::d2b(x / m_a) / (m_a * m_a); }

protected:
    Real m_a = 1.0;
};

template<class BarrierFunc, size_t _Dimension>
struct CollapsePreventionDet : public NormalizedBarrierFunction<BarrierFunc> {
    using Real = typename BarrierFunc::Real;
    using BF = NormalizedBarrierFunction<BarrierFunc>;

    static constexpr size_t Dimension    = _Dimension;
    static constexpr size_t N            = Dimension;
    static constexpr EDensityType EDType = EDensityType::CBased;

    static_assert(N == 2, "Only 2x2 supported for now");

    using MNd = Eigen::Matrix<Real, N, N>;
    using Matrix = MNd; // Needed for bindings

    CollapsePreventionDet(Real a = 1.0) : BF(a) { setC(MNd::Identity()); }

    CollapsePreventionDet(const CollapsePreventionDet &other) = default;
    CollapsePreventionDet &operator=(const CollapsePreventionDet &) = default;

    // Constructor copying material properties only, not the current deformation
    CollapsePreventionDet(const CollapsePreventionDet &other,
                        UninitializedDeformationTag &&)
        : BF(other.m_a) { }

    static const char *name() { return "CollapsePreventionDet"; }

    void setC(Eigen::Ref<const MNd> C) {
        m_det = C.determinant();
        m_grad_det <<  C(1, 1), -C(1, 0),
                      -C(0, 1),  C(0, 0);
    }

    Real   energy() const { return BF::b(m_det); }
    MNd PK2Stress() const { return BF::db(m_det) * m_grad_det; }

    template<class Mat_>
    MNd delta_PK2Stress(const Mat_ &dC) const {
        MNd delta_grad_det;
        delta_grad_det <<  dC(1, 1), -dC(1, 0),
                          -dC(0, 1),  dC(0, 0);

        return ((BF::d2b(m_det) * doubleContract(m_grad_det, dC.matrix().template cast<Real>()))) * m_grad_det
               + BF:: db(m_det) * delta_grad_det;
    }

    template<class Mat1_, class Mat2_>
    MNd delta2_PK2Stress(const Mat1_ &dC_a, const Mat2_ &dC_b) const {
        throw std::runtime_error("Unimplemented");
    }

    // For debugging scalar function of det + its derivatives
    void setDet(Real det) { m_det = det; }
    Real det() const { return m_det; }
    Real normalizedDet()  const { return m_det / BF::m_a; }
    Real denergy_ddet()   const { return BF::db(m_det);  }
    Real d2energy_d2det() const { return BF::d2b(m_det); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Real m_det;
    MNd m_grad_det;
};

template<class Real, size_t N>
using CollapsePreventionEnergyDet = CollapsePreventionDet<BarrierFuncLogSq<Real>, N>;

} // namespace MeshFEM

#endif /* COLLAPSEPREVENTIONENERGY_HH */
