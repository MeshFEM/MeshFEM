////////////////////////////////////////////////////////////////////////////////
// MetricFitting.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  C-based energy density that simply fits the first fundamental form to
//  a target value.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/24/2023 22:57:20
*///////////////////////////////////////////////////////////////////////////////
#ifndef METRICFITTING_HH
#define METRICFITTING_HH

#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

template <typename _Real, size_t _Dimension>
struct MetricFittingEnergy {
    static constexpr size_t Dimension = _Dimension;
    static constexpr size_t N         = _Dimension;
    static constexpr EDensityType EDType = EDensityType::CBased;
    using Real = _Real;
    using MNd  = Eigen::Matrix<_Real, N, N>;
    using Matrix = MNd; // Needed for bindings

    MetricFittingEnergy(const MNd &target = MNd::Identity())
        : targetMetric(target) { m_C.setIdentity(); }

    MetricFittingEnergy(const MetricFittingEnergy &other) = default;
    MetricFittingEnergy &operator=(const MetricFittingEnergy &) = default;

    // Constructor copying material properties only, not the current deformation
    MetricFittingEnergy(const MetricFittingEnergy &other,
                        UninitializedDeformationTag &&)
        : targetMetric(other.targetMetric) { }

    static const char *name() { return "MetricFitting"; }

    void setC(Eigen::Ref<const MNd> C) {
        m_C = C;
    }

    double energy() const { return 0.25 * (m_C - targetMetric).squaredNorm(); }

    // d psi / d E,    E := 0.5 (C - I)
    MNd PK2Stress() const { return m_C - targetMetric; }

    template<class Mat_>
    MNd delta_PK2Stress(const Mat_ &dC) const { return dC.matrix(); } // 4th order identity tensor

    // Hessian is constant, third derivatives are zero.
    template<class Mat_, class Mat2_>
    MNd delta2_PK2Stress(const Mat_ &/* dC_a */, const Mat2_ &/* dC_b */) const { return MNd::Zero(); }

    MNd currentMetric() const {
        return m_C;
    }

    MNd targetMetric;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    MNd m_C;
};

#endif /* end of include guard: METRICFITTING_HH */
