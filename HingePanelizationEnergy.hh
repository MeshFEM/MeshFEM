////////////////////////////////////////////////////////////////////////////////
// HingeEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements the bending energy from [Grinspun et al. 2003: Discrete Shells].
//  We provide analytical gradients for the energy, but resort to automatic
//  differentiation for Hessians.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/25/2019 15:43:17
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cmath>
#include <Eigen/Dense>
#include <array>
#include <MeshFEM/AutomaticDifferentiation.hh>

// Bending energy contributed by a single "hinge" (mesh edge) between two
// triangles.
// The triangle points are indexed as follows:
//            p0-----p1            n1   n2
//              \ 1 / \             ^   ^
//               \ /`.2\             \ /
//               p2   `.\             o
//                      p3
// We call the complement of the dihedral angle "theta". Then the bending energy is given by
//      0.5 (theta - theta_bar)^2 ||e_bar||/h_bar
// (we expect the code using this class to scale everything by the bending stiffness).

// Get the indices corresponding to the four vertices in the hinge stencil for halfedge "he".
//  p0-he->p1
//    \ 1 /2|
//     \ /`.|
//     p2  p3
// Note: "he" lies in triangle 2.


// Templated by real number type "_Real" for autodiff.
template<class _Real>
struct HingePanelizationEnergy {
    using Real = _Real;
    using Pt = Eigen::Matrix<_Real, 3, 1>;
    using Vec = Pt;

    // Reference configuration quantities
    // (We don't need autodiff types for these...)
    double e_bar_len, h_bar;
    double theta_bar;
    Real delta = 0.1;
    // Deformed configuration quantities
    Eigen::Matrix<Real, 3, 4> deformed_pts;
    Real theta, e_len;
    Real squared_dbl_A1, squared_dbl_A2;
    Vec N1, N2; // un-normalized triangle normals (cross products of edge vectors)
    Real e_01_dot_ehat, e_02_dot_ehat;
    Real e_11_dot_ehat, e_12_dot_ehat;

    // Copy the reference configuration quantities from an existing class of a different type
    template<class _Real2>
    HingePanelizationEnergy(HingePanelizationEnergy<_Real2> h2) : e_bar_len(h2.e_bar_len), h_bar(h2.h_bar), theta_bar(h2.theta_bar) {
        e_bar_len      = h2.e_bar_len;
        h_bar          = h2.h_bar;
        theta_bar      = h2.theta_bar;

        theta          = h2.theta;
        e_len          = h2.e_len;
        squared_dbl_A1 = h2.squared_dbl_A1;
        squared_dbl_A2 = h2.squared_dbl_A2;
        N1             = h2.N1;
        N2             = h2.N2;
        e_01_dot_ehat  = h2.e_01_dot_ehat;
        e_02_dot_ehat  = h2.e_02_dot_ehat;

        e_11_dot_ehat  = h2.e_11_dot_ehat;
        e_12_dot_ehat  = h2.e_12_dot_ehat;
    }

    HingePanelizationEnergy(Eigen::Ref<const Pt> ref_p0,
                Eigen::Ref<const Pt> ref_p1,
                Eigen::Ref<const Pt> ref_p2,
                Eigen::Ref<const Pt> ref_p3) {
        Vec e = ref_p1 - ref_p0;
        e_bar_len = e.norm();
        e /= e_bar_len;

        Vec ref_n1 = (ref_p2 - ref_p0).cross(ref_p1 - ref_p0),
            ref_n2 = (ref_p1 - ref_p0).cross(ref_p3 - ref_p0);
        double dbl_A1 = ref_n1.norm(),
               dbl_A2 = ref_n2.norm();
        h_bar = (dbl_A1 + dbl_A2) / (6.0 * e_bar_len); // 1/6 (h1 + h2) = 1/6 (b * h1 + b * h2) / b = 1/6(2 A1 + 2 A2) / b

        // Note: n1, n2 needn't be normalized since atan2 is invariant to uniform scaling of its arguments.
        theta_bar = atan2(ref_n2.cross(ref_n1).dot(e), ref_n1.dot(ref_n2)); // Note: can't use std::atan2 since this breaks ADL for autodiff types

        setDeformedConfiguration(ref_p0, ref_p1, ref_p2, ref_p3);
    }

    void setDeformedConfiguration(Eigen::Ref<const Pt> p0,
                                  Eigen::Ref<const Pt> p1,
                                  Eigen::Ref<const Pt> p2,
                                  Eigen::Ref<const Pt> p3) {
        deformed_pts.col(0) = p0;
        deformed_pts.col(1) = p1;
        deformed_pts.col(2) = p2;
        deformed_pts.col(3) = p3;

        Vec e = p1 - p0;
        e_len = e.norm();
        e /= e_len;

        N1 = (p2 - p0).cross(p1 - p0),
        N2 = (p1 - p0).cross(p3 - p0);

        squared_dbl_A1 = N1.squaredNorm();
        squared_dbl_A2 = N2.squaredNorm();

        // Note: n1, n2 needn't be normalized since atan2 is invariant to uniform scaling of its arguments.
        theta = atan2(N2.cross(N1).dot(e), N1.dot(N2)); // Note: can't use std::atan2 since this breaks ADL for autodiff types

        e_01_dot_ehat = e.dot(p1 - p2);
        e_02_dot_ehat = e.dot(p1 - p3); // really the negation of e_02 based on the labeling in the derivation figure...

        e_11_dot_ehat = e.dot(p2 - p0);
        e_12_dot_ehat = e.dot(p3 - p0);

        // Effectively disable this hinge's energy in degenerate configurations since
        // these will introduce large and pseudorandom values into the gradient and Hessian,
        // breaking the optimization.
        if ((e_len < 1e-9) || (squared_dbl_A1 < 1e-16) || (squared_dbl_A2 < 1e-16)) {
            e.setZero();
            e[0] = 1.0;
            theta = theta_bar;
            squared_dbl_A1 = 1.0;
            squared_dbl_A2 = 1.0;
            N1.setZero();
            N2.setZero();
            e_01_dot_ehat = e_02_dot_ehat = e_11_dot_ehat = e_12_dot_ehat = 0.0;
        }
    }

    // Gradient of theta with respect to the matrix [p0 | p1 | p2 | p3].
    Eigen::Matrix<Real, 3, 4> gradTheta() const {
        Eigen::Matrix<Real, 3, 4> result;
        result.col(0) = (e_01_dot_ehat / squared_dbl_A1) * N1 + (e_02_dot_ehat / squared_dbl_A2) * N2;
        result.col(1) = (e_11_dot_ehat / squared_dbl_A1) * N1 + (e_12_dot_ehat / squared_dbl_A2) * N2;
        result.col(2) = (-e_len / squared_dbl_A1) * N1;
        result.col(3) = (-e_len / squared_dbl_A2) * N2;
        return result;
    }

    using HessType = Eigen::Matrix<Real, 12, 12>;
    HessType hessTheta() const {
        HessType result;
        using ADType = Eigen::AutoDiffScalar<Eigen::Matrix<Real, 12, 1>>;
        HingePanelizationEnergy<ADType> diff_he(*this);

        Eigen::Matrix<ADType, 3, 4> ad_deformed_pts = deformed_pts;

        for (size_t j = 0; j < 12; ++j) {
            ad_deformed_pts.data()[j].derivatives().setZero();
            ad_deformed_pts.data()[j].derivatives()[j] = 1.0;
        }

        diff_he.setDeformedConfiguration(ad_deformed_pts.col(0),
                                         ad_deformed_pts.col(1),
                                         ad_deformed_pts.col(2),
                                         ad_deformed_pts.col(3));
        auto diff_g = diff_he.gradTheta();

        for (size_t i = 0; i < 12; ++i)
            result.row(i) = diff_g.data()[i].derivatives().transpose();

        return result;
    }

    Real sigmoid(Real x) const {
        Real x2 = x*x;
        return x2 / (x2 + delta);
    }

    Real dsigmoid(Real x) const {
        Real x2 = x*x;
        return 2 * delta * x / ((x2 + delta) * (x2 + delta));
    }

    Real ddsigmoid(Real x) const {
        Real x2 = x*x;
        return 2 * delta * (delta - 3 * x2) / ((x2 + delta) * (x2 + delta) * (x2 + delta));
    }
      

    Real energy() const {
        // Note: this is 1/2 the energy in [Grinspun 2003]
        return sigmoid(theta);
    }

    using Gradient = Eigen::Matrix<Real, 12, 1>;
    Gradient gradient() const {
        Gradient result;
        Eigen::Map<Eigen::Matrix<Real, 3, 4>>(result.data()) = dsigmoid(theta) * gradTheta();
        return result;
    }

    HessType hessian() const {
        auto g = gradTheta();
        auto gFlattened = Eigen::Map<Eigen::Matrix<Real, 12, 1>>(g.data());
        return ddsigmoid(theta) * gFlattened * gFlattened.transpose() + dsigmoid(theta) * hessTheta();
    }
};
