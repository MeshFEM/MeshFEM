////////////////////////////////////////////////////////////////////////////////
// DihedralAngle.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Evaluates and differentiates the dihedral angle of a triangle flap.
//  This can be used to implement hinge-based energies like
//  [Grinspun et al. 2003: Discrete Shells].
//
//  We label a generic triangle flap as follows:
//             p1<-e--p0            n1   n0
//               \ 0 / \             ^ θ ^
//                \ /`.1\             \ /
//                p2   `.\             o
//                       p3
//  and define the dihedral angle as:
//      θ := atan2((n0 x n1) . e, n0 . n1),
//  where "e" is the vector pointing along hinge half-edge lying in triangle 0.
//  Note that this angle definition is invariant when the roles of triangles 0
//  and 1 are exchanged since this flips the signs of both `(n0 x n1)` and `e`.
//
//  Note that ±e is halfedge 2 of both incident triangles (which have corners
//  (0, 1, 2) and (1, 0, 3), respectively.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/09/2023 10:40:43
*///////////////////////////////////////////////////////////////////////////////
#ifndef DIHEDRALANGLE_HH
#define DIHEDRALANGLE_HH

#include <iostream>
#include "DeformedTriangleGeometry.hh"

namespace elements {

template<class Real>
struct DihedralAngle {
    using StencilPoints = Eigen::Matrix<Real, 4, 3, Eigen::RowMajor>;
    static constexpr size_t NumVars = 4 * 3;
    static constexpr size_t HingeHE = 2; // The hinge is halfedge 2 in both triangles.
    using Gradient = Eigen::Matrix<Real, NumVars, 1>;
    using Hessian  = Eigen::Matrix<Real, NumVars, NumVars>;

    static constexpr size_t TriCorners[2][3] = {{0, 1, 2}, {1, 0, 3}};

    using DTG = DeformedTriangleGeometry<Real>;

    DihedralAngle(const StencilPoints &pts) { configure(pts); }
    DihedralAngle() { } // WARNING: leaves members uninitalized!

    void configure(const Eigen::Ref<const StencilPoints> &pts) {
        using CP = typename DTG::CornerPositions;
        de[0].configure((CP() << pts.row(0), pts.row(1), pts.row(2)).finished());
        de[1].configure((CP() << pts.row(1), pts.row(0), pts.row(3)).finished());

        const auto &n0 = de[0].normal;
        const auto &n1 = de[1].normal;
        const auto &e  = de[0].edgeVecs.col(HingeHE);

        theta = atan2(n0.cross(n1).dot(e), n0.dot(n1) * de[0].edgeLens[HingeHE]); // Note: can't use std::atan2 since this breaks ADL for autodiff types
    }

    Real value() const { return theta; }

    Real hingeEdgeLen() const { return de[0].edgeLens[HingeHE]; }
    Real avgHeight()    const { return 0.5 * (de[0].h[HingeHE] + de[1].h[HingeHE]); }

    Gradient gradient() const {
        Gradient result;
        auto grad_pts = Eigen::Map<StencilPoints>(result.data());
        using M3dR = Eigen::Matrix<Real, 3, 3, Eigen::RowMajor>;

        auto contrib_0 = (-1.0 / (de[0].edgeVecDotProducts(HingeHE, HingeHE) * de[0].h[HingeHE])) * de[0].edgeVecDotProducts.col(HingeHE) * de[0].normal.transpose(); // evaluated upon assignment below
        M3dR contrib_1 = (-1.0 / (de[1].edgeVecDotProducts(HingeHE, HingeHE) * de[1].h[HingeHE])) * de[1].edgeVecDotProducts.col(HingeHE) * de[1].normal.transpose();

        // Contribution from tri 0 (corners (0, 1, 2))
        grad_pts.template topRows<3>() = contrib_0;
        // Contribution from tri 1 (corners (1, 0, 3))
        grad_pts.row(1) += contrib_1.row(0);
        grad_pts.row(0) += contrib_1.row(1);
        grad_pts.row(3)  = contrib_1.row(2);
        return result;
    }

    Hessian hessian() const {
        Hessian result;
        result.setZero();

        // n ⨂  ehatp_i term
        Real liSq = de[0].edgeVecDotProducts(HingeHE, HingeHE);
        {
            auto contrib = ((1.0 / liSq)
                    * (de[0].normal * de[0].unitEdgePerpendiculars.col(HingeHE).transpose() +
                       de[1].normal * de[1].unitEdgePerpendiculars.col(HingeHE).transpose())).eval();

            result.template block<3, 3>(0, 0) =  contrib;
            result.template block<3, 3>(3, 3) =  contrib;
            result.template block<3, 3>(0, 3) = -contrib;
        }

        for (size_t tri = 0; tri < 2; ++tri) {
            const auto &de_t = de[tri];
            Real coeff = 1.0 / (liSq * de_t.h[HingeHE]);
            for (size_t local_l = 0; local_l < 3; ++local_l) {
                for (size_t local_k = 0; local_k < 3; ++local_k) {
                    size_t k = TriCorners[tri][local_k];
                    size_t l = TriCorners[tri][local_l];
                    result.template block<3, 3>(3 * k, 3 * l) += (coeff * de_t.edgeVecDotProducts(HingeHE, local_l) / de_t.h[local_k]) * de_t.normal * de_t.unitEdgePerpendiculars.col(local_k).transpose()
                                                              +  (coeff * de_t.edgeVecDotProducts(HingeHE, local_k) / de_t.h[local_l]) * de_t.unitEdgePerpendiculars.col(local_l) * de_t.normal.transpose();
                }
            }
        }

        result.template triangularView<Eigen::Lower>() = result.transpose();
        return result;
    }

private:
    DTG de[2];
    Real theta;
};

}

#endif /* end of include guard: DIHEDRALANGLE_HH */
