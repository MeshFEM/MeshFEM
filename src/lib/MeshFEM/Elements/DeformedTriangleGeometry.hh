////////////////////////////////////////////////////////////////////////////////
// DeformedTriangleGeometry.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Collect some triangle geometry information that is useful for implementing
//  hinge and plate elements.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/09/2023 10:56:51
*///////////////////////////////////////////////////////////////////////////////
#ifndef DEFORMEDTRIANGLEGEOMETRY_HH
#define DEFORMEDTRIANGLEGEOMETRY_HH

#include "../Types.hh"

namespace elements {

template<class Real>
struct DeformedTriangleGeometry {
    static constexpr size_t NumCorners = 3;
    using CornerPositions = Eigen::Matrix<Real, NumCorners, 3, Eigen::RowMajor>;
    using V3d = Vec3_T<Real>;
    using M3d = Mat3_T<Real>;

    DeformedTriangleGeometry() { } // WARNING: leaves members uninitalized!

    DeformedTriangleGeometry(const CornerPositions &x) { configure(x); }

    //      2
    //      *
    //     / \
    //    1   0
    //   /     \
    // 0*---2---* 1
    void configure(const CornerPositions &x) {
        edgeVecs << (x.row(2) - x.row(1)).transpose(),
                    (x.row(0) - x.row(2)).transpose(),
                    (x.row(1) - x.row(0)).transpose();
        edgeVecDotProducts = edgeVecs.transpose() * edgeVecs;
        edgeLens = edgeVecDotProducts.diagonal().cwiseSqrt();
        normal = edgeVecs.col(0).cross(edgeVecs.col(1));
        Real dblArea = normal.norm();
        h = dblArea / edgeLens.array(); // height of the triangle over each edge.
        normal /= dblArea;
        unitEdgePerpendiculars = edgeVecs.colwise().cross(-normal) * edgeLens.asDiagonal().inverse();
    }

    V3d normal, edgeLens, h;
    M3d edgeVecs,               // e_i in *column* i
        unitEdgePerpendiculars, // ehatperp_i in *column* i
        edgeVecDotProducts;     // e_i . e_j in entry (i, j)
};

}

#endif /* end of include guard: DEFORMEDTRIANGLEGEOMETRY_HH */
