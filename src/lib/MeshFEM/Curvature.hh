////////////////////////////////////////////////////////////////////////////////
// Curvature.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Functions to evaluate curvautre quantities and their shape derivatives.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/22/2020 04:07:08
////////////////////////////////////////////////////////////////////////////////
#ifndef CURVATURE_HH
#define CURVATURE_HH
#include "FEMMesh.hh"
#include "AutomaticDifferentiation.hh"

template<class _FEMMesh>
struct GaussianCurvatureSensitivity;

template<size_t Deg>
struct GaussianCurvatureSensitivity<FEMMesh<2, Deg, Point3D>> {
    using Mesh = FEMMesh<2, Deg, Point3D>;
    using VXd = Eigen::VectorXd;
    using V3d = Vector3D;
    using M3d = Eigen::Matrix3d;

    using Tri = typename Mesh::template EHandle<const Mesh>;

    GaussianCurvatureSensitivity(const Mesh &m)
        : m_mesh(m) { }

    const Mesh &mesh() const { return m_mesh; }

    static V3d edgelenSq(const Tri &tri) {
        return V3d{(tri.node(2)->p - tri.node(1)->p).squaredNorm(),
                   (tri.node(0)->p - tri.node(2)->p).squaredNorm(),
                   (tri.node(1)->p - tri.node(0)->p).squaredNorm()};
    }

    static V3d cornerAngles(Eigen::Ref<const V3d> p0, Eigen::Ref<const V3d> p1, Eigen::Ref<const V3d> p2) {
        V3d e0 = p2 - p1,           //      2
            e1 = p0 - p2,           //     / ^,
            e2 = p1 - p0;           // e1 /   \ e0
        return V3d{angle(e2, -e1),  //   v     \,
                   angle(e0, -e2),  //  0------>1
                   angle(e1, -e0)}; //     e2
    }

    static V3d cornerAngles(const Tri &tri) { return cornerAngles(tri.node(0)->p, tri.node(1)->p, tri.node(2)->p); }

    static V3d voronoiAreaContribs(const Tri &tri) {
        // 1/8 sum_i (a_i^2 cot(alpha_i) + b_i^2 cot(beta_i))
#if 0   // gradLambda-based version:
        //      grad lambda_i = e_i^perp / (2 A)
        //      grad lambda_i . grad lambda_j * A = -cos / (sin 2 A) * A = -1 / 2 cot
        //      ==> 1 / 8 cot(alpha_i) = -A / 4 (grad lambda_i . grad lambda_j)
        Real neg_A_div_4 = -tri->volume() * 0.25;
        const auto &gradLambdas = tri->gradBarycentric();
        V3d lSq_cot_div_8{neg_A_div_4 * (tri.node(2)->p - tri.node(1)->p).squaredNorm() * (gradLambdas.col(2).dot(gradLambdas.col(1))),
                          neg_A_div_4 * (tri.node(0)->p - tri.node(2)->p).squaredNorm() * (gradLambdas.col(0).dot(gradLambdas.col(2))),
                          neg_A_div_4 * (tri.node(1)->p - tri.node(0)->p).squaredNorm() * (gradLambdas.col(1).dot(gradLambdas.col(0)))};
#else   // Intrinsic, edge-length version:
        //      cos(alpha_i) = (l_j^2 + l_k^2 - l_i^2) / (2 l_j l_k) (law of cosines)
        //      sin(alpha_i) = 2 A / (l_j l_k)
        //      cot(alpha_i) = (l_j^2 + l_k^2 - l_i^2) / (4 A)

        Real inv_32A = 1.0 / (32 * tri->volume());
        V3d lSq = edgelenSq(tri);
        V3d lSq_cot_div_8{inv_32A * lSq[0] * (lSq[1] + lSq[2] - lSq[0]),
                          inv_32A * lSq[1] * (lSq[2] + lSq[0] - lSq[1]),
                          inv_32A * lSq[2] * (lSq[0] + lSq[1] - lSq[2])};
#endif
        return V3d{lSq_cot_div_8[1] + lSq_cot_div_8[2],
                   lSq_cot_div_8[2] + lSq_cot_div_8[0],
                   lSq_cot_div_8[0] + lSq_cot_div_8[1]};

        // Alternative expression avoiding lSq_cot_div_8:
        // return inv_32A * V3d{(lSq[1] + lSq[2]) * lSq[0] - std::pow(lSq[1] - lSq[2], 2),
        //                      (lSq[2] + lSq[0]) * lSq[1] - std::pow(lSq[2] - lSq[0], 2),
        //                      (lSq[0] + lSq[1]) * lSq[2] - std::pow(lSq[0] - lSq[1], 2)};
    }


    VXd voronoiAreas() const {
        VXd result = VXd::Zero(mesh().numVertices());
        for (const auto tri : mesh().elements()) {
            V3d contrib = voronoiAreaContribs(tri);
            for (const auto v : tri.vertices())
                result[v.index()] += contrib[v.localIndex()];
        }
        return result;
    }

    VXd mixedVoronoiAreas() const {
        VXd result = VXd::Zero(mesh().numVertices());
        for (const auto tri : mesh().elements()) {
            V3d angles = cornerAngles(tri);
            int maxCorner;
            Real maxAngle = angles.maxCoeff(&maxCorner);
            V3d contrib;
            if (maxAngle > M_PI / 2) {
                Real A = tri->volume();
                contrib[ maxCorner         ] = 0.50 * A;
                contrib[(maxCorner + 1) % 3] = 0.25 * A;
                contrib[(maxCorner + 2) % 3] = 0.25 * A;
            }
            else { contrib = voronoiAreaContribs(tri); }
            for (const auto v : tri.vertices())
                result[v.index()] += contrib[v.localIndex()];
        }
        return result;
    }

    // On interior vertices: Gaussian curvaure integrated over the
    // Voronoi/averaging region.
    // On boundary vertices: discrete geodesic curvature.
    VXd integratedK() const {
        const auto &m = mesh();
        VXd result = VXd::Constant(m.numVertices(), 2 * M_PI);
        for (const auto bv : m.boundaryVertices())
            result[bv.volumeVertex().index()] = M_PI;

        for (const auto tri : m.elements()) {
            V3d angles = cornerAngles(tri);
            for (const auto v : tri.vertices())
                result[v.index()] -= angles[v.localIndex()];
        }

        return result;
    }

    VXd K() const { return integratedK().array() / mixedVoronoiAreas().array(); }

    // Mostly for debugging...
    VXd deltaVoronoiAreas(Eigen::Ref<const VXd> deltaP, bool mixed = false) const {
        VXd result = VXd::Zero(mesh().numVertices());
        for (const auto tri : mesh().elements()) {
            M3d corners;
            corners << tri.node(0)->p, tri.node(1)->p, tri.node(2)->p;

            std::array<M3d, 3> gradContrib;
            if (mixed) gradContrib = gradMixedVoronoiAreaContribs(corners, *tri);
            else       gradContrib =      gradVoronoiAreaContribs(corners, *tri);

            M3d deltaCorners;
            deltaCorners << deltaP.segment<3>(3 * tri.vertex(0).index()),
                            deltaP.segment<3>(3 * tri.vertex(1).index()),
                            deltaP.segment<3>(3 * tri.vertex(2).index());

            for (const auto v : tri.vertices())
                result[v.index()] += (gradContrib[v.localIndex()].transpose() * deltaCorners).trace();
        }
        return result;
    }

    // On interior vertices: Gaussian curvaure integrated over the
    // Voronoi/averaging region.
    // On boundary vertices: discrete geodesic curvature.
    VXd deltaIntegratedK(Eigen::Ref<const VXd> deltaP) const {
        VXd result = VXd::Zero(mesh().numVertices());
        for (const auto tri : mesh().elements()) {
            M3d corners;
            corners << tri.node(0)->p, tri.node(1)->p, tri.node(2)->p;

            auto gradAngles = gradCornerAngles(corners, *tri);

            M3d deltaCorners;
            deltaCorners << deltaP.segment<3>(3 * tri.vertex(0).index()),
                            deltaP.segment<3>(3 * tri.vertex(1).index()),
                            deltaP.segment<3>(3 * tri.vertex(2).index());

            for (const auto v : tri.vertices())
                result[v.index()] -= (gradAngles[v.localIndex()].transpose() * deltaCorners).trace();
        }

        return result;
    }

    VXd deltaK(Eigen::Ref<const VXd> deltaP) const {
        VXd result = VXd::Zero(mesh().numVertices());

        const VXd Kint = integratedK();
        const VXd va   = mixedVoronoiAreas();
        for (const auto tri : mesh().elements()) {
            M3d corners;
            corners << tri.node(0)->p, tri.node(1)->p, tri.node(2)->p;

            auto gradAngles = gradCornerAngles(corners, *tri);
            auto gradVA     = gradMixedVoronoiAreaContribs(corners, *tri);

            M3d deltaCorners;
            deltaCorners << deltaP.segment<3>(3 * tri.vertex(0).index()),
                            deltaP.segment<3>(3 * tri.vertex(1).index()),
                            deltaP.segment<3>(3 * tri.vertex(2).index());

            for (const auto v : tri.vertices()) {
                size_t li = v.localIndex();
                size_t vi = v.index();
                M3d negGradKContrib = (Kint[vi] / (va[vi] * va[vi])) * gradVA[li]
                                    +  gradAngles[li] / va[vi];
                result[v.index()] -= (negGradKContrib.transpose() * deltaCorners).trace();
            }
        }

        return result;
    }

    // Hess K (21 x 21 matrix) via autodiff on Grad K.
private:
    const Mesh &m_mesh;

    // Derivative of each corner of triangle element e's Voronoi area with respect to each corner position.
    // The corner positions are collected in "corners".
    // result[vi](:, vj) holds the gradient of vi's voronoi area with respect to vertex vj.
    // (in each column of the output).
    template<class M3d_, class EmbeddedElement_>
    static std::array<M3d_, 3> gradVoronoiAreaContribs(const M3d_ &corners, const EmbeddedElement_ &e) {
        using Real_ = typename M3d_::Scalar;
        using V3d_  = typename Eigen::Matrix<Real_, 3, 1>;

        Real_ inv_32A = 1.0 / (32 * e.volume());

        V3d_ lSq{(corners.col(2) - corners.col(1)).squaredNorm(),
                 (corners.col(0) - corners.col(2)).squaredNorm(),
                 (corners.col(1) - corners.col(0)).squaredNorm()};

        std::array<M3d_, 3> grad_lSq;
        grad_lSq[0] << V3d_::Zero(), 2 * (corners.col(1) - corners.col(2)), 2 * (corners.col(2) - corners.col(1));
        grad_lSq[1] << 2 * (corners.col(0) - corners.col(2)), V3d_::Zero(), 2 * (corners.col(2) - corners.col(0));
        grad_lSq[2] << 2 * (corners.col(0) - corners.col(1)), 2 * (corners.col(1) - corners.col(0)), V3d_::Zero();

        Eigen::Matrix<Real_, 3, 3> grad_inv_32A = -e.gradBarycentric() / (32 * e.volume());

        std::array<M3d_, 3> grad_lSq_cot_div_8{{
            grad_inv_32A * lSq[0] * (lSq[1] + lSq[2] - lSq[0]) + inv_32A * grad_lSq[0] * (lSq[1] + lSq[2] - lSq[0]) + inv_32A * lSq[0] * (grad_lSq[1] + grad_lSq[2] - grad_lSq[0]),
            grad_inv_32A * lSq[1] * (lSq[2] + lSq[0] - lSq[1]) + inv_32A * grad_lSq[1] * (lSq[2] + lSq[0] - lSq[1]) + inv_32A * lSq[1] * (grad_lSq[2] + grad_lSq[0] - grad_lSq[1]),
            grad_inv_32A * lSq[2] * (lSq[0] + lSq[1] - lSq[2]) + inv_32A * grad_lSq[2] * (lSq[0] + lSq[1] - lSq[2]) + inv_32A * lSq[2] * (grad_lSq[0] + grad_lSq[1] - grad_lSq[2])
        }};

        return std::array<M3d_, 3>{{grad_lSq_cot_div_8[1] + grad_lSq_cot_div_8[2],
                                    grad_lSq_cot_div_8[2] + grad_lSq_cot_div_8[0],
                                    grad_lSq_cot_div_8[0] + grad_lSq_cot_div_8[1]}};
    }

    template<class M3d_, class EmbeddedElement_>
    static std::array<M3d_, 3> gradMixedVoronoiAreaContribs(const M3d_ &corners, const EmbeddedElement_ &e) {
        // Note: angle computation is not autodiffed, but angles are only used
        // for a non-differentiable branch anyway...
        V3d angles = cornerAngles(stripAutoDiff(corners.col(0)), stripAutoDiff(corners.col(1)), stripAutoDiff(corners.col(2)));
        int maxCorner;
        Real maxAngle = angles.maxCoeff(&maxCorner);
        if (maxAngle < M_PI / 2) { return gradVoronoiAreaContribs(corners, e); }

        std::array<M3d_, 3> gradContrib;
        auto vol = e.volume();
        gradContrib[ maxCorner         ] = (0.50 * vol) * e.gradBarycentric();
        gradContrib[(maxCorner + 1) % 3] = (0.25 * vol) * e.gradBarycentric();
        gradContrib[(maxCorner + 2) % 3] = (0.25 * vol) * e.gradBarycentric();

        return gradContrib;
    }

    template<class M3d_, class EmbeddedElement_>
    static std::array<Mat3_T<typename M3d_::Scalar>, 3> gradCornerAngles(const M3d_ &corners, const EmbeddedElement_ &e) {
        //      2
        //     / ^,
        // e1 /   \ e0
        //   v     \,
        //  0------>1
        //     e2
        M3d_ eperp_invlen;
        for (int i = 0; i < 3; ++i){
            eperp_invlen.col(i) = e.normal().cross(corners.col((i + 2) % 3) - corners.col((i + 1) % 3) );
            eperp_invlen.col(i) /= eperp_invlen.col(i).squaredNorm();
        }

        std::array<M3d_, 3> result;
        for (int i = 0; i < 3; ++i) {
            result[i].col((i    ) % 3) =  eperp_invlen.col((i + 2) % 3) + eperp_invlen.col((i + 1) % 3);
            result[i].col((i + 1) % 3) = -eperp_invlen.col((i + 2) % 3);
            result[i].col((i + 2) % 3) = -eperp_invlen.col((i + 1) % 3);
        }

        return result;
    }
};

// Gaussian curvature variation energy:
//      1/2 k^T L k
// Gradient:
//      (k^T L) grad k
// Hessian:
//      (grad k^T) L (grad k) + (k^T L) Hess k

#endif /* end of include guard: CURVATURE_HH */
