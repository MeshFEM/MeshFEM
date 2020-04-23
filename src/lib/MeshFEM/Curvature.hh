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

template<class _FEMMesh>
struct GaussianCurvatureSensitivity;

template<size_t Deg>
struct GaussianCurvatureSensitivity<FEMMesh<2, Deg, Point3D>> {
    using Mesh = FEMMesh<2, Deg, Point3D>;
    using VecX = Eigen::VectorXd;
    using Vec3 = Vector3D;
    using Tri = typename Mesh::template EHandle<const Mesh>;

    GaussianCurvatureSensitivity(const Mesh &m)
        : m_mesh(m) { }

    const Mesh &mesh() const { return m_mesh; }

    static Vec3 edgelenSq(const Tri &tri) {
        return Vec3{(tri.node(2)->p - tri.node(1)->p).squaredNorm(),
                    (tri.node(0)->p - tri.node(2)->p).squaredNorm(),
                    (tri.node(1)->p - tri.node(0)->p).squaredNorm()};
    }

    static Vec3 cornerAngles(const Tri &tri) {
        Vec3 e0 = tri.node(2)->p - tri.node(1)->p, //      2
             e1 = tri.node(0)->p - tri.node(2)->p, //     / ^,
             e2 = tri.node(1)->p - tri.node(0)->p; // e1 /   \ e0
        return Vec3{angle(e2, -e1),                //   v     \,
                    angle(e0, -e2),                //  0------>1
                    angle(e1, -e0)};               //     e2
    }

    static Vec3 voronoiAreaContribs(const Tri &tri) {
        // 1/8 sum_i (a_i^2 cot(alpha_i) + b_i^2 cot(beta_i))
#if 0   // gradLambda-based version:
        //      grad lambda_i = e_i^perp / (2 A)
        //      grad lambda_i . grad lambda_j * A = -cos / (sin 2 A) * A = -1 / 2 cot
        //      ==> 1 / 8 cot(alpha_i) = -A / 4 (grad lambda_i . grad lambda_j)
        Real neg_A_div_4 = -tri->volume() * 0.25;
        const auto &gradLambdas = tri->gradBarycentric();
        Vec3 lSq_cot_div_8{neg_A_div_4 * (tri.node(2)->p - tri.node(1)->p).squaredNorm() * (gradLambdas.col(2).dot(gradLambdas.col(1))),
                           neg_A_div_4 * (tri.node(0)->p - tri.node(2)->p).squaredNorm() * (gradLambdas.col(0).dot(gradLambdas.col(2))),
                           neg_A_div_4 * (tri.node(1)->p - tri.node(0)->p).squaredNorm() * (gradLambdas.col(1).dot(gradLambdas.col(0)))};
#else   // Intrinsic, edge-length version:
        //      cos(alpha_i) = (l_j^2 + l_k^2 - l_i^2) / (2 l_j l_k) (law of cosines)
        //      sin(alpha_i) = 2 A / (l_j l_k)
        //      cot(alpha_i) = (l_j^2 + l_k^2 - l_i^2) / (4 A)

        Real inv_32A = 1.0 / (32 * tri->volume());
        Vec3 lSq = edgelenSq(tri);
        Vec3 lSq_cot_div_8{inv_32A * lSq[0] * (lSq[1] + lSq[2] - lSq[0]),
                           inv_32A * lSq[1] * (lSq[2] + lSq[0] - lSq[1]),
                           inv_32A * lSq[2] * (lSq[0] + lSq[1] - lSq[2])};
#endif
        return Vec3{lSq_cot_div_8[1] + lSq_cot_div_8[2],
                    lSq_cot_div_8[2] + lSq_cot_div_8[0],
                    lSq_cot_div_8[0] + lSq_cot_div_8[1]};

        // Alternative expression avoiding lSq_cot_div_8:
        // return inv_32A * Vec3{(lSq[1] + lSq[2]) * lSq[0] - std::pow(lSq[1] - lSq[2], 2),
        //                       (lSq[2] + lSq[0]) * lSq[1] - std::pow(lSq[2] - lSq[0], 2),
        //                       (lSq[0] + lSq[1]) * lSq[2] - std::pow(lSq[0] - lSq[1], 2)};
    }


    VecX voronoiAreas() const {
        VecX result = VecX::Zero(mesh().numVertices());
        for (const auto &tri : mesh().elements()) {
            Vec3 contrib = voronoiAreaContribs(tri);
            for (const auto &v : tri.vertices())
                result[v.index()] += contrib[v.localIndex()];
        }
        return result;
    }

    VecX mixedVoronoiAreas() const {
        VecX result = VecX::Zero(mesh().numVertices());
        for (const auto &tri : mesh().elements()) {
            Vec3 angles = cornerAngles(tri);
            int maxCorner;
            Real maxAngle = angles.maxCoeff(&maxCorner);
            Vec3 contrib;
            if (maxAngle > M_PI / 2) {
                Real A = tri->volume();
                contrib[ maxCorner         ] = 0.50 * A;
                contrib[(maxCorner + 1) % 3] = 0.25 * A;
                contrib[(maxCorner + 2) % 3] = 0.25 * A;
            }
            else { contrib = voronoiAreaContribs(tri); }
            for (const auto &v : tri.vertices())
                result[v.index()] += contrib[v.localIndex()];
        }
        return result;
    }

    // On interior vertices: Gaussian curvaure integrated over the
    // Voronoi/averaging region.
    // On boundary vertices: discrete geodesic curvature.
    VecX integratedK() const {
        const auto &m = mesh();
        VecX result = VecX::Constant(m.numVertices(), 2 * M_PI);
        for (const auto &bv : m.boundaryVertices())
            result[bv.volumeVertex().index()] = M_PI;

        for (const auto &tri : m.elements()) {
            Vec3 angles = cornerAngles(tri);
            for (const auto &v : tri.vertices())
                result[v.index()] -= angles[v.localIndex()];
        }

        return result;
    }

    VecX K() const { return integratedK().array() / mixedVoronoiAreas().array(); }

    // Tri Area
    // Angle
    // Voronoi Area
    // K (accumulate angles over tris)

    // Grad K
    //  (Accmulate over tris to one-ring stencil)
    //  (On average 7 * 3 = 21 variables per K)

    // Hess K (21 x 21 matrix) via autodiff on Grad K.
private:
    const Mesh &m_mesh;
};

// Gaussian curvature variation energy:
//      1/2 k^T L k
// Gradient:
//      (k^T L) grad k
// Hessian:
//      (grad k^T) L (grad k) + (k^T L) Hess k

#endif /* end of include guard: CURVATURE_HH */
