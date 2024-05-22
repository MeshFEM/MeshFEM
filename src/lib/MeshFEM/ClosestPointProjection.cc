#include "ClosestPointProjection.hh"
#include "libigl_aabb/point_simplex_squared_distance.h"
#include "libigl_aabb/AABB.h"

namespace detail {

template<typename _CoordinateType>
struct SurfaceAABB : public iglaabb::AABB<Eigen::Matrix<typename _CoordinateType::Scalar, Eigen::Dynamic, Eigen::Dynamic>, _CoordinateType::RowsAtCompileTime> {
    using Real = typename _CoordinateType::Scalar;
    static constexpr int N = _CoordinateType::RowsAtCompileTime;
    using MXd = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    using Base = iglaabb::AABB<MXd, N>;
    using Base::Base;
};

}

template<typename CT>
ClosestPointProjection<CT>::ClosestPointProjection(const MXd &V, const MXi &F)
    : m_V(V), m_F(F)
{
    if (V.cols() != D) throw std::runtime_error("V must have " + std::to_string(D) + " columns");
    if ((F.cols() < 2) || (F.cols() > 3)) throw std::runtime_error("Expected a polyline or triangle mesh");
    m_aabb = std::make_unique<AABB>();
    m_aabb->init(m_V, m_F);
}

template<typename CT>
typename ClosestPointProjection<CT>::ProjectionResult
ClosestPointProjection<CT>::project(const VecN_T<Real, D> &q, bool computeJacobian) const {
    int eidx;
    RowVecN_T<Real, D> closestPt;

    ProjectionResult result;
    result.squaredDist = m_aabb->squared_distance(m_V, m_F, q.transpose().eval(), result.element, closestPt);
    result.p = closestPt.transpose();

    // Get barycentric coordinates
    double dist;
    Eigen::Matrix<Real, 1, Eigen::Dynamic, Eigen::RowMajor, 1, /* MaxCols */ 3> baryCoords(m_F.cols());
    RowVecN_T<Real, D> dummy; // (reprojection--unneeded)
    // Compute barycentric coordinates of the closest point projection
    // (TODO: modify `squared_distance` to return barycentric coordinates directly)
    // Note, that we must use the original query point `q` here instead of `closestPt` because
    // the latter can lead to slightly interior baricentric coordinates for points that
    // project precisely onto vertices or edges.
    iglaabb::point_simplex_squared_distance<D>(q.transpose().eval(), m_V, m_F, result.element, dist, dummy, baryCoords);

    result.barycoords = baryCoords.transpose();

    if (!computeJacobian) return result;

    auto &dp_dq = result.dp_dq;
    dp_dq.resize(D, D);

    // Determine projection sensitivities based on which barycentric
    // coordinates are nonzero; the Jacobian of the projection is
    // `dp_dq = sum_i t_i \otimes t_i`, where {t_i} are an orthonormal basis
    // for the (sub)simplex containing the closest point projection.
    // For example, if the closest point projection lies within a:
    //      vertex   ==>  dp_dq = 0
    //      edge e   ==>  dp_dq = e.normalize() * e.normalize().transpose()
    //      face f   ==>  dp_dq = I - n * n.transpose(), where n is the face normal
    std::array<int, 3> nonzeroLoc;
    int numNonzero = 0;
    for (int i = 0; i < baryCoords.size(); ++i) {
        if (baryCoords[i] == 0.0) continue;
        // It is extremely unlikely a vertex will be closest to a point/edge if this is not a stable association.
        // Therefore we assume even for smoothish surfaces that points are constrained to lie on their closest
        // simplex.
        nonzeroLoc[numNonzero++] = i;
    }

    auto f = m_F.row(result.element);
    if (numNonzero == 1) { dp_dq.setZero(); return result; } // Vertex
    if (numNonzero == 2) { // Edge
        CT e = (m_V.row(f[nonzeroLoc[0]]) - m_V.row(f[nonzeroLoc[1]])).normalized();
        dp_dq = e * e.transpose();
        return result;
    }

    if (numNonzero == 3) { // Face
        dp_dq.setIdentity();
        if constexpr (D == 2) return result;
        if constexpr (D == 3) {
            CT e1 = m_V.row(f[nonzeroLoc[1]]) - m_V.row(f[nonzeroLoc[0]]),
               e2 = m_V.row(f[nonzeroLoc[2]]) - m_V.row(f[nonzeroLoc[0]]);
            CT n = e1.cross(e2).normalized();
            dp_dq -= n * n.transpose();
            return result;
        }
        throw std::logic_error("Faces should only exist in 2D or 3D");
    }

    throw std::runtime_error("Unexpected number of nonzero barycentric coordinates: " + std::to_string(numNonzero));
}

template<typename CT> ClosestPointProjection<CT>::~ClosestPointProjection() { }

////////////////////////////////////////////////////////////////////////////////
// Explicit instantiations
////////////////////////////////////////////////////////////////////////////////
template struct MESHFEM_EXPORT ClosestPointProjection<VecN_T<Real, 2>>;
template struct MESHFEM_EXPORT ClosestPointProjection<VecN_T<Real, 3>>;
