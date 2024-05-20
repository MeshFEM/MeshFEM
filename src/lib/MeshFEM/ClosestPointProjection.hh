#ifndef CLOSESTPOINTPROJECTION_HH
#define CLOSESTPOINTPROJECTION_HH
#include <memory>
#include "Types.hh"

namespace detail {

template<typename _CoordinateType>
struct SurfaceAABB;

}

template<typename _CoordinateType>
struct MESHFEM_EXPORT ClosestPointProjection {
    using Real = typename _CoordinateType::Scalar;
    static constexpr size_t D = _CoordinateType::RowsAtCompileTime;
    static_assert((D > 1) && (D < 4), "Only 2D, and 3D are supported");
    using MXd  = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using MXi  = Eigen::MatrixXi;
    using AABB = detail::SurfaceAABB<_CoordinateType>;

    ClosestPointProjection(const MXd &V, const MXi &F);

    template<class Mesh>
    ClosestPointProjection(const Mesh &m)
        : ClosestPointProjection(getV(m), getF(m)) { }

    struct ProjectionResult {
        _CoordinateType p;
        int element;
        VecMaxN_T<Real, D> barycoords;
        Real squaredDist;

        // Jacobian of the projected point with respect to the query point.
        // This is only computed when requested.
        MatMaxN_T<Real, D> dp_dq; 
    };

    ProjectionResult project(const VecN_T<Real, D> &q, bool computeJacobian = false) const;

    ~ClosestPointProjection(); // Out-of-line becuase of incomplete type SurfaceAABB

    int numVertices()       const { return m_V.rows(); }
    int numElements()       const { return m_F.rows(); }
    int numElementCorners() const { return m_F.cols(); }

    const MXd &V() const { return m_V; }
    const MXi &F() const { return m_F; }

private:
    MXd m_V;
    MXi m_F; // Note: 3D case could be a polyline or triangle surface!

    std::unique_ptr<AABB> m_aabb;
};
#endif /* end of include guard: CLOSESTPOINTPROJECTION_HH */
