////////////////////////////////////////////////////////////////////////////////
// RigidMotionPins.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Support for constraining the rigid motion of a deformable object using
//  6 simple variable pin constraints for objects embedded in 3D (3 in 2D).
//
//  This is done by first rotating the object's deformed configuration so that
//  specially chosen vertices lie on the global coordinate axes.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/07/2020 14:43:34
////////////////////////////////////////////////////////////////////////////////
#ifndef RIGIDMOTIONPINS_HH
#define RIGIDMOTIONPINS_HH

#include "Types.hh"
#include <type_traits>
#include <utility>
#include "newton_optimizer/dense_newton.hh"

template<class Object, typename /* enabler */ = std::true_type>
struct RigidMotionPins;

template<class Object>
struct RigidMotionPins<Object, std::integral_constant<bool, Object::N == 3>> {
    using PinVars     = std::array<size_t, 6>;
    using PinVertices = std::array<size_t, 3>;
    using PinInfo     = std::tuple<PinVars, PinVertices>;
    static PinInfo run(Object &obj) {
        using M3d = Eigen::Matrix<typename Object::Real, 3, 3>;
        // Note: we only allow vertices (not edge nodes) as pins
        // to simplify the traversal to influenced elements.
        auto P = obj.deformedPositions().topRows(obj.mesh().numVertices()).eval();

        // Pick centermost vertex "c" and place it at the origin.
        int c_idx;
        auto cm = P.colwise().mean().eval();
        (P.rowwise() - cm).rowwise().squaredNorm().minCoeff(&c_idx);
        auto c_pos = P.row(c_idx).eval();
        P.rowwise() -= c_pos;

        // Pick "p" defining the unit x axis vector "x_hat"
        int p_idx;
        P.rowwise().squaredNorm().maxCoeff(&p_idx);
        auto x_hat = P.row(p_idx).normalized().transpose().eval();

        // Pick "q" defining a robust x-y plane (and thus a unit y axis vector "y_hat")
        int q_idx;
        P.rowwise().cross(x_hat).rowwise().squaredNorm().maxCoeff(&q_idx);
        auto y_hat = (P.row(q_idx).transpose() - x_hat.dot(P.row(q_idx)) * x_hat).normalized().eval();
        auto z_hat = x_hat.cross(y_hat).normalized().eval();

        // Detect cases where points p and q lie the XY plane but the rotation ends up
        // flipping this plane over; we undo this by flipping the y (and z) axis vectors.
        if (std::abs(z_hat[2] - (-1)) < 1e-9) {
            y_hat *= -1;
            z_hat *= -1;
        }

        M3d R; // inverse of the [xhat, yhat, zhat] frame matrix, rotating these vectors to the global coordinate axes.
        R << x_hat.transpose(),
             y_hat.transpose(),
             z_hat.transpose();

        obj.applyRigidTransform(R, -(R * c_pos.transpose()));

        return std::make_tuple(PinVars{{
            // Pin center
            3 * c_idx + 0ul,
            3 * c_idx + 1ul,
            3 * c_idx + 2ul,
            // Pin rotation around the z and y axes by constraining the
            // (y, z) components of the point at [x, 0, 0]
            3 * p_idx + 1ul,
            3 * p_idx + 2ul,
            // Pin rotation around the x axis by constraining the z component
            // of the point at [x_q, y, 0]
            3 * q_idx + 2ul
        }}, PinVertices{{ size_t(c_idx), size_t(p_idx), size_t(q_idx) }});
    }
};

template<class Object>
struct RigidMotionPins<Object, std::integral_constant<bool, Object::N == 2>> {
    using PinVars     = std::array<size_t, 3>;
    using PinVertices = std::array<size_t, 2>;
    using PinInfo     = std::tuple<PinVars, PinVertices>;
    static PinInfo run(Object &obj) {
        using M2d = Eigen::Matrix<typename Object::Real, 2, 2>;
        auto P = obj.deformedPositions();

        // Pick centermost vertex "c" and place it at the origin.
        int c_idx;
        auto cm = P.colwise().mean().eval();
        (P.rowwise() - cm).rowwise().squaredNorm().minCoeff(&c_idx);
        auto c_pos = P.row(c_idx).eval();
        P.rowwise() -= c_pos;

        // Pick "p", defining the unit x axis vector "x_hat"
        int p_idx;
        P.rowwise().squaredNorm().maxCoeff(&p_idx);
        auto x_hat = P.row(p_idx).normalized().transpose().eval();

        decltype(x_hat) y_hat(-x_hat[1], x_hat[0]); // 90deg counter-clockwise rotation

        M2d R; // inverse of the [xhat, yhat, zhat] frame matrix, rotating these vectors to the global coordinate axes.
        R << x_hat.transpose(),
             y_hat.transpose();

        obj.applyRigidTransform(R, -(R * c_pos.transpose()));

        return std::make_tuple(PinVars{{
            // Pin center
            2 * c_idx + 0ul,
            2 * c_idx + 1ul,
            // Pin rotation by constraining the y component of the point at [x, 0]
            2 * p_idx + 1ul
        }}, PinVertices{{ size_t(c_idx), size_t(p_idx) }});
    }
};

////////////////////////////////////////////////////////////////////////////////
// Rigid motion pin artifact filtering.
////////////////////////////////////////////////////////////////////////////////
// When constraining rigid motion with variable pins, intermediate Newton
// steps tend to introduce high local distortion around the pins which slows
// convergence. This is especially problematic when a modified Hessian is used
// which does not have infinitesimal rigid motions in its nullspace,
// but seems to be an issue in general--presumably due to *finite* rotations
// not lying in the nullspace of the local quadratic model.
//
// We mitigate these artifacts by minimizing the energy stored in the mesh with
// respect to only the pinned vertices (holding all others fixed) and then
// applying a global rigid transformation that places these pinned vertices
// back where the constraints want them.
////////////////////////////////////////////////////////////////////////////////
template<class Object>
struct SingleVertexOptProblem {
    static constexpr size_t N = Object::N;
    using Real     = typename Object::Real;
    using VarType  = Eigen::Matrix<Real, N, 1>;
    using HessType = Eigen::Matrix<Real, N, N>;

    SingleVertexOptProblem(Object &obj, size_t vi)
        : m_obj(obj), m_vi(vi)
    {
        const auto &m = obj.mesh();
        if (vi >= m.numVertices()) throw std::runtime_error("Vertex index out of bounds");
        m.vertex(vi).visitIncidentElements([this](size_t ei) { m_incidentElements.push_back(ei); });
    }

    Real energy() const {
        Real result = 0.0;
        for (const size_t ei: m_incidentElements)
            result += m_obj.elementEnergy(ei);
        return result;
    }

    VarType gradient() const {
        VarType result(VarType::Zero());
        for (const size_t ei: m_incidentElements) {
            const auto &e = m_obj.mesh().element(ei);
            const auto g = m_obj.elementGradient(ei);
            bool found = false;
            for (const auto v : e.vertices()) {
                if (size_t(v.index()) == m_vi) {
                    result += g.template segment<N>(N * v.localIndex());
                    found = true;
                }
            }
            if (found == false) throw std::logic_error("Vertex not found in influenced element");
        }
        return result;
    }

    HessType hessian() const {
        HessType result(HessType::Zero());
        for (const size_t ei: m_incidentElements) {
            const auto &e = m_obj.mesh().element(ei);
            const auto H = m_obj.elementHessian(ei, /* disable Hessian projection (Dense newton solver can deal with indefiniteness better) */ true);
            bool found = false;
            for (const auto v : e.vertices()) {
                if (size_t(v.index()) == m_vi) {
                    const size_t vo = N * v.localIndex();
                    for (size_t c_a = 0; c_a < N; ++c_a) {
                        for (size_t c_b = 0; c_b < N; ++c_b)
                            result(c_a, c_b) += H[Object::perElementHessianFlattening(vo + c_a, vo + c_b)];
                    }
                    found = true;
                }
            }
            if (found == false) throw std::logic_error("Vertex not found in influenced element");
        }
        return result;
    }

    static constexpr size_t numVars() { return N; }

    void setVars(const VarType &vars) {
        auto fullVars = m_obj.getVars();
        fullVars.template segment<N>(N * m_vi) = vars;
        m_obj.setVars(fullVars);
    }

    const VarType getVars() const { return m_obj.getVars().template segment<N>(N * m_vi); }

    void solve() { dense_newton(*this, 100, 1e-10, false); }

private:
    Object &m_obj;
    const size_t m_vi;
    std::vector<size_t> m_incidentElements;
};

template<class Derived, class V3d>
std::enable_if_t<Derived::ColsAtCompileTime == 3, Eigen::Matrix<typename Derived::Scalar, 3, 3>>
rotationZeroingPins(const std::array<size_t, 3> &pinVertices, const Eigen::MatrixBase<Derived> &P, const V3d &c_pos, const V3d &x_hat) {
    using M3d = Eigen::Matrix<typename Derived::Scalar, 3, 3>;

    V3d y_hat = P.row(pinVertices[2]).transpose() - c_pos;
    y_hat = (y_hat - x_hat.dot(y_hat) * x_hat).normalized().eval();
    V3d z_hat = x_hat.cross(y_hat).normalized().eval();
    M3d R; // inverse of the [xhat, yhat, zhat] frame matrix, rotating these vectors to the global coordinate axes.
    R << x_hat.transpose(),
         y_hat.transpose(),
         z_hat.transpose();
    return R;
}

template<class Derived, class V2d>
std::enable_if_t<Derived::ColsAtCompileTime == 2, Eigen::Matrix<typename Derived::Scalar, 2, 2>>
rotationZeroingPins(const std::array<size_t, 2> &/* pinVertices */, const Eigen::MatrixBase<Derived> &/* P */, const V2d &/* c_pos */, const V2d &x_hat) {
    using M2d = Eigen::Matrix<typename Derived::Scalar, 2, 2>;
    V2d y_hat(-x_hat[1], x_hat[0]);
    M2d R; // inverse of the [xhat, yhat, zhat] frame matrix, rotating these vectors to the global coordinate axes.
    R << x_hat.transpose(),
         y_hat.transpose();
    return R;
}

template<class Object>
void filterRMPinArtifacts(Object &obj, const typename RigidMotionPins<Object>::PinVertices &pinVertices) {
    constexpr size_t N = Object::N;
    static_assert((N == 3) || (N == 2), "Unsupported dimension.");

    for (size_t i = 0; i < pinVertices.size(); ++i)
        SingleVertexOptProblem<Object>(obj, pinVertices[i]).solve();

    using VNd = Eigen::Matrix<typename Object::Real, N, 1>;
    auto P = obj.deformedPositions();

    VNd c_pos = P.row(pinVertices[0]);
    VNd x_hat = (P.row(pinVertices[1]).transpose() - c_pos).normalized();
    auto R = rotationZeroingPins(pinVertices, P, c_pos, x_hat);

    obj.applyRigidTransform(R, -(R * c_pos));
}

#endif /* end of include guard: RIGIDMOTIONPINS_HH */
