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

template<class Object, typename /* enabler */ = std::true_type>
struct RigidMotionPins;

template<class Object>
struct RigidMotionPins<Object, std::integral_constant<bool, Object::N == 3>> {
    using PinVars = std::array<size_t, 6>;
    static PinVars run(Object &obj) {
        using M3d = Eigen::Matrix<typename Object::Real, 3, 3>;
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

        // Pick "q", defining the unit y axis vector "y_hat"
        int q_idx;
        P.rowwise().cross(x_hat).rowwise().squaredNorm().maxCoeff(&q_idx);
        auto y_hat = (P.row(q_idx).transpose() - x_hat.dot(P.row(q_idx)) * x_hat).normalized().eval();
        auto z_hat = x_hat.cross(y_hat).normalized().eval();

        M3d R; // inverse of the [xhat, yhat, zhat] frame matrix, rotating these vectors to the global coordinate axes.
        R << x_hat.transpose(),
             y_hat.transpose(),
             z_hat.transpose();

        obj.applyRigidTransform(R, -(R * c_pos.transpose()));

        return PinVars{{
            // Pin center
            3 * c_idx + 0ul,
            3 * c_idx + 1ul,
            3 * c_idx + 2ul,
            // Pin rotation around the z and y axes by constraining the
            // (y, z) components of the point at [x, 0, 0]
            3 * p_idx + 1ul,
            3 * p_idx + 2ul,
            // Pin rotation around the x axis by constraining the z component
            // of the point at [0, y, 0]
            3 * q_idx + 2ul
        }};
    }
};

template<class Object>
struct RigidMotionPins<Object, std::integral_constant<bool, Object::N == 2>> {
    using PinVars = std::array<size_t, 3>;
    static PinVars run(Object &obj) {
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

        return PinVars{{
            // Pin center
            2 * c_idx + 0ul,
            2 * c_idx + 1ul,
            // Pin rotation by constraining the y component of the point at [x, 0]
            2 * p_idx + 1ul
        }};
    }
};

#endif /* end of include guard: RIGIDMOTIONPINS_HH */
