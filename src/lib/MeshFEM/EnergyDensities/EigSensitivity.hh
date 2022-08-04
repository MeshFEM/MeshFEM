////////////////////////////////////////////////////////////////////////////////
// EigSensitivity.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  First and second derivatives of symmetric 2x2 matrix eigenvalues.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/05/2019 10:32:17
////////////////////////////////////////////////////////////////////////////////
#ifndef EIGSENSITIVITY_HH
#define EIGSENSITIVITY_HH

#include <Eigen/Dense>
#include <array>
#include <MeshFEM/EnergyDensities/Tensor.hh>

// 2x2 case only for now, also can be sped up.
template<typename Real>
struct EigSensitivity {
    using V2d  = Eigen::Matrix<Real, 2, 1>;
    using M2d  = Eigen::Matrix<Real, 2, 2>;

    EigSensitivity() { }

    template<typename Derived>
    EigSensitivity(const Eigen::MatrixBase<Derived> &A) { setMatrix(A); }

    template<typename Derived>
    void setMatrix(const Eigen::MatrixBase<Derived> &A) {
        static_assert((Derived::RowsAtCompileTime == 2) && (Derived::ColsAtCompileTime == 2), "Only 2x2 supported for now");

        const Real a_minus_c = A(0, 0) - A(1, 1);
        const Real b = A(0, 1);
        if (std::abs(b - A(1, 0)) > 1e-15) throw std::runtime_error("Only symmetric matrices are supported");
        // d := descriminant of characteristic quadratic
        const Real sqrt_d = std::sqrt(a_minus_c * a_minus_c + 4 * b * b);
        m_Lambda << 0.5 * (A.trace() + sqrt_d),
                    0.5 * (A.trace() - sqrt_d);

        V2d q0(a_minus_c + sqrt_d, 2 * b);
        q0.normalize();
        m_Q.col(0) = q0;
        m_Q.col(1) << -q0[1], q0[0];

        if (sqrt_d < 1e-14) m_degenerate = true;
        else                m_degenerate = false;
    }

    auto      q(size_t i) const { return m_Q.col(i); }
    Real lambda(size_t i) const { return m_Lambda[i]; }

    // Access Eigendecomposition
    const M2d &     Q() const { return m_Q; }
    const V2d &Lambda() const { return m_Lambda; }

    V2d dLambda(const M2d &dA) const { return (m_Q.transpose() * dA * m_Q).diagonal(); }
    M2d dLambda(size_t i) const { return q(i) * q(i).transpose(); }
    Real dLambda(size_t i, const M2d &dA) const { return q(i).dot(dA * q(i)); }

    // Only correct for symmetric dA_b!
    V2d d2Lambda(const M2d &dA_a, const M2d &dA_b) const {
        if (m_degenerate) return V2d(0.0, 0.0);
        Real d2lambda_0 = (2.0 / (m_Lambda[0] - m_Lambda[1])) * (q(0).dot(dA_a * q(1))) * (q(0).dot(dA_b * q(1)));
        return V2d(d2lambda_0, -d2lambda_0);
    }

    // Only correct for symmetric dA!
    M2d delta_dLambda(size_t i, const M2d &dA) const {
        if (m_degenerate) return M2d::Zero();
        double sign = (i == 0) ? 1.0 : -1.0;
        M2d result = ((sign * (2.0 / (m_Lambda[0] - m_Lambda[1])) * (q(0).dot(dA * q(1)))) * q(0)) * q(1).transpose();
        symmetrize(result);
        return result;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    M2d m_Q;
    V2d m_Lambda;

    bool m_degenerate;
};

#endif /* end of include guard: EIGSENSITIVITY_HH */
