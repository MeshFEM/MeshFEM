// Prototype of a Cholesky-plus-Schur-complement approach for enforcing
// global rigid motion constraints. This can be accelerated, e.g., with faster
// batched solves and Hessian column extraction.
#ifndef RIGIDMOTIONFACTORIZATION_HH
#define RIGIDMOTIONFACTORIZATION_HH

#include <MeshFEM/newton_optimizer/NewtonHessian.hh>
#include <MeshFEMSparse/Solvers/make_cholesky_factorizer.hh>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>

namespace MeshFEM {

enum class RigidMotionConstraints { Translations, All };

// Inverse of an unshifted planar Hessian restricted to R^perp. R uses the
// Euclidean nodal inner product and is frozen throughout a Taylor expansion.
// Pins are auxiliary: solve() releases them and enforces distributed constraints.
// Only a sparse Hessian is supported (no dense variables/low-rank terms/KKT).
class RigidMotionFactorization {
public:
    using VXd = Eigen::VectorXd;
    using MXd = Eigen::MatrixXd;

    RigidMotionFactorization(const VXd &x, RigidMotionConstraints constraints,
                             CholeskyProvider provider = CholeskyProvider::CatamariNesdisParallel)
        : m_constraints(constraints), m_solver(make_cholesky_factorizer(provider)) {
        setGeometry(x);
        if (constraints == RigidMotionConstraints::All && x.size() < 8)
            throw std::invalid_argument("Full rigid constraints require at least four vertices for the block-pinned solver");
        // Pin whole two-variable blocks, retaining the block Cholesky path.
        Eigen::Map<const Eigen::Matrix<double, 2, Eigen::Dynamic>> uv(x.data(), 2, x.size() / 2);
        auto centered = (uv.colwise() - uv.rowwise().mean()).eval();
        Eigen::Index a, b;
        centered.colwise().squaredNorm().maxCoeff(&a);
        m_pins = {size_t(2 * a), size_t(2 * a + 1)};
        if (m_constraints == RigidMotionConstraints::All) {
            (uv.colwise() - uv.col(a)).colwise().squaredNorm().maxCoeff(&b);
            m_pins.insert(m_pins.end(), {size_t(2 * b), size_t(2 * b + 1)});
            std::sort(m_pins.begin(), m_pins.end());
        }
        m_solver->setSuppressWarnings(true);
    }

    // Keep the auxiliary pins fixed for symbolic reuse; update R at a new iterate.
    void setGeometry(const VXd &x) {
        m_ready = false;
        if (x.size() < 6 || x.size() % 2 || !x.allFinite() || (m_R.rows() && x.size() != m_R.rows()))
            throw std::invalid_argument("Rigid-motion constraints require finite planar vertex coordinates of unchanged size");
        const Eigen::Index n = x.size() / 2;
        m_R.setZero(x.size(), m_constraints == RigidMotionConstraints::All ? 3 : 2);
        const double scale = 1.0 / std::sqrt(double(n));
        for (Eigen::Index i = 0; i < n; ++i) { m_R(2*i, 0) = scale; m_R(2*i+1, 1) = scale; }
        if (m_constraints == RigidMotionConstraints::All) {
            Eigen::Map<const Eigen::Matrix<double, 2, Eigen::Dynamic>> uv(x.data(), 2, n);
            auto centered = (uv.colwise() - uv.rowwise().mean()).eval();
            const double norm = centered.norm();
            if (!(norm > 0)) throw std::invalid_argument("Cannot constrain rotation of coincident vertices");
            for (Eigen::Index i = 0; i < n; ++i) {
                m_R(2*i, 2) = -centered(1, i) / norm;
                m_R(2*i+1, 2) = centered(0, i) / norm;
            }
        }
    }

    void factorizeSymbolic(const NewtonHessian &H) {
        m_ready = false;
        m_validate(H);
        m_solver->factorizeSymbolic(*H.H_ss, m_pins);
    }

    // A failed principal or border Cholesky leaves the solve invalid. The caller
    // may ask its projection controller for another Hessian; no shift is added.
    void factorizeNumeric(const NewtonHessian &H) {
        m_ready = false;
        m_validate(H);
        m_solver->factorizeNumeric(*H.H_ss);
        if (!m_solver->checkPosDef()) throw std::runtime_error("Pinned Hessian is not positive definite");
        if (m_constraints == RigidMotionConstraints::Translations) {
            // HT=0 implies K=P E A^{-1} E^T P, P=I-TT^T. No border needed.
            m_ready = true;
            return;
        }

        const Eigen::Index p = m_pins.size(), n = m_R.rows();
        MXd E(n, p), F(p, p), Rf = m_R, Rp(p, m_R.cols());
        VXd unit = VXd::Zero(n);
        for (Eigen::Index j = 0; j < p; ++j) {
            unit[m_pins[j]] = 1;
            E.col(j) = H.H_ss->apply(unit); // Extract just the four pinned columns.
            unit[m_pins[j]] = 0;
        }
        for (Eigen::Index j = 0; j < p; ++j) {
            F.row(j) = E.row(m_pins[j]);
            Rp.row(j) = m_R.row(m_pins[j]);
            E.row(m_pins[j]).setZero();
            Rf.row(m_pins[j]).setZero();
        }
        MXd AE(n, p);
        for (Eigen::Index j = 0; j < p; ++j) AE.col(j) = m_solver->solve(E.col(j).eval());
        m_AR.resize(n, Rf.cols());
        for (Eigen::Index j = 0; j < Rf.cols(); ++j) m_AR.col(j) = m_solver->solve(Rf.col(j).eval());

        MXd M = Rf.transpose() * m_AR;
        m_M.compute(0.5 * (M + M.transpose()));
        if (m_M.info() != Eigen::Success) throw std::runtime_error("Rigid constraint Gram matrix is not positive definite");
        MXd D = Rp - E.transpose() * m_AR;
        MXd MD = m_M.solve(D.transpose());
        m_B = -AE - m_AR * MD;
        MXd S = F - E.transpose() * AE + D * MD;
        m_S.compute(0.5 * (S + S.transpose()));
        if (m_S.info() != Eigen::Success) throw std::runtime_error("Hessian is not positive definite on the rigid-constraint subspace");
        m_ready = true;
    }

    void solve(const VXd &b, VXd &x) const {
        if (!m_ready) throw std::runtime_error("Rigid-motion solve requires a successful factorization at the current geometry");
        if (b.size() != m_R.rows() || !b.allFinite()) throw std::invalid_argument("Invalid rigid-motion solve RHS");
        if (m_constraints == RigidMotionConstraints::Translations) {
            VXd rhs = b;
            m_center(rhs);
            x = m_solver->solve(rhs);
            m_center(x);
            return;
        }
        VXd v = m_solver->solve(b);
        VXd rhs = m_B.transpose() * b;
        for (size_t j = 0; j < m_pins.size(); ++j) rhs[j] += b[m_pins[j]];
        VXd y = m_S.solve(rhs);
        // v is zero at the pins, so R^T v = R_f^T v_f.
        x = v - m_AR * m_M.solve(m_R.transpose() * v) + m_B * y;
        for (size_t j = 0; j < m_pins.size(); ++j) x[m_pins[j]] = y[j];
    }

    const CholeskyFactorizerBase &solver() const { return *m_solver; }

private:
    void m_validate(const NewtonHessian &H) const {
        H.validate();
        if (H.numVars() != size_t(m_R.rows()) || H.numDenseVars() || H.low_rank_rank() || H.C_s.size() || H.C_d.size())
            throw std::invalid_argument("Rigid-motion factorization requires a planar sparse Hessian without dense, low-rank, or equality-constraint blocks");
    }
    static void m_center(VXd &x) {
        Eigen::Map<Eigen::Matrix<double, 2, Eigen::Dynamic>> uv(x.data(), 2, x.size() / 2);
        const Eigen::Vector2d mean = uv.rowwise().mean();
        uv.colwise() -= mean;
    }
    const RigidMotionConstraints m_constraints;
    std::unique_ptr<CholeskyFactorizerBase> m_solver;
    std::vector<size_t> m_pins;
    MXd m_R, m_AR, m_B;
    Eigen::LLT<MXd> m_M, m_S;
    bool m_ready = false;
};

} // namespace MeshFEM
#endif
