////////////////////////////////////////////////////////////////////////////////
// NewtonHessian.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Implements a flexible format for representing and factorizing Hessians that
// arise from optimization problems with a variety of sparsity structures.
// The core of the representation is our block sparse matrix datastructure
// `BlockCSCHessian`, which is used to represent `H_ss`, the "sparse part" of
// the Hessian. This sparse part can then be augmented in three ways:
//
//   i) padding with dense rows/columns:
//      [H_ss H_sd]
//      [H_ds H_dd],
//      where the `d` subscripts correspond to the "dense"/"global" variables
//      declared in the `OptimizationVarStructure` instance (currently stored
//      inside the `BlockCSCHessian` instance); and
//  ii) adding a symmetric low-rank term of the form `V V^T` (of rank `r`); and
// iii) Imposing equality constraints `C x = d` via Lagrange multipliers:
//      [A C^T][x] = [b]
//      [C   0][λ]   [d]
//
// The low-rank term can be handled using the Sherman-Morrison-Woodbury formula,
// but this is equivalent to (conceptually) introducing `r` new dense variables
// called λ and employing our implementation of i) on the padded system:
//      [H_ss   H_sd   V_s][x_s] = [-g_s]
//      [H_ds   H_dd   V_d][x_d]   [-g_d]
//      [V_s^T  V_d^T -I_r][λ  ]   [   0]
// where V = [V_s; V_d].
//
// The system above has the high-level structure:
//      [H_ss  B][x] = [b]
//      [ B^T  D][y]   [c]
// which can be solved using block Gaussian elimination:
//      [H_ss                    B][x] = [b                 ]
//      [   0  D - B^T H_ss^{-1} B][y]   [c - B^T H_ss^{-1}b]
// denoting the Schur complement of H_ss by S := (D - B^T H_ss^{-1} B)^{-1},
// we find:
//      y = S^{-1} (c - B^T H_ss^{-1} b)
//      x = H_ss^{-1} (b - B y),
//  Evaluating these formulas requires sparse solves for each column of `B`;
//  after that, S and the vector `B^T H_ss^{-1} b` can be computed via matrix
//  multiplication. Then a single final sparse solve can be used to obtain `x`.
//
//  From the symmetric block Gaussian elimination formula:
//      [A   B] = [I        0][A 0][I A^-1 B]
//      [B^T D] = [B^T A^-1 I][0 S][0      I]
//  we observe that the full Hessian is positive definite if and only if
//  both `A` and `S` are positive definite.
//  Thus after standard Hessian projection/modification strategies are used
//  to make `A` positive definite, the full Hessian can be made positive
//  definite by projecting the dense block `S` to be positive definite.
//  The latter projection can be done using a dense Eigenvalue decomposition
//  since `S` should be small (of row/col size `r + n_d` where `n_d` is the
//  number of dense variables). This should be small enough for the anticipated
//  use cases that we can use a dense Eigenvalue decomposition to invert `S`,
//  simultaneously applying a Hessian projection.
//
//  Implementing equality constraints (iii) with Lagrange multipliers can be
//  done in essentially the same way:
//      [H_ss    B C_1^T][x]   [b]
//      [ B^T    D C_2^T][y] = [c]
//      [ C_1  C_2     0][λ]   [r]
//  again running symmetric block Gaussian elimination:
//      [I                         0  0][H_ss 0  0][I                         0  0]^T[x]   [b]
//      [B^T H_ss^{-1}             I  0][   0 S  0][B^T H_ss^{-1}             I  0]  [y] = [c]
//      [C_1 H_ss^{-1} C_2' H_ss^{-1} I][   0 0  L][C_1 H_ss^{-1} C_2' H_ss^{-1} I]  [y]   [r]
//  Where C_2' = C_2 - C_1 H_ss^{-1} B and
//        L = - C_1 H_ss^{-1} C_1^T - C_2' H_ss^{-1} C_2'^T
//  In this case, `L` is negative definite, and we do *not* want to apply a
//  Hessian projection to it since we wish to solve for a saddle point
//  (a minimum with respect to x, y; maximum with respect to λ).
//  Implicit in this approach is an assumption that [H_ss B; B^T D] will be
//  positive definite at a solution to the constrained minimization problem.
//  This will not be the case if the energy landscape has a direction of
//  negative curvature that is normal to the constraint manifold.
//  This implementation will supplant the former `KKTSolver` class.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  12/29/2024 20:20:30
*///////////////////////////////////////////////////////////////////////////////
#ifndef NEWTONHESSIAN_HH
#define NEWTONHESSIAN_HH

#include "WorkingSet.hh"
#include "NewtonOptions.hh"
#include <MeshFEM/BlockCSCHessian.hh>

struct NewtonHessian {
    std::unique_ptr<BlockCSCHessianBase> H_ss;

    // Storage of dense blocks induced by "global" variables.
    Eigen::MatrixXd H_sd, H_dd;
    // Storage of the low-rank term:
    //  [V_s; V_d] [V_s; V_d]^T
    Eigen::MatrixXd V_s, V_d;

    // Storage of equality constraints
    //  [C_s; C_d] [x_s; x_d]
    Eigen::MatrixXd C_s, C_d;
};

// Copy-on-write-style optimization for Hessian that only occasionally needs
// modification (when working set is nonempty).
// Assumes that the matrix passed to `set` stays alive for the duration of this
// object's lifetime.
struct OptionallyModifiedHessian {
    OptionallyModifiedHessian() : m_H(nullptr) { }

    OptionallyModifiedHessian(const SuiteSparseMatrix &H_cached) { set(H_cached); }

    void set(const SuiteSparseMatrix &H_cached) {
        m_H = &H_cached;
        m_H_tmp.reset();
    }

    const SuiteSparseMatrix *get() const { return m_H; }
          SuiteSparseMatrix *getMutable() {
        if (m_H == nullptr) throw std::runtime_error("Matrix doesn't exist");
        if (!m_H_tmp) {
            m_H_tmp = std::make_unique<SuiteSparseMatrix>(*get());
            m_H = m_H_tmp.get();
        }
        return m_H_tmp.get();
    }

    operator const SuiteSparseMatrix &() const { return *get(); }
    explicit operator bool() const { return get() != nullptr; }
private:
    const SuiteSparseMatrix *m_H;
    std::unique_ptr<SuiteSparseMatrix> m_H_tmp;
};

// Cache to avoid repeated re-evaluation of our rough Hessian eigenvalue
// estimate. Uses the trace to detect when the Hessian's spectrum has changed
// substantially.
struct MESHFEM_EXPORT CachedHessianL2Norm {
    CachedHessianL2Norm() { reset(); }

    static constexpr double TRACE_TOL = 0.5;
    Real get(const NewtonProblem &p) {
        const auto &H = p.hessian();
        Real tr = H.trace();
        if (std::abs(tr - hessianTrace) > TRACE_TOL * std::abs(hessianTrace)) {
            hessianTrace = tr;
            hessianL2Norm = p.hessianL2Norm();
        }
        return hessianL2Norm;
    }

    void reset() { hessianTrace  = std::numeric_limits<Real>::max();
                   hessianL2Norm = 1.0; }
private:
    Real hessianTrace, hessianL2Norm;
};

struct NewtonOptimizer;

// A factorization type for solving systems involving a `NewtonHessian`.
struct NewtonHessianFactorization {
    NewtonHessianFactorization(std::shared_ptr<NewtonProblem> p,
                               const NewtonOptimizerOptions &options)
        : m_options(options), m_problem(p) { }

    // Compute/recompute the Hessian factorization.
    Real update(const WorkingSet &ws, Real &beta, const Real betaMin);

    Real tauScale() const { return (m_options.hessianScaledBeta ? m_cachedHessianL2Norm.get(*m_problem) : 1.0) / m_problem->metricL2Norm(); }

    void solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const {
        if (!exists()) throw std::runtime_error("Factorization doesn't exist.");
        solver().solve(b, x);
    }

    void updateSymbolicFactorization(bool force = false) {
        m_problem->updateSparsityPattern();
        if (force || (m_problem->sparsityPatternID() != m_factorizedSparsityPatternID)) {
            m_solver->factorizeSymbolic(m_problem->hessianSparsityPattern(), m_problem->fixedVars());
            m_factorizedSparsityPatternID = m_problem->sparsityPatternID();
        }
    }

    CholeskyFactorizerBase &solver() {
        if (!m_solver || (m_solver->provider() != m_options.factorizer)) {
            m_solver = make_cholesky_factorizer(m_options.factorizer);
            updateSymbolicFactorization();
        }
        return *m_solver;
    }

    const CholeskyFactorizerBase &solver() const {
        if (!m_solver) throw std::runtime_error("Solver doesn't exist.");
        return *m_solver;
    }

    bool exists() const { return m_solver && m_solver->hasFactorization(); }

private:
    friend struct NewtonOptimizer;
    void m_beginningOptimization() { m_cachedHessianL2Norm.reset(); }

    const NewtonOptimizerOptions &m_options; // Owned by our owner (`NewtonOptimizer`).
    std::shared_ptr<NewtonProblem> m_problem;
    std::shared_ptr<CholeskyFactorizerBase> m_solver;

    mutable CachedHessianL2Norm m_cachedHessianL2Norm;

    // Record the sparsity pattern for which the most recent symbolic
    // factorization was computed by `m_solver`.
    size_t m_factorizedSparsityPatternID = std::numeric_limits<size_t>::max(); // None
};

////////////////////////////////////////////////////////////////////////////////
// DEPRECATED: (TODO: remove)
////////////////////////////////////////////////////////////////////////////////
// Cache temporaries and solve the KKT system:
// [H   a][   x  ] = [   b    ]
// [a^T 0][lambda]   [residual]
struct MESHFEM_EXPORT KKTSolver {
    Eigen::VectorXd Hinv_a, a;
    template<class Factorizer>
    void update(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> a_) {
        a = a_;
        solver.solve(a, Hinv_a);
    }

    Real           lambda(Eigen::Ref<const Eigen::VectorXd> Hinv_b, const Real residual = 0) const { return (a.dot(Hinv_b) - residual) / a.dot(Hinv_a); }
    Eigen::VectorXd solve(Eigen::Ref<const Eigen::VectorXd> Hinv_b, const Real residual = 0) const { return Hinv_b - lambda(Hinv_b, residual) * Hinv_a; }

    template<class Factorizer>
    Eigen::VectorXd operator()(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> b, const Real residual = 0) const { return solve(solver, b, residual); }

    template<class Factorizer>
    Eigen::VectorXd solve(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> b, const Real residual = 0) const {
        Eigen::VectorXd Hinv_b;
        solver.solve(b.eval(), Hinv_b);
        return solve(Hinv_b, residual);
    }
};

#endif /* end of include guard: NEWTONHESSIAN_HH */
