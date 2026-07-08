#ifndef NEWTONHESSIANFACTORIZATION_HH
#define NEWTONHESSIANFACTORIZATION_HH

#include "NewtonOptions.hh"
#include "NewtonHessian.hh"

namespace MeshFEM {

struct WorkingSet;
struct NewtonOptimizer;
struct NewtonProblem;

// Cache to avoid repeated re-evaluation of our rough Hessian eigenvalue
// estimate. Uses the trace to detect when the Hessian's spectrum has changed
// substantially.
struct MESHFEM_EXPORT CachedHessianL2Norm {
    CachedHessianL2Norm() { reset(); }

    static constexpr double TRACE_TOL = 0.5;
    Real get(const NewtonProblem &p);

    void reset() { hessianTrace  = std::numeric_limits<Real>::max();
                   hessianL2Norm = 1.0; }
private:
    Real hessianTrace, hessianL2Norm;
};

// A factorization type for solving systems involving a `NewtonHessian`.
// Factorizes the block matrix:
//      [H_ss B]
//      [B^T  D]
//      where B = [H_sd V_s]
//      and   D = [H_dd     V_d]
//                [V_d^T   -I_r]
// using block Gaussian elimination
struct MESHFEM_EXPORT NewtonHessianFactorization final : public BorderedSparseFactorization {
    NewtonHessianFactorization(std::shared_ptr<NewtonProblem> p, const NewtonOptimizerOptions &options);

    // Compute/recompute the Hessian factorization.
    Real update(const WorkingSet &ws, Real &beta, const Real betaMin);

    Real tauScale() const;

    // The symbolic factorization must be updated if either the sparsity pattern
    // changes or the fixed variables set changes.
    // Since the fixed variables set cannot change during the optimization, we avoid the
    // overhead of comparing the sets unless `m_fixedVarsCouldHaveChanged` is true.
    void updateSymbolicFactorization();

    void solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    using BorderedSparseFactorization::solver;
    CholeskyFactorizerBase &solver();

    // When recording matrices to reproduce a sequence of sparsity patterns used
    // in factorization, it helps to dump the final sparsity pattern (along with
    // its corresponding update count) so that we know how many calls to
    // `m_updateSparsityPattern` were made after the last symbolic factorization
    // update was triggered. This method can be used to force this final dump.
    void recordFinalSymbolicMatrix() const;

    virtual ~NewtonHessianFactorization();
private:
    friend struct NewtonOptimizer;

    Real m_updateSparseFactorization(const NewtonHessian &H, const WorkingSet &ws, Real &beta, const Real betaMin);

    void m_beginningOptimization() {
        m_cachedHessianL2Norm.reset();
        m_fixedVarsCouldHaveChanged = true;
    }

    const NewtonOptimizerOptions &m_options; // Owned by our owner (`NewtonOptimizer`).
    std::shared_ptr<NewtonProblem> m_problem;

    mutable CachedHessianL2Norm m_cachedHessianL2Norm;
    bool m_fixedVarsCouldHaveChanged = true; // Fixed vars can change only between separate runs of the optimizer.

    // Record the sparsity pattern for which the most recent symbolic
    // factorization was computed by `m_solver`.
    size_t m_factorizedSparsityPatternID = std::numeric_limits<size_t>::max(); // None

    Real m_shift = 0.0; // The multiple of the identity matrix added during
                        // factorization to make the Hessian positive definite.
};

} // namespace MeshFEM

#endif /* end of include guard: NEWTONHESSIANFACTORIZATION_HH */
