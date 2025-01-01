////////////////////////////////////////////////////////////////////////////////
// newton_optimizer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Newton-type optimization method for large, sparse problems.
//  This is Newton's method with a (sparse) Hessian modification strategy to
//  deal with the indefinite case.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/27/2018 11:29:48
////////////////////////////////////////////////////////////////////////////////
#ifndef NEWTON_OPTIMIZER_HH
#define NEWTON_OPTIMIZER_HH

#include <vector>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Solvers/make_cholesky_factorizer.hh>

#include "NewtonOptions.hh"
#include "NewtonProblem.hh"
#include "WorkingSet.hh"
#include "NewtonHessian.hh"

#include <MeshFEM_export.h>

struct MESHFEM_EXPORT NewtonOptimizer {
    NewtonOptimizer(std::shared_ptr<NewtonProblem> p)
        : prob(p), m_hessianFactorization(p, options) { }

    void setFixedVars(const std::vector<size_t> &fixedVars) {
        prob->setFixedVars(fixedVars);
        m_hessianFactorization.updateSymbolicFactorization(/* force = */ true);
    }

    ConvergenceReport optimize();
    ConvergenceReport optimize(WorkingSet &ws);

    Real newton_step(Eigen::VectorXd &step, const Eigen::VectorXd &neg_g, const WorkingSet &ws, Real &beta, const Real betaMin, const bool feasibility = false);

    // Calculate a Newton step with empty working set and default beta/betaMin.
    Real newton_step(Eigen::VectorXd &step, const Eigen::VectorXd &neg_g) {
        Real beta = options.beta;
        const Real betaMin = std::min(beta, 1e-6);
        WorkingSet ws(*prob);
        return newton_step(step, neg_g, ws, beta, betaMin);
    }

    // Update the factorizations of the Hessian/KKT system with the current
    // iterate's Hessian. This is necessary for sensitivity analysis after
    // optimize() has been called: when optimization terminates either because
    // the problem is solved or the iteration limit is reached, solver/kkt_solver
    // hold values from the previous iteration (before the final linesearch
    // step).
    void update_factorizations(const WorkingSet &ws) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("update_factorizations");
        m_hessianFactorization.update(ws, options.beta, std::min(options.beta, 1e-6));
    }

    void update_factorizations() { update_factorizations(WorkingSet(*prob)); }

    const NewtonProblem &get_problem() const { return *prob; }
          NewtonProblem &get_problem()       { return *prob; }

    const NewtonHessianFactorization &hessianFactorization() const { return m_hessianFactorization; }
          NewtonHessianFactorization &hessianFactorization()       { return m_hessianFactorization; }

    NewtonOptimizerOptions options;

private:
    std::shared_ptr<NewtonProblem> prob;
    NewtonHessianFactorization m_hessianFactorization;
};

#endif /* end of include guard: NEWTON_OPTIMIZER_HH */
