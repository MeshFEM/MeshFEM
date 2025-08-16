#include "newton_optimizer.hh"
#include "../AutomaticDifferentiation.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/ParallelVectorOps.hh>
#include <Eigen/src/Core/Matrix.h>

#include <MeshFEM/Solvers/MatrixRecorder.hh>
#include "NewtonHessian.hh"

// Solve the Newton system `H d = -g`, modifying H to be pos. def. if it is indefinite.
// Returns "tau", the coefficient of the metric term that was added to make the Hessian positive definite.
// "-tau" can be interpreted as an estimate (lower bound) for the smallest generalized eigenvalue for "H d = lambda M d"
// (Returns 0 if the Hessian is already positive definite).
// Upon return, "solver" holds a factorization of the matrix:
//     (H + tau (M / ||M||_2))
Real NewtonOptimizer::newton_step(Eigen::VectorXd &step, const Eigen::VectorXd &neg_g, const WorkingSet &ws, Real &beta, const Real betaMin, const bool feasibility) {
    BENCHMARK_SCOPED_TIMER_SECTION ns_timer("newton_step");
    step.resize(neg_g.size());
    if (&ws.problem() != &get_problem()) throw std::runtime_error("Working set is for a different problem");

    Real tau = NAN; // tau is unknown/undefined if we're reusing an old factorization; no negative curvature direction will be attempted by caller.

    auto &hUpdtCtr = options.getHessianUpdateController();
    const bool reuseFactorization = m_hessianFactorization.exists() && !hUpdtCtr.needsUpdate() && (ws.size() == 0); // TODO: Reusing factorizations with bound constraints needs more care
    if (reuseFactorization) hUpdtCtr.reusedHessian();
    else                    tau = m_hessianFactorization.update(ws, beta, betaMin);

    // Solve Newton/KKT system using the current factorization.
    if (ws.size()) m_hessianFactorization.solve(ws.getFreeComponent(neg_g), step);
    else           m_hessianFactorization.solve(neg_g, step);

    if (prob->hasLEQConstraint()) {
        // TODO: reenable
        throw std::runtime_error("Unimplemented (LEQ constraints are disabled during refactoring)");
        // const Real r = feasibility ? prob->LEQConstraintResidual() : 0.0;
        // step = kkt_solver.solve(step, r);
    }

    // ws.validateStep(step);

    return tau;
}

ConvergenceReport NewtonOptimizer::optimize() {
    // Indices of the bound constraints in our working set.
    WorkingSet workingSet(*prob);
    return optimize(workingSet);
}

// Return `true` on success, `false` on failure.
// Record the value at which the line search terminated in `alpha`.
using VXd = Eigen::VectorXd;
bool backtrackingLineSearch(NewtonProblem &prob, const NewtonOptimizerOptions &options, const VXd &vars, const VXd &step, const Real currEnergy, const Real directionalDerivative, Real &alpha, Real initialAlpha = 1.0) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Backtracking");
    // Simple backtracking line search to ensure a sufficient decrease
    prob.lineSearchBegan(step, directionalDerivative);

    Real feasible_alpha;
    size_t blocking_idx;
    std::tie(feasible_alpha, blocking_idx) = prob.feasibleStepLength(vars, step);

    // To add multiple nearby bounds to the working set at once, we allow the
    // step to overshoot the bounds (note: variables will be clamped to the bounds anyway before
    // evaluating the objective). Then all bounds violated by the step length obtaining
    // sufficient decrease are added to the working set.
    alpha = std::min(initialAlpha, feasible_alpha * 2);

    // Also clamp the step to a feasible size permitted by the problem
    alpha = alpha * std::min(1.0, prob.customFeasibleStepLength(vars, alpha * step));

    const Real c_1 = options.armijo_c1;
    size_t bit;

    Eigen::VectorXd steppedVars;
    for (bit = 0; bit < options.nbacktrack_iter; ++bit) {
        steppedVars = vars + alpha * step;
        prob.applyBoundConstraintsInPlace(steppedVars);
        prob.setVars(steppedVars);
        const Real steppedEnergy = prob.objective();
        const Real sufficientDecrease = -c_1 * alpha * directionalDerivative;
        Real decrease = currEnergy - steppedEnergy;
        if (std::isfinite(steppedEnergy) && !std::isfinite(currEnergy))
            decrease = safe_numeric_limits<Real>::max(); // always accept steps from invalid to valid states.
        // Terminate line search successfully if a sufficient decrease is achieved
        // (or if we cannot expect to evaluate the energy decrease accurately
        // enough to measure a sufficient decrease--and the energy does not
        // increase significantly)
        if  ((decrease >= sufficientDecrease)
                || (std::abs(sufficientDecrease) < 1e-8 * std::abs(currEnergy)
                        && (decrease > -1e-10 * std::abs(currEnergy)))) {
            break;
        }

        if (alpha > feasible_alpha) {
            // It's possible that our slight overshooting and clamping to the bounds did not achieve a sufficient
            // decrease whereas a step to the first violated bound would; make sure we try this exact step too
            // before continuing the backtracking search.
            alpha = feasible_alpha;
        }
        else {
            alpha *= options.backtrack_shrink_factor;
        }

        if (bit == options.nbacktrack_iter - 1) {
            std::cout << "Backtracking failure with:" << std::endl
                      << "Curr energy: " << currEnergy << std::endl
                      << "Stepped energy: " << steppedEnergy << std::endl
                      << "sufficientDecrease: " << sufficientDecrease << std::endl
                      << "decrease: " << decrease << std::endl
                      << std::endl;
        }
    }

    const bool failed = bit == options.nbacktrack_iter;
    return !failed;
}

ConvergenceReport NewtonOptimizer::optimize(WorkingSet &workingSet) {
    size_t ngd_fallback_steps = options.ngd_fallback_steps; // maximum number of gradient descent steps to take as a fallback when backtracking for the newton step fails.

    prob->setUseIdentityMetric(options.useIdentityMetric);
    prob->writeIterates = options.writeIterateFiles;

    // Respect any bound constraints before we start
    if (prob->numBoundConstraints() && !(prob->feasible(prob->getVars())))
        prob->setVars(prob->applyBoundConstraints(prob->getVars()));

    Eigen::VectorXd vars, step;

    Real beta = options.beta;
    const Real betaMin = std::min(beta, 1e-10); // Initial shift "tau" to use when an indefinite matrix is detected.
    m_hessianFactorization.m_beginningOptimization();

    if (prob->hasLEQConstraint()) {
        if (!prob->LEQConstraintIsFeasible()) {
            if (options.feasibilitySolve) {
                // std::cout << "Running feasibility solve with residual " << prob->LEQConstraintResidual() << ", energy " << prob->objective() << std::endl;
                prob->iterationCallback(0);
                newton_step(step, -prob->gradient(true), workingSet, beta, betaMin, true);
                // We must take a full step to ensure feasibility
                // TODO: use multiple iterations and a line search to get feasible?
                prob->setVars(prob->applyBoundConstraints(step + prob->getVars()));
                // std::cout << "Post feasibility solve residual " << prob->LEQConstraintResidual() << ", energy " << prob->objective() << std::endl;
            }
            else {
                prob->LEQStepFeasible();
            }
        }
        prob->setVars(prob->applyBoundConstraints(prob->getVars()));
        if (!prob->LEQConstraintIsFeasible()) {
            std::cout << "Post feasibility step residual: " << prob->LEQConstraintResidual() << std::endl;
            throw std::runtime_error("Iterate still infeasible");
        }
    }

    const auto &fixedVars = prob->fixedVars();
    {
        const size_t nv = prob->numVars();
        for (size_t fv : fixedVars)
            if (fv >= nv) throw std::runtime_error("fixedVars out of bounds");
    }
    auto zeroOutFixedVarsInPlace = [&](Eigen::VectorXd &g) { for (size_t var : fixedVars) g[var] = 0.0; };
    auto zeroOutFixedVars = [&](Eigen::VectorXd g) { zeroOutFixedVarsInPlace(g); return g; };

    ConvergenceReport report;

    Real alpha = 0;
    bool isIndefinite = false;
    auto reportIterate = [&](size_t i, Real energy, Real g_free_norm, bool forcePrintIfVerbose) {
        prob->writeIterateFiles(i);
        report.addEntry(energy, g_free_norm, alpha, isIndefinite, prob->hessianWasProjected());

        if (options.verbose && (((i % options.verbose) == 0) || forcePrintIfVerbose)) {
            std::cout << i << '\t';
            report.printEntry();
            if (i % options.stdoutFlushInterval == 0)
                std::cout << std::flush;
        }
    };

    BENCHMARK_SCOPED_TIMER_SECTION nitTimer("Newton iterations"); // Stopped at the end of this function.
    size_t it;
    Eigen::VectorXd za, neg_g, neg_g_ws_free_storage;
    Eigen::VectorXd *neg_g_ws_free_ptr = &neg_g;
    if (prob->hasLEQConstraint()) { za = zeroOutFixedVars(prob->LEQConstraintMatrix()); }
    // Kill off components of "v" in the span of the LEQ constraint vectors
    auto projectOutLEQConstrainedComponents = [&](Eigen::VectorXd &v) { if (prob->hasLEQConstraint()) v -= za * (v.dot(za) / za.squaredNorm()); };

    options.getHessianProjectionController().reset();
    options.getHessianUpdateController()    .reset();

    for (it = 1; it <= options.niter; ++it) {
        BENCHMARK_SCOPED_TIMER_SECTION it_timer("Newton iterate");

        Real currEnergy, g_free_norm;
        { BENCHMARK_SCOPED_TIMER_SECTION t2("Preamble");

        // std::cout << "pre-update gradient: " << zeroOutFixedVars(prob->gradient(false)).norm() << std::endl;
        {
            BENCHMARK_SCOPED_TIMER_SECTION cbTimer("Callback");
            bool earlyExit = prob->iterationCallback(it);
            if (earlyExit) {
                if (options.verbose) std::cout << "Early termination requested by user callback" << std::endl;
                break;
            }
        }

        // Note: we allow the iteration callback to modify the variables!
        // (in case the user wants to run some custom projection/filter at the start
        //  of each Newton iteration).
        vars = prob->getVars();

        currEnergy = prob->objective();
        neg_g = -prob->gradient(true);

        zeroOutFixedVarsInPlace(neg_g);
        projectOutLEQConstrainedComponents(neg_g);

        // Gradient with respect to the "free" variables (components corresponding to fixed/actively constrained variables zero-ed out)
        if (workingSet.size()) {
            neg_g_ws_free_storage = workingSet.getFreeComponent(neg_g);
            neg_g_ws_free_ptr = &neg_g_ws_free_storage;
            g_free_norm = neg_g_ws_free_storage.norm();

            // Free variables in the working set from their bound constraints, if necessary
            bool ws_updated = workingSet.remove_if([&](size_t bc_idx) {
                    bool shouldRemove = prob->boundConstraint(bc_idx).shouldRemoveFromWorkingSet(neg_g, g_free_norm);
                    if (shouldRemove && options.verboseWorkingSet) { std::cout << "Removed constraint " << bc_idx << " from working set" << std::endl; }
                    return shouldRemove;
                });

            if (ws_updated) {
                neg_g_ws_free_storage = workingSet.getFreeComponent(neg_g);
                g_free_norm = neg_g_ws_free_storage.norm();
            }
        }
        else {
            neg_g_ws_free_ptr = &neg_g;
            g_free_norm = neg_g.norm();
        }

        if ((!isIndefinite) && (g_free_norm < options.gradTol)) {
            report.success = true;
            break;
        }

        } // End of 'Preamble' timer


        { BENCHMARK_SCOPED_TIMER_SECTION t2("Compute descent direction");

        Real old_beta = beta;
        Real tau;
        try {
            tau = newton_step(step, *neg_g_ws_free_ptr, workingSet, beta, betaMin);
            prob->setLastFactorizationShiftMagnitude(tau);
        }
        catch (std::exception &e) {
            // Tau ran away
            std::cout << "Caught exception in newton_step: " << e.what() << std::endl;
            break;
        }
        isIndefinite = (tau != 0.0);

        // Only add in negative curvature directions when "tau" is a reasonable estimate for the smallest eigenvalue and the gradient has become small.
        if (options.useNegativeCurvatureDirection && ((tau > old_beta) || (tau == betaMin)) && (g_free_norm < 100 * options.gradTol)) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Negative curvature dir");
            // std::cout.precision(19);
            std::cout << "Computing negative curvature direction for scaled tau = " << tau / prob->metricL2Norm() << '\n';
            Eigen::VectorXd d;
            // TODO: update to account for dense terms of the NewtonHessianFactorization...
            if (options.useIdentityMetric || !(prob->providesMetric())) {
                d = negativeCurvatureDirection(m_hessianFactorization.solver(), nullptr, 1e-3);
            }
            else {
                OptionallyModifiedHessian M(prob->metric());
                if (workingSet.size()) fixVariablesInWorkingSet(*prob, *M.getMutable(), workingSet);
                d = negativeCurvatureDirection(m_hessianFactorization.solver(), M.get(), 1e-3);
            }

            Real dnorm = d.norm();
            if (dnorm != 0.0) {
                workingSet.getFreeComponentInPlace(d); // Enforce the active bound constraints.
                // {
                //     const SuiteSparseMatrix &H = prob->hessian();
                //     H.applyRaw(d.data(), tmp.data());
                //     Real lambda = d.dot(tmp);
                //     std::cout << "Found negative curvature direction with eigenvalue " << lambda << std::endl;
                // }
                if (d.dot(*neg_g_ws_free_ptr) > 0) d *= -1; // Move in the opposite direction as the gradient (So we still produce a descent direction)
                const Real cd = prob->characteristicDistance(d);
                if (cd <= 0) // problem doesn't provide one
                    step += std::sqrt(step.squaredNorm() / d.squaredNorm()) * d; // TODO: find a better balance between newton step and negative curvature.
                else {
                    step += 1e-2 * (d / cd);
                }
            }
            else { std::cout << "Negative curvature direction calculation failed" << std::endl; }
        }

        } // End of 'Compute descent direction' timer

        Real directionalDerivative = -(neg_g.dot(step));
        // if (options.verbose)
        //     std::cout << "Found step with directional derivative: " << directionalDerivative << std::endl;

        options.getHessianProjectionController().notifyDirectionalDerivative(directionalDerivative);
        bool line_search_succeeded = backtrackingLineSearch(*prob, options, vars, step, currEnergy, directionalDerivative, alpha);
        prob->lineSearchTerminated();

        reportIterate(it - 1, currEnergy, g_free_norm, false); // Record iterate statistics, now that we know alpha, isIndefinite
        prob->customIterateReport(report);

        // Add to the working set all bounds encountered by the step of length "alpha"
        for (size_t bci = 0; bci < prob->numBoundConstraints(); ++bci) {
            if (alpha >= prob->boundConstraint(bci).feasibleStepLength(vars, step)) {
                if (workingSet.contains(bci)) {
                    const auto &bc = prob->boundConstraint(bci);
                    std::cerr << "Bound constraint on variable " << bc.idx << " reencountered";
                    std::cerr << "step component: " << step[bc.idx] << std::endl;
                    std::cerr << "neg_g_ws_free component: " << (*neg_g_ws_free_ptr)[bc.idx] << std::endl;

                    std::cerr << "throwing logic error (this freezes Knitro!!!)" << std::endl;
                    throw std::logic_error("Re-encountered bound in working set");
                }
                workingSet.add(bci);
                if (options.verboseWorkingSet) std::cout << "Added constraint " << bci << " to working set\n";
            }
        }

        if (!line_search_succeeded) {
            if (options.verbose) std::cout << "Initial backtracking failed; attempting gradient descent.\n";

            if (ngd_fallback_steps-- == 0) {
                if (options.verbose) std::cout << "Maximum number of gradient descent fallback steps reached.\n";
                prob->setVars(vars);
                break;
            }

            directionalDerivative = -g_free_norm * g_free_norm;
            Real initialAlpha = step.norm() / g_free_norm; // Start with the same step length as Newton's method
            bool success = backtrackingLineSearch(*prob, options, vars, *neg_g_ws_free_ptr, currEnergy, directionalDerivative, alpha,
                                                  /* initialAlpha =  */ step.norm() / g_free_norm);
            if (!success) {
                if (options.verbose) std::cout << "Gradient descent backtracking failed.\n";
                prob->setVars(vars);
                break;
            }
        }
    }

    // Report the last iterate; gradient must be re-computed in case the iteration limit was exceeded
    if (it > options.niter) {
        prob->iterationCallback(it);
        neg_g = -prob->gradient(true);
        zeroOutFixedVarsInPlace(neg_g);
    }
    projectOutLEQConstrainedComponents(neg_g);
    prob->customIterateReport(report);
    reportIterate(it - 1, prob->objective(), workingSet.getFreeComponent(neg_g).norm(),
                  /* force report under any nonzero verbosity level */ true);
    std::cout << std::flush;

    if ((options.verboseWorkingSet > 1) && workingSet.size()) {
        std::cout << "Terminated with working set:\n";
        workingSet.report(prob->getVars(), neg_g);
    }

    // std::cout << "Before apply bound constraints: " << prob->objective() << std::endl;
    // prob->setVars(prob->applyBoundConstraints(prob->getVars()));
    // std::cout << "After  apply bound constraints: " << prob->objective() << std::endl;
    // std::cout << "Terminating with report.backtracking_failure = " << report.backtracking_failure << std::endl;

    return report;
}
