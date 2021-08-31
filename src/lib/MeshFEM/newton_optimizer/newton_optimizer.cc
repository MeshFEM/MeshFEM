#include "newton_optimizer.hh"
#include "../AutomaticDifferentiation.hh"

// Modify `H` to enforce the active bound constraints (which are of the form d_i = 0 when solving H d = -g).
// In order to preserve H's sparsity pattern, instead of removing the rows/columns for pinned variables `i`,
// we replace these rows/columns with rows/columns of the identity.
void fixVariablesInWorkingSet(const NewtonProblem &prob, SuiteSparseMatrix &H, const WorkingSet &ws) {
    if (ws.size() == 0) return;

    BENCHMARK_START_TIMER("fixVariablesInWorkingSet");
    // Zero out the rows corresponding to all variables in the working set
    for (size_t elem = 0; elem < H.Ai.size(); ++elem)
        if (ws.fixesVariable(H.Ai[elem])) H.Ax[elem] = 0.0;

    // Zero out working set vars' columns/gradient components, placing a 1 on the diagonal
    const SuiteSparseMatrix::index_type nv = prob.numVars();
    for (SuiteSparseMatrix::index_type var = 0; var < nv; ++var) {
        if (!ws.fixesVariable(var)) continue;
        const auto start = H.Ap[var    ],
                   end   = H.Ap[var + 1];
        Eigen::Map<Eigen::VectorXd>(H.Ax.data() + start, end - start).setZero();
        assert(H.Ai[end - 1] == var);
        H.Ax[end - 1] = 1.0; // Diagonal should be the column's last entry; we assume it exists in the sparsity pattern!
    }

    BENCHMARK_STOP_TIMER("fixVariablesInWorkingSet");
}

// Solve the Newton system `H d = -g`, modifying H to be pos. def. if it is indefinite.
// Returns "tau", the coefficient of the metric term that was added to make the Hessian positive definite.
// "-tau" can be interpreted as an estimate (lower bound) for the smallest generalized eigenvalue for "H d = lambda M d"
// (Returns 0 if the Hessian is already positive definite).
// Upon return, "solver" holds a factorization of the matrix:
//     (H + tau (M / ||M||_2))
Real NewtonOptimizer::newton_step(Eigen::VectorXd &step, const Eigen::VectorXd &g, const WorkingSet &ws, Real &beta, const Real betaMin, const bool feasibility) {
    BENCHMARK_SCOPED_TIMER_SECTION ns_timer("newton_step");
    step.resize(g.size());

    // The following Hessian modification strategy is an improved version of
    // "Cholesky with added multiple of the identity" from
    // Nocedal and Wright 2006, pp 51.
    // We use a custom matrix instead of the identity, drawing an analogy
    // to trust region methods: the multiplier (scaledTau) that we use
    // corresponds to some trust region radius in the metric defined by the
    // added matrix, and some metrics can work much better than the
    // Euclidean distance in the parameter space. For instance,
    // the mass matrix is a good choice.
    Real tau = 0;

    // Though the full mass matrix is cached by NewtonProblem, we also want to cache
    // the reduced version (if it is ever needed).
    std::unique_ptr<SuiteSparseMatrix> M_reduced;

    Eigen::VectorXd x, gReduced;

    auto postprocessSolution = [&]() {
        extractFullSolution(x, step);
        step *= -1;
        // ws.validateStep(step);

        if (prob->hasLEQConstraint()) {
            // TODO: handle more than a single constraint...
            Eigen::VectorXd a = removeFixedEntries(ws.getFreeComponent(prob->LEQConstraintMatrix()));
            kkt_solver.update(solver, a);
            const Real r = feasibility ? prob->LEQConstraintResidual() : 0.0;
            extractFullSolution(kkt_solver.solve(-x, r), step);
        }
    };

    auto &hUpdtCtr = options.getHessianUpdateController();
    auto &hProjCtr = options.getHessianProjectionController();

    Eigen::VectorXd g_free = ws.getFreeComponent(g); // Zero out the entries with active bound constraints.

    if (solver.hasFactorization()) {
        if (!hUpdtCtr.needsUpdate() && (ws.size() == 0)) { // TODO: Reusing factorizations with bound constraints needs more care
            hUpdtCtr.reusedHessian();
            gReduced = removeFixedEntries(g_free);
            solver.solveExistingFactorization(gReduced, x);
            postprocessSolution();
            return NAN; // tau is unknown/undefined since we're reusing an old factorization; no negative curvature direction will be attempted by caller.
        }
    }

    SuiteSparseMatrix H_reduced;
    { BENCHMARK_SCOPED_TIMER_SECTION hevalTimer("hessEval");
        H_reduced = prob->hessian(hProjCtr.shouldUseProjection());
        fixVariablesInWorkingSet(*prob, H_reduced, ws);
        H_reduced.rowColRemoval([&](SuiteSparse_long i) { return isFixed[i]; });
    }

    Real currentTauScale = 0; // simple caching mechanism to avoid excessive calls to tauScale()
    while (true) {
        try {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Newton solve");
            if (tau != 0) {
                if (!M_reduced) {
                    M_reduced = std::make_unique<SuiteSparseMatrix>(prob->metric());
                    fixVariablesInWorkingSet(*prob, *M_reduced, ws);
                    M_reduced->rowColRemoval([&](SuiteSparse_long i) { return isFixed[i]; });
                }

                auto Hmod = H_reduced;
                Hmod.addWithIdenticalSparsity(*M_reduced, tau * currentTauScale); // Note: rows/cols corresponding to vars with active bounds will now have a nonzero value different from 1 on the diagonal, but this is fine since the RHS component is zero...
                solver.updateFactorization(std::move(Hmod));
            }
            else {
                solver.updateFactorization(H_reduced);
            }

            BENCHMARK_SCOPED_TIMER_SECTION solve("Solve");

            gReduced = removeFixedEntries(g_free);
            solver.solve(gReduced, x);
            if (!solver.checkPosDef()) throw std::runtime_error("System matrix is not positive definite");
            postprocessSolution();

            break;
        }
        catch (std::exception &e) {
            tau  = std::max(  4 * tau, beta);
            beta = std::max(0.5 * tau, betaMin);
            if (options.verboseNonPosDef) std::cout << e.what() << "; increasing tau to " << tau << "\n";
            if (currentTauScale == 0) currentTauScale = tauScale();
            if (tau > 1e80) {
                // prob->writeDebugFiles("tau_runaway");
                std::cout << "Tau running away\n";
                std::cout << "||H||_2: "    << prob->hessianL2Norm() << std::endl;
                std::cout << "||M||_2: "    << prob->metricL2Norm()  << std::endl;
                std::cout << "Scaled tau: " << tau * currentTauScale << std::endl;
                throw std::runtime_error("Tau running away");
            }
        }
    }

    // Notify controllers that we have factorized a new Hessian
    // and whether or not it was indefinite.
    bool isIndefinite = tau != 0.0;
    hProjCtr.notifyDefiniteness(isIndefinite);
    hUpdtCtr.newHessian(isIndefinite);

    return tau;
}

ConvergenceReport NewtonOptimizer::optimize() {
    size_t ngd_fallback_steps = options.ngd_fallback_steps; // maximum number of gradient descent steps to take as a fallback when backtracking for the newton step fails.

    prob->setUseIdentityMetric(options.useIdentityMetric);
    prob->writeIterates = options.writeIterateFiles;

    prob->setVars(prob->applyBoundConstraints(prob->getVars()));
    Eigen::VectorXd vars, step;

    // Indices of the bound constraints in our working set.
    WorkingSet workingSet(*prob);

    Real beta = options.beta;
    const Real betaMin = std::min(beta, 1e-10); // Initial shift "tau" to use when an indefinite matrix is detected.

    solver.setSuppressWarnings(!options.verboseNonPosDef);

    m_cachedHessianL2Norm.reset();

    if (prob->hasLEQConstraint()) {
        if (!prob->LEQConstraintIsFeasible()) {
            if (options.feasibilitySolve) {
                // std::cout << "Running feasibility solve with residual " << prob->LEQConstraintResidual() << ", energy " << prob->energy() << std::endl;
                prob->iterationCallback(0);
                newton_step(step, prob->gradient(true), workingSet, beta, betaMin, true);
                // We must take a full step to ensure feasibility
                // TODO: use multiple iterations and a line search to get feasible?
                prob->setVars(prob->applyBoundConstraints(step + prob->getVars()));
                // std::cout << "Post feasibility solve residual " << prob->LEQConstraintResidual() << ", energy " << prob->energy() << std::endl;
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
    auto zeroOutFixedVars = [&](const Eigen::VectorXd &g) { auto result = g; for (size_t var : fixedVars) result[var] = 0.0; return result; };

    ConvergenceReport report;

    Real alpha = 0;
    bool isIndefinite = false;
    auto reportIterate = [&](size_t i, Real energy, const Eigen::VectorXd &g, const Eigen::VectorXd &g_free) {
        prob->writeIterateFiles(i);
        report.addEntry(energy, g.norm(), g_free.norm(), alpha, isIndefinite);

        if (options.verbose && ((i % options.verbose) == 0)) {
            std::cout << i << '\t';
            report.printEntry();
            if (i % options.stdoutFlushInterval == 0)
                std::cout << std::flush;
        }
    };

    BENCHMARK_START_TIMER_SECTION("Newton iterations");
    size_t it;
    Eigen::VectorXd g(prob->numVars());

    Eigen::VectorXd za, zg, g_free;
    if (prob->hasLEQConstraint()) { za = zeroOutFixedVars(prob->LEQConstraintMatrix()); }
    // Kill off components of "v" in the span of the LEQ constraint vectors
    auto projectOutLEQConstrainedComponents = [&](Eigen::VectorXd &v) { if (prob->hasLEQConstraint()) v -= za * (v.dot(za) / za.squaredNorm()); };

    options.getHessianProjectionController().reset();
    options.getHessianUpdateController()    .reset();

    for (it = 1; it <= options.niter; ++it) {
        BENCHMARK_SCOPED_TIMER_SECTION it_timer("Newton iterate");

        Real currEnergy;
        { BENCHMARK_SCOPED_TIMER_SECTION t2("Preamble");

        // std::cout << "pre-update gradient: " << zeroOutFixedVars(prob->gradient(false)).norm() << std::endl;
        {
            BENCHMARK_SCOPED_TIMER_SECTION cbTimer("Callback");
            prob->iterationCallback(it);
        }
        // Note: we allow the iteration callback to modify the variables!
        // (in case the user wants to run some custom projection/filter at the start
        //  of each Newton iteration).
        vars = prob->getVars();

        g = prob->gradient(true);
        currEnergy = prob->energy();

        zg = zeroOutFixedVars(g); // non-fixed components of the gradient; used for termination criteria
        projectOutLEQConstrainedComponents(zg);
        // Gradient with respect to the "free" variables (components corresponding to fixed/actively constrained variables zero-ed out)
        g_free = workingSet.getFreeComponent(zg);

        if ((!isIndefinite) && (zg.norm() < options.gradTol)) {
            report.success = true;
            break; // TODO: termination criterion when bounds are active at the optimum
        }

        // Free variables in the working set from their bound constraints, if necessary
        bool ws_updated = workingSet.remove_if([&](size_t bc_idx) {
                bool shouldRemove = prob->boundConstraint(bc_idx).shouldRemoveFromWorkingSet(g, g_free);
                if (shouldRemove) { std::cout << "Removed constraint " << bc_idx << " from working set" << std::endl; }
                return shouldRemove;
            });

        if (ws_updated) g_free = workingSet.getFreeComponent(zg);

        } // End of 'Preamble' timer


        { BENCHMARK_SCOPED_TIMER_SECTION t2("Compute descent direction");

        Real old_beta = beta;
        Real tau;
        try {
            tau = newton_step(step, g_free, workingSet, beta, betaMin);
        }
        catch (std::exception &e) {
            // Tau ran away
            break;
        }
        isIndefinite = (tau != 0.0);

        // Only add in negative curvature directions when "tau" is a reasonable estimate for the smallest eigenvalue and the gradient has become small.
        if (options.useNegativeCurvatureDirection && ((tau > old_beta) || (tau == betaMin)) && (g_free.norm() < 100 * options.gradTol)) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Negative curvature dir");
            // std::cout.precision(19);
            std::cout << "Computing negative curvature direction for scaled tau = " << tau / prob->metricL2Norm() << '\n';
            auto M_reduced = prob->metric();
            fixVariablesInWorkingSet(*prob, M_reduced, workingSet);
            M_reduced.rowColRemoval([&](SuiteSparse_long i) { return isFixed[i]; });
            auto d = negativeCurvatureDirection(solver, M_reduced, 1e-6);
            {
                Real dnorm = d.norm();
                if (dnorm != 0.0) {
                    Eigen::VectorXd tmp(step.size());
                    extractFullSolution(d, tmp); // negative curvature direction was computed in reduced variables...
                    d = tmp;
                    // {
                    //     const SuiteSparseMatrix &H = prob->hessian();
                    //     H.applyRaw(d.data(), tmp.data());
                    //     Real lambda = d.dot(tmp);
                    //     std::cout << "Found negative curvature direction with eigenvalue " << lambda << std::endl;
                    // }
                    if (d.dot(g) > 0) d *= -1; // Move in the opposite direction as the gradient (So we still produce a descent direction)
                    const Real cd = prob->characteristicDistance(d);
                    if (cd <= 0) // problem doesn't provide one
                        step += std::sqrt(step.squaredNorm() / d.squaredNorm()) * d; // TODO: find a better balance between newton step and negative curvature.
                    else {
                        step += 1e-2 * (d / cd);
                    }
                }
                else { std::cout << "Negative curvature direction calculation failed" << std::endl; }
            }
        }

        } // End of 'Compute descent direction' timer

        Real directionalDerivative = g_free.dot(step);
        // if (options.verbose)
        //     std::cout << "Found step with directional derivative: " << directionalDerivative << std::endl;

        BENCHMARK_START_TIMER_SECTION("Backtracking");
        // Simple backtracking line search to ensure a sufficient decrease

        Real feasible_alpha;
        size_t blocking_idx;
        std::tie(feasible_alpha, blocking_idx) = prob->feasibleStepLength(vars, step);

        // To add multiple nearby bounds to the working set at once, we allow the
        // step to overshoot the bounds slightly (note: variables will be clamped to the bounds anyway before
        // evaluating the objective). Then any bound violated by the step length obtaining
        // sufficient decrease is added to the working set.
        alpha = std::min(1.0, feasible_alpha + 1e-3);

        const Real c_1 = 1e-2;
        size_t bit;

        Eigen::VectorXd steppedVars;
        for (bit = 0; bit < options.nbacktrack_iter; ++bit) {
            steppedVars = vars + alpha * step;
            prob->applyBoundConstraintsInPlace(steppedVars);
            prob->setVars(steppedVars);
            const Real steppedEnergy = prob->energy();
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
                alpha *= 0.5;
            }
        }
        BENCHMARK_STOP_TIMER_SECTION("Backtracking");

        reportIterate(it - 1, currEnergy, zg, g_free); // Record iterate statistics, now that we know alpha, isIndefinite
        prob->customIterateReport(report);

        // Add to the working set all bounds encountered by the step of length "alpha"
        for (size_t bci = 0; bci < prob->numBoundConstraints(); ++bci) {
            if (alpha >= prob->boundConstraint(bci).feasibleStepLength(vars, step)) {
                if (workingSet.contains(bci)) {
                    const auto &bc = prob->boundConstraint(bci);
                    std::cerr << "Bound constraint on variable " << bc.idx << " reencountered";
                    std::cerr << "step component: " << step[bc.idx] << std::endl;
                    std::cerr << "g_free component: " << g_free[bc.idx] << std::endl;

                    std::cerr << "throwing logic error (this freezes Knitro!!!)" << std::endl;
                    throw std::logic_error("Re-encountered bound in working set");
                }
                workingSet.add(bci);
                std::cout << "Added constraint " << bci << " to working set\n";
            }
        }

        if (bit == options.nbacktrack_iter) {
            if (options.verbose) std::cout << "Initial backtracking failed; attempting gradient descent.\n";

            if (ngd_fallback_steps-- == 0) {
                if (options.verbose) std::cout << "Maximum number of gradient descent fallback steps reached.\n";
                prob->setVars(vars);
                break;
            }

            size_t gd_bit;
            directionalDerivative = -g_free.squaredNorm();
            alpha *= step.norm() / g_free.norm(); // Start with the same step magnitude where the Newton step backtracking failed....
            step = -g_free;
            for (gd_bit = 0; gd_bit < options.nbacktrack_iter; ++gd_bit) {
                steppedVars = vars + alpha * step;
                prob->applyBoundConstraintsInPlace(steppedVars);
                prob->setVars(steppedVars);
                Real steppedEnergy = prob->energy();

                if  (steppedEnergy - currEnergy <= c_1 * alpha * directionalDerivative)
                    break;
                alpha *= 0.5;
            }
        }
    }

    // Report the last iterate; gradient must be re-computed in case the iteration limit was exceeded
    if (it > options.niter) {
        prob->iterationCallback(it);
        g = prob->gradient(true);
    }
    zg = zeroOutFixedVars(g);
    projectOutLEQConstrainedComponents(zg);
    prob->customIterateReport(report);
    reportIterate(it - 1, prob->energy(), zg, workingSet.getFreeComponent(zg));
    std::cout << std::flush;

    if (workingSet.size()) {
        std::cout << "Terminated with working set:\n";
        vars = prob->getVars();
        for (size_t bci = 0; bci < prob->numBoundConstraints(); ++bci) {
            if (workingSet.contains(bci)) prob->boundConstraint(bci).report(vars, g);
        }
    }

    // std::cout << "Before apply bound constraints: " << prob->energy() << std::endl;
    // prob->setVars(prob->applyBoundConstraints(prob->getVars()));
    // std::cout << "After  apply bound constraints: " << prob->energy() << std::endl;
    // std::cout << "Terminating with report.backtracking_failure = " << report.backtracking_failure << std::endl;

    BENCHMARK_STOP_TIMER_SECTION("Newton iterations");

    return report;
}
