#include "NewtonHessian.hh"

Real NewtonHessianFactorization::update(const WorkingSet &ws, Real &beta, const Real betaMin) {
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

    auto &s = solver();
    s.setSuppressWarnings(!m_options.verboseNonPosDef);

    updateSymbolicFactorization(); // Update the symbolic factorization if sparsity pattern has changed.

    auto &hUpdtCtr = m_options.getHessianUpdateController();
    auto &hProjCtr = m_options.getHessianProjectionController();
    const auto &H = m_problem->hessian(hProjCtr.shouldUseProjection());
    const SuiteSparseMatrix *M = nullptr;

    // OptionallyModifiedHessian H(m_problem->hessian(hProjCtr.shouldUseProjection())), M;
    // if (ws.size()) {
    //     BENCHMARK_SCOPED_TIMER_SECTION hevalTimer("hessMod");
    //     fixVariablesInWorkingSet(*m_problem, *H.getMutable(), ws);
    // }
    if (ws.size()) throw std::runtime_error("TODO: WorkingSet support has been disabled for refactoring and must be reimplemented.");

    Real currentTauScale = 0; // simple caching mechanism to avoid excessive calls to tauScale()
    while (true) {
        try {
            if (tau != 0) {
                if (m_options.useIdentityMetric || !(m_problem->providesMetric())) {
                    s.factorizeNumericWithShift(H, tau * currentTauScale);
                }
                else {
                    if (!M) {
                        BENCHMARK_SCOPED_TIMER_SECTION solve("Eval metric");
                        M = &m_problem->metric();
                        // M.set(m_problem->metric());
                        // if (ws.size()) fixVariablesInWorkingSet(*m_problem, *M.getMutable(), ws);
                    }

                    s.factorizeNumericWithShift(H, tau * currentTauScale, *M);
                }
            }
            else {
                if (m_problem->hessianShift == 0)
                    s.factorizeNumeric(H);
                else s.factorizeNumericWithShift(H, m_problem->hessianShift);
            }

            if (!s.checkPosDef()) throw std::runtime_error("System matrix is not positive definite"); // Needed in case CHOLMOD decides on an LDL factorization...
            break;
        }
        catch (std::exception &e) {
            // std::cout << "Caught exception: " << e.what() << std::endl;
            tau  = std::max(4.0 * tau, beta);
            beta = std::max(0.5 * tau, betaMin);
            if (m_options.verboseNonPosDef) std::cout << e.what() << "; increasing tau to " << tau << "\n";
            if (currentTauScale == 0) currentTauScale = tauScale();
            if (tau > 1e80) {
                // m_problem->writeDebugFiles("tau_runaway");
                std::cout << "Tau running away\n";
                std::cout << "||H||_2: "    << m_problem->hessianL2Norm() << std::endl;
                std::cout << "||M||_2: "    << m_problem->metricL2Norm()  << std::endl;
                std::cout << "Scaled tau: " << tau * currentTauScale << std::endl;
                throw std::runtime_error("Tau running away");
            }
        }
    }

    if (m_problem->hasLEQConstraint()) {
        // Eigen::VectorXd a = ws.getFreeComponent(m_problem->LEQConstraintMatrix());
        // kkt_solver.update(s, a);
        throw std::runtime_error("Unimplemented (LEQ constraints are disabled during refactoring)");
    }

    // Notify controllers that we have factorized a new Hessian
    // and whether or not it was indefinite.
    bool isIndefinite = tau != 0.0;
    hProjCtr.notifyDefiniteness(isIndefinite);
    hUpdtCtr.newHessian(isIndefinite);

    return tau;
}
