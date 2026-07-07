#include "NewtonHessianFactorization.hh"
#include "NewtonProblem.hh"
#include "WorkingSet.hh"

namespace MeshFEM {

NewtonHessianFactorization::NewtonHessianFactorization(std::shared_ptr<NewtonProblem> p,
        const NewtonOptimizerOptions &options)
    : m_options(options), m_problem(p) { }

Real NewtonHessianFactorization::tauScale() const { return (m_options.hessianScaledBeta ? m_cachedHessianL2Norm.get(*m_problem) : 1.0) / m_problem->metricL2Norm(); }

void NewtonHessianFactorization::updateSymbolicFactorization() {
    auto &s = solver();

    g_matrixRecorder.countSparsityUpdateCall();

    m_problem->updateSparsityPattern();

    bool needsUpdate = (m_problem->sparsityPatternID() != m_factorizedSparsityPatternID);

    // If the solver changed out from under us, it won't have a symbolic
    // factorization for the current pattern, and we need to force an update.
    needsUpdate |= !s.hasFactorization(CholeskyFactorizerBase::FactorizationType::Symbolic);

    if (!needsUpdate) {
        // Even if the sparsity pattern ID is the same, the fixed variables might have changed.
        m_setFixedVars(m_problem->fixedVars());
        const bool fixedVarsChanged = m_fixedVarsCouldHaveChanged && !s.fixesSameVarsAsSortedUnique(sparseFixedVars());

        needsUpdate |= fixedVarsChanged;
    }

    if (!needsUpdate && s.wantsSymbolicFactorizationRecompute()) {
        std::cout << "Solver triggered a symbolic factorization recomputation!" << std::endl;
        needsUpdate = true;
    }
    needsUpdate |= s.wantsSymbolicFactorizationRecompute();

    if (needsUpdate) {
        // std::cout << "symbolic factorization update " << s.symbolic_mat_name_suffix << std::endl;
        NewtonHessian Hsp = m_problem->hessianSparsityPattern(/*needsUpdate = */ false);
        m_sparseDenseStructure = Hsp.varStructure().sparseDenseStructure();
        m_setFixedVars(m_problem->fixedVars()); // Note, this must happen after m_sparseDenseStructure has been initialized!

        // std::cout << "Symbolic factorization of sparsity pattern with " << Hsp.H_ss->scalarNNZ() << " nonzeros" << std::endl;
        s.factorizeSymbolic(*(Hsp.H_ss), sparseFixedVars());
        m_factorizedSparsityPatternID = m_problem->sparsityPatternID();
        m_lowRankRank = Hsp.low_rank_rank();
    }

    m_fixedVarsCouldHaveChanged = false; // Suppress further checks until the next optimization run.
}

CholeskyFactorizerBase &NewtonHessianFactorization::solver() {
    if (!m_solver || (m_solver->provider() != m_options.factorizer))
        m_solver = make_cholesky_factorizer(m_options.factorizer);

    return *m_solver;
}

void NewtonHessianFactorization::recordFinalSymbolicMatrix() const {
    NewtonHessian Hsp = m_problem->hessianSparsityPattern(/*needsUpdate = */ false);
    g_matrixRecorder.recordSymbolic(*(Hsp.H_ss), sparseFixedVars());
}

NewtonHessianFactorization::~NewtonHessianFactorization() { }

Real CachedHessianL2Norm::get(const NewtonProblem &p) {
    const auto &H = p.hessian();
    Real tr = H.trace();
    if (std::abs(tr - hessianTrace) > TRACE_TOL * std::abs(hessianTrace)) {
        hessianTrace = tr;
        hessianL2Norm = p.hessianL2Norm();
    }
    return hessianL2Norm;
}

Real NewtonHessianFactorization::update(const WorkingSet &ws, Real &beta, const Real betaMin) {
    updateSymbolicFactorization(); // Update the symbolic factorization if sparsity pattern has changed.

    auto &hUpdtCtr = m_options.getHessianUpdateController();
    auto &hProjCtr = m_options.getHessianProjectionController();

    // Note this `H_hn` reference is bound to the cached Hessian stored in `m_problem`.
    // If `m_updateSparseFactorization` forces a Hessian reevaluation, then
    // `H_hn` will be updated in-place!
    // (This is what we want for the subsequent dense factorization steps.)
    // std::cout << "NewtonHessianFactorization update hProjCtr.shouldUseProjection(): " << hProjCtr.shouldUseProjection() << std::endl;
    hProjCtr.prepareForInitialFactorizationAttempt();
    const NewtonHessian &H_nh = m_problem->hessian(hProjCtr.shouldUseProjection());

    H_nh.validate(); // Make sure everything in H_nh is of the expected size.

    if (H_nh.C_s.size() + H_nh.C_d.size() > 0) {
        throw std::runtime_error("KKT unimplemented");
    }

    Real tau = 0;
    if (H_nh.varStructure().numSparseVars() > 0) {
        tau = m_updateSparseFactorization(H_nh, ws, beta, betaMin);
    }

    // Notify the update controller that we have factorized a new Hessian.
    // (Projection controller has already been notified of the indefiniteness state.)
    bool isIndefinite = tau != 0.0;
    hUpdtCtr.newHessian(isIndefinite);

    if ((H_nh.varStructure().numDenseVars() > 0) || (H_nh.low_rank_rank() > 0)) {
        m_lowRankRank = H_nh.low_rank_rank();
        m_updateDenseFactorization(H_nh);
    }

    return tau;
}

Real NewtonHessianFactorization::m_updateSparseFactorization(const NewtonHessian &H, const WorkingSet &ws, Real &beta, const Real betaMin) {
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
    m_shift = 0.0;

    auto &s = solver();
    s.setSuppressWarnings(!m_options.verboseNonPosDef);

    // Accessor for the sparse block of the Hessian;
    // we dereference each time we access in case the block pointer has been updated
    // by Hessian reevaluation (unlikely since re-evaluations should happen in-place).
    auto getH = [&H]() -> BlockCSCHessianBase & {
        if (H.H_ss) return *H.H_ss;
        throw std::runtime_error("No sparse block present.");
    };

    const SuiteSparseMatrix *M = nullptr;

    // OptionallyModifiedHessian H(m_problem->hessian(hProjCtr.shouldUseProjection())), M;
    // if (ws.size()) {
    //     BENCHMARK_SCOPED_TIMER_SECTION hevalTimer("hessMod");
    //     fixVariablesInWorkingSet(*m_problem, *H.getMutable(), ws);
    // }
    if (ws.size()) throw std::runtime_error("TODO: WorkingSet support has been disabled for refactoring and must be reimplemented.");

    Real currentTauScale = 0; // simple caching mechanism to avoid excessive calls to tauScale()
    size_t numIndefiniteFactorizationsOfCurrentHessian = 0; // Number of times the *same* underlying Hessian has been found to be indefinite; this is reset if the Hessian is reevaluated.
    while (true) {
        try {
            if (tau != 0) {
                if (m_options.useIdentityMetric || !(m_problem->providesMetric())) {
                    s.factorizeNumericWithShift(getH(), tau * currentTauScale);
                    m_shift = tau * currentTauScale;
                }
                else {
                    if (!M) {
                        BENCHMARK_SCOPED_TIMER_SECTION solve("Eval metric");
                        M = &m_problem->metric();
                        // M.set(m_problem->metric());
                        // if (ws.size()) fixVariablesInWorkingSet(*m_problem, *M.getMutable(), ws);
                    }

                    s.factorizeNumericWithShift(getH(), tau * currentTauScale, *M);
                }
            }
            else {
                if (m_problem->hessianShift == 0)
                    s.factorizeNumeric(getH());
                else {
                    Real shift = m_problem->hessianShift;
                    if (m_problem->useRelativeHessianShift)
                        shift *= (getH().trace() / m_problem->numVars());
                    s.factorizeNumericWithShift(getH(), shift);
                    m_shift = shift;
                }
            }

            if (!s.checkPosDef()) throw std::runtime_error("System matrix is not positive definite"); // Needed in case CHOLMOD decides on an LDL factorization...
            break;
        }
        catch (std::exception &e) {
            ++numIndefiniteFactorizationsOfCurrentHessian;
            if (numIndefiniteFactorizationsOfCurrentHessian == 1) {
                // Immediately notify the projection controller of indefiniteness of
                // the unshifted, potentially reevaluated Hessian;
                // if the controller returns `true`, then we need to recompute
                // the Hessian with projection before trying shifts.
                if (m_options.getHessianProjectionController().notifyDefiniteness(/* isIndefinite = */ true)) {
                    // std::cout << "Indefinite Hessian; hessian projection controller requested a reevaluation of the Hessian with projection.\n";
                    m_problem->invalidateCachedHessian();
                    m_problem->hessian(true).validate(); // Updates the Hessian obtained by `getH` in-place.
                    numIndefiniteFactorizationsOfCurrentHessian = 0;
                    continue; // No shifts at this time; we've just enabled projection
                }
            }

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

    // The projection controller has only been notified so far if the Hessian was indefinite;
    // send the positive-definite notification now.
    if (tau == 0.0) m_options.getHessianProjectionController().notifyDefiniteness(/* isIndefinite = */ false);

    return tau;
}

void NewtonHessianFactorization::solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const {
    BorderedSparseFactorization::solve(b, x);

    // Attempt to use Neumann series to correct for the shift applied during factorization...
    const size_t numCorrections = 0;
    if (m_shift > 0 && numCorrections > 0) {
        Eigen::VectorXd x_orig = x;
        Eigen::VectorXd x_tilde;
        for (size_t i = 0; i < numCorrections; ++i) {
            x.swap(x_tilde);
            BorderedSparseFactorization::solve(x_tilde, x);
            x = x_orig + m_shift * x;
        }
    }
}

} // namespace MeshFEM
