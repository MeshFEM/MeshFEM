#include "NewtonHessian.hh"
#include "NewtonProblem.hh"
#include "WorkingSet.hh"

NewtonHessianFactorization::NewtonHessianFactorization(std::shared_ptr<NewtonProblem> p,
        const NewtonOptimizerOptions &options)
    : m_options(options), m_problem(p) { }

Real NewtonHessianFactorization::tauScale() const { return (m_options.hessianScaledBeta ? m_cachedHessianL2Norm.get(*m_problem) : 1.0) / m_problem->metricL2Norm(); }

void NewtonHessianFactorization::updateSymbolicFactorization() {
    if (!m_solver) return; // Solver hasn't been created yet; nothing to update.

    m_problem->updateSparsityPattern();
    const bool fixedVarsChanged = m_fixedVarsCouldHaveChanged && !m_solver->fixesSameVarsAsSortedUnique(m_problem->fixedVars());
    if (fixedVarsChanged || (m_problem->sparsityPatternID() != m_factorizedSparsityPatternID)) {
        m_solver->factorizeSymbolic(*(m_problem->hessianSparsityPattern().H_ss), m_problem->fixedVars());
        m_factorizedSparsityPatternID = m_problem->sparsityPatternID();
    }
    m_fixedVarsCouldHaveChanged = false; // Suppress further checks until the next optimization run.
}

CholeskyFactorizerBase &NewtonHessianFactorization::solver() {
    if (!m_solver || (m_solver->provider() != m_options.factorizer)) {
        m_solver = make_cholesky_factorizer(m_options.factorizer);
        m_solver->recordMatrices(m_options.matrixRecordDir); // enable recording if the user specified a directory
        updateSymbolicFactorization();
    }

    return *m_solver;
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

    if ((H_nh.varStructure().numDenseVars()) > 0 || H_nh.low_rank_rank() > 0) {
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
    size_t numIndefiniteFactorizations = 0;
    bool hessianReevaluated = false; // whether the projection controller forced a reevaluation of the Hessian with projection within this step
    while (true) {
        try {
            if (tau != 0) {
                if (m_options.useIdentityMetric || !(m_problem->providesMetric())) {
                    s.factorizeNumericWithShift(getH(), tau * currentTauScale);
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
                else s.factorizeNumericWithShift(getH(), m_problem->hessianShift);
            }

            if (!s.checkPosDef()) throw std::runtime_error("System matrix is not positive definite"); // Needed in case CHOLMOD decides on an LDL factorization...
            break;
        }
        catch (std::exception &e) {
            ++numIndefiniteFactorizations;
            if (numIndefiniteFactorizations == 1) { // First time we've encountered indefiniteness
                // We immediately notify the projection controller of
                // indefiniteness of the unshifted Hessian; if the controller
                // returns `true`, then we need to recompute the Hessian with
                // projection before trying shifts.
                if (m_options.getHessianProjectionController().notifyDefiniteness(/* isIndefinite = */ true)) {
                    m_problem->invalidateCachedHessian();
                    m_problem->hessian(true).validate(); // Updates the Hessian obtained by `getH` in-place.
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

    // If the Hessian was reevaluated with projection and was found to *still*
    // be indefinite, we notify the controller of this fact.
    // However we don't want to send a second notification for the original
    // Hessian if it was not reevaluated.
    if (hessianReevaluated && (tau > 0.0)) {
        // Notify the projection controller of the definiteness of the Hessian used to compute this step.
        m_options.getHessianProjectionController().notifyDefiniteness(/* isIndefinite = */ true);
    }

    // The projection controller has only been notified so far if the Hessian was indefinite;
    // send the positive-definite notification now.
    if (tau == 0.0) m_options.getHessianProjectionController().notifyDefiniteness(/* isIndefinite = */ false);

    return tau;
}


// Compute the "dense part" of the factorization of:
//  [H_ss B]
//  [B^T  D]
// where B = [H_sd V_s]
// and   D = [H_dd     V_d]
//           [V_d^T   -I_r]
// The Cholesky sparse factorization of (a potentially modified) `H_ss` has
// already been computed in `solver`.
bool NewtonHessianFactorization::m_updateDenseFactorization(const NewtonHessian &H) {
    H.validate();

    size_t nsv = H.varStructure().numSparseVars();
    size_t ndv = H.varStructure().numDenseVars();
    size_t r   = H.low_rank_rank();
    size_t numDenseCols = nsv + r;
    assert(numDenseCols > 0 && "m_updateDenseFactorization should not have been called!");

    Eigen::MatrixXd D(numDenseCols, numDenseCols);
    B.resize(nsv, ndv + r);

    B << H.H_sd, H.V_s;
    D << H.H_dd, H.V_d,
         H.V_d.transpose(), -Eigen::MatrixXd::Identity(r, r);

    solver().solveMultiRHS(B, H_ss_inv_B);

    S = D - B.transpose() * H_ss_inv_B;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(S);
    S_Q = eigs.eigenvectors();
    // TODO: enable customization of the eigenvalue modification strategy!
    S_lambda = eigs.eigenvalues().cwiseAbs().cwiseMax(1e-10);

    return (eigs.eigenvalues().array() < 0).any();
}

void NewtonHessian::validate() const {
    const size_t nsv = varStructure().numSparseVars();
    const size_t ndv = varStructure().numDenseVars();

    if (nsv > 0) {
        if (!H_ss) throw std::runtime_error("H_ss is null");

        if (nsv != H_ss->numScalarCols()) throw std::runtime_error("H_ss has the wrong number of columns");
        if (nsv != H_ss->numScalarRows()) throw std::runtime_error("H_ss is not square");
    }

    if ((ndv > 0) && (nsv > 0)) {
        if (nsv != size_t(H_sd.rows())) throw std::runtime_error("H_sd has the wrong number of rows");
        if (ndv != size_t(H_sd.cols())) throw std::runtime_error("H_sd has the wrong number of columns");
        if ((ndv != size_t(H_dd.rows())) || (ndv != size_t(H_dd.cols()))) throw std::runtime_error("H_dd is the wrong shape");
    }

    if ((V_s.size() != 0) || (V_d.size() != 0)) {
        if (nsv != size_t(V_s.rows())) throw std::runtime_error("V_s has the wrong number of columns");
        if (ndv != size_t(V_d.rows())) throw std::runtime_error("V_d has the wrong number of columns");
    }

    if ((C_s.size() != 0) || (C_d.size() != 0)) {
        if (nsv != size_t(C_s.rows())) throw std::runtime_error("C_s has the wrong number of columns");
        if (ndv != size_t(C_d.rows())) throw std::runtime_error("C_d has the wrong number of columns");
    }
}

void NewtonHessian::addNZ(size_t i, size_t j, const Real val) {
    assert(i <= j); // Only support `UPPER_TRIANGLE` symmetry mode...
    const auto &vs = varStructure();
    bool isSparse_i = vs.isSparseVar(i), // Note that sparse variables could
         isSparse_j = vs.isSparseVar(j); // be collected at the beginning or
                                         // end of the variable list!
    if (isSparse_i && isSparse_j) H_ss->addNZScalar(i - vs.sparseVarOffset(), j - vs.sparseVarOffset(), val);
    else if (isSparse_i) H_sd(i - vs.sparseVarOffset(), j - vs.denseVarOffset()) += val;
    else if (isSparse_j) H_sd(j - vs.sparseVarOffset(), i - vs.denseVarOffset()) += val;
    else H_dd(i - vs.denseVarOffset(), j - vs.denseVarOffset()) += val;
}

void NewtonHessian::applyRaw(const double *x_ptr, double *result_ptr) const {
    const auto &vs = varStructure();
    Eigen::Map<const Eigen::VectorXd> x(x_ptr, vs.numVars());
    Eigen::Map<Eigen::VectorXd> result(result_ptr, vs.numVars());

    validate();

    H_ss->applyRaw(vs.sparseVars(x).data(), vs.sparseVars(result).data());

    // Padding terms
    if (vs.numDenseVars() > 0) {
        if (vs.numSparseVars() > 0) {
            vs.sparseVars(result) += H_sd * vs.denseVars(x);
            vs. denseVars(result)  = H_sd.transpose() * vs.sparseVars(x) + H_dd * vs.denseVars(x); // initializes!
        }
        else {
            vs. denseVars(result) = H_dd * vs.denseVars(x);
        }
    }

    // Low-rank term V V^T x
    if (low_rank_rank() > 0) {
        const auto Vt_x = V_s.transpose() * vs.sparseVars(x) + V_d.transpose() * vs.denseVars(x);
        vs.sparseVars(result) += V_s * Vt_x;
        vs. denseVars(result) += V_d * Vt_x;
    }
}

// Solve the system:
//    [H_ss B][x] = [b_s]
//    [B^T  D][y] = [b_d; 0]
// Using the Schur complement formulas:
//    S := D - B^T H_ss^{-1} B
//    y = S^{-1} (c - B^T H_ss^{-1} b)
//    x = H_ss^{-1} (b - B y),
void NewtonHessianFactorization::solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const {
    if (!exists()) throw std::runtime_error("Factorization doesn't exist.");
    size_t nsv = 1;
    size_t ndv = 0; // TODO: get these...

    if (ndv > 0) {
        // TODO
    }

    if (nsv > 0) {
        solver().solve(b, x);
    }

}
