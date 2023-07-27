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
#include <cmath>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Eigensolver.hh>
#include "ConvergenceReport.hh"
#include "HessianProjectionController.hh"
#include "HessianUpdateController.hh"

#include <MeshFEM_export.h>

struct MESHFEM_EXPORT NewtonProblem {
    using VXd = Eigen::VectorXd;
    virtual void setVars(const VXd &vars) = 0;
    virtual const VXd getVars() const = 0;
    virtual size_t numVars() const = 0;

    // Called at the start of each new iteration (after line search has been performed)
    void iterationCallback(size_t i) {
        m_clearCache();
        m_iterationCallback(i);
    }

    virtual Real energy() const = 0;
    // freshIterate: whether the gradient is being called immediately
    // after an iteration callback (without any change to the variables in between) instead
    // of, e.g., during the line search.
    // For some problems, a less expensive gradient expression can be used in this case.
    virtual VXd gradient(bool freshIterate = false) const = 0;

    const SuiteSparseMatrix &hessian(bool projectionMask = true) const {
        if (!m_cachedHessian) { m_cachedHessian = std::make_unique<SuiteSparseMatrix>(hessianSparsityPattern()); }
        if (disableCaching || !m_cachedHessianUpToDate) {
            m_evalHessian(*m_cachedHessian, projectionMask);
            m_cachedHessianUpToDate = true;
        }
        return *m_cachedHessian;
    }

    // Positive definite matrix defining the metric used to define trust regions.
    // For efficiency, it must have the same sparsity pattern as the Hessian.
    // (This matrix is added to indefinite Hessians to produce a positive definite modified Hessian.)
    const SuiteSparseMatrix &metric() const {
        if (!m_cachedMetric) { m_cachedMetric = std::make_unique<SuiteSparseMatrix>(hessianSparsityPattern()); }
        if (disableCaching || !m_cachedMetricUpToDate) {
            m_cachedMetric = std::make_unique<SuiteSparseMatrix>(hessianSparsityPattern());
            m_evalMetric(*m_cachedMetric);
            m_cachedMetricUpToDate = true;
        }
        if (m_useIdentityMetric)
            m_cachedMetric->setIdentity(/*preserveSparsity*/ true);
        return *m_cachedMetric;
    }

    Real hessianL2Norm() const { return largestMagnitudeEigenvalue(hessian(), 1e-2); }

    // Since computing the L2 norm is slightly expensive, we assume that it remains
    // constant throughout the solve. This is exactly true for ElasticRods, and should be
    // a good approximation for RodLinkages under mild deformation.
    // Also, an exact result should not be necessary since it's only used to determine a reasonable
    // initial guess for the Hessian modification magnitude.
    Real metricL2Norm() const {
        if (m_useIdentityMetric) return 1.0;
        if (m_metricL2Norm <= 0) m_metricL2Norm = largestMagnitudeEigenvalue(metric(), 1e-2);
        return m_metricL2Norm;
    }
    void setUseIdentityMetric(bool useIdentityMetric) { m_useIdentityMetric = useIdentityMetric; }

    // A compressed column sparse matrix with nonzero placeholders wherever the Hessian can ever have nonzero entries.
    virtual SuiteSparseMatrix hessianSparsityPattern() const = 0;

    // sparsity pattern with fixed variable rows/cols removed.
    virtual SuiteSparseMatrix hessianReducedSparsityPattern() const {
        auto hsp = hessianSparsityPattern();
        hsp.fill(1.0);
        std::vector<char> isFixed(numVars(), false);
        for (size_t fv : m_fixedVars) isFixed.at(fv) = true;
        hsp.rowColRemoval([&isFixed] (size_t i) { return isFixed[i]; });
        return hsp;
    }
    bool sparsityPatternFactorizationUpToDate() const { return m_sparsityPatternFactorizationUpToDate; }
    void sparsityPatternFactorizationUpToDate(bool val) { m_sparsityPatternFactorizationUpToDate = val; }

    const std::vector<size_t> &fixedVars() const { return m_fixedVars; }
    size_t numFixedVars() const { return fixedVars().size(); }
    size_t numReducedVars() const { return numVars() - fixedVars().size(); } // number of remaining variables after fixing fixedVars

    // WARNING: updating the fixed variables *after* constructing a
    // NewtonOptimizer from this problem won't work; then you must call
    // NewtonOptimizer::setFixedVars.
    void setFixedVars(const std::vector<size_t> &fv) { m_fixedVars = fv; }
    void addFixedVariables(const std::vector<size_t> &fv) { m_fixedVars.insert(std::end(m_fixedVars), std::begin(fv), std::end(fv)); }

    virtual bool         hasLEQConstraint()       const { return false; }
    virtual Eigen::VectorXd LEQConstraintMatrix() const { return Eigen::VectorXd(); }
    virtual Real            LEQConstraintRHS()    const { return 0.0; }
    virtual void         setLEQConstraintRHS(Real)      { throw std::runtime_error("Problem doesn't apply a LEQ constraint."); }
    virtual Real            LEQConstraintTol()    const { return 1e-7; }
    virtual void            LEQStepFeasible()           { throw std::runtime_error("Problem type doesn't implement direct feasibility step."); }
    // r = b - Ax
    Real LEQConstraintResidual() const { return LEQConstraintRHS() - LEQConstraintMatrix().dot(getVars()); }
    bool LEQConstraintIsFeasible() const { return std::abs(LEQConstraintResidual()) <= LEQConstraintTol(); }

    bool writeIterates = false;
    virtual void writeIterateFiles(size_t /* it */) const { };
    virtual void writeDebugFiles(const std::string &/* errorName */) const { };

    NewtonProblem &operator=(const NewtonProblem &b) = delete;

    struct BoundConstraint {
        enum Type { LOWER, UPPER};
        size_t idx;
        Real val;
        Type type;

        BoundConstraint(size_t i, Real v, Type t) : idx(i), val(v), type(t) { }

        // To avoid numerical issues as iterates approach the bound constraints, a constraint
        // is considered active if the variable is within "tol" of the bound.
        bool active(const VXd &vars, const VXd &g, Real tol = 1e-8) const {
            return ((type == Type::LOWER) && (vars[idx] <= val + tol) && ((g.size() == 0) || (g[idx] >= 0)))
                || ((type == Type::UPPER) && (vars[idx] >= val - tol) && ((g.size() == 0) || (g[idx] <= 0)));
        }

        // Decide whether the bound constraint should be removed from the working set.
        // For the Lagrange multiplier estimate to be accurate, the reduced gradient must be small.
        // (Since we're working with bound constraints, the first-order Lagrange multiplier estimate is simply the gradient component)
        bool shouldRemoveFromWorkingSet(const VXd &g, Real g_free_norm) const {
            if (type == Type::UPPER) { return g[idx] >  10 * g_free_norm; }
            if (type == Type::LOWER) { return g[idx] < -10 * g_free_norm; }
            throw std::runtime_error("Unknown bound type");
        }

        bool feasible(const VXd &vars) const {
            if (type == Type::LOWER) return vars[idx] >= val;
            else                     return vars[idx] <= val;
            throw std::runtime_error("Unknown bound type");
        }
        void apply(VXd &vars) const {
            if ((type == Type::LOWER) && (vars[idx] < val)) vars[idx] = val;
            if ((type == Type::UPPER) && (vars[idx] > val)) vars[idx] = val;
        }
        Real feasibleStepLength(const VXd &vars, const VXd &step) const {
            Real alpha = std::numeric_limits<Real>::max();
            if      (type == Type::LOWER) { if (step[idx] < 0) alpha = (val - vars[idx]) / step[idx]; }
            else if (type == Type::UPPER) { if (step[idx] > 0) alpha = (val - vars[idx]) / step[idx]; }
            else throw std::runtime_error("Unknown bound type");
            // Note: alpha will be negative if "vars" are already infeasible and step is nonzero.
            // This should never happen assuming active constraints are detected/handled properly.
            if (alpha < 0) throw std::runtime_error("Feasible step is negative");
            return alpha;
        }

        void report(const VXd &vars, const VXd &g) const {
            std::cout << "\t" << ((type == Type::LOWER) ? "lower" : "upper") << " bd on var " << idx
                      << " (curr val:" << vars[idx] << ", bd: " << val << ", lagrange multiplier: " << g[idx] << ")" << std::endl;
        }
    };

    const std::vector<BoundConstraint> &boundConstraints() const { return m_boundConstraints; }
    size_t                           numBoundConstraints() const { return m_boundConstraints.size(); }
    const BoundConstraint &boundConstraint(size_t i) const { return m_boundConstraints[i]; }

    VXd applyBoundConstraints(VXd vars) const {
        for (auto &bc : m_boundConstraints) bc.apply(vars);
        return vars;
    }

    void applyBoundConstraintsInPlace(VXd &vars) const {
        for (auto &bc : m_boundConstraints) bc.apply(vars);
    }

    std::vector<BoundConstraint> activeBoundConstraints(const VXd &vars, const VXd &g = VXd(), Real tol = 1e-8) const {
        std::vector<BoundConstraint> result;
        for (auto &bc : m_boundConstraints) {
            if (bc.active(vars, g, tol)) result.push_back(bc);
        }
        return result;
    }

    bool feasible(const VXd &vars) {
        for (auto &bc : boundConstraints())
            if (!bc.feasible(vars)) return false;
        return true;
    }

    // Get feasible step length and the index of the step-limiting bound
    virtual std::pair<Real, size_t> feasibleStepLength(const VXd &vars, const VXd &step) const {
        Real alpha = std::numeric_limits<Real>::max();
        size_t blocking_idx = std::numeric_limits<size_t>::max();

        for (size_t i = 0; i < m_boundConstraints.size(); ++i) {
            Real len = m_boundConstraints[i].feasibleStepLength(vars, step);
            if (len < alpha) { alpha = len; blocking_idx = i; }
        }
        return std::make_pair(alpha, blocking_idx);
    }

    // Get feasible step length and the index of the step-limiting bound
    std::pair<Real, size_t> feasibleStepLength(const VXd &step) const {
        return feasibleStepLength(getVars(), step);
    }

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Used to determine reasonable step lengths to take when the Newton step is not possible.)
    virtual Real characteristicDistance(const VXd &/* d */) const { return -1.0; }

    // Allow problems to attach custom convergence information to each optimization iterate.
    virtual void customIterateReport(ConvergenceReport &/* report */) const { }

    virtual ~NewtonProblem() { }

    bool disableCaching = false; // To be used when, e.g., this problem is wrapped by another problem which does its own Hessian caching...

protected:
    // Clear the cached per-iterate quantities
    void m_clearCache() { m_cachedHessianUpToDate = false, m_cachedMetricUpToDate = false, m_cachedMetric.reset(); /* TODO: decide if we want this: m_metricL2Norm = -1; */ }
    // Called at the start of each new iteration (after line search has been performed)
    virtual void m_iterationCallback(size_t /* i */) { }

    virtual void m_evalHessian(SuiteSparseMatrix &result, bool projectionMask) const = 0;
    virtual void m_evalMetric (SuiteSparseMatrix &result) const = 0;

    std::vector<BoundConstraint> m_boundConstraints;
    std::vector<size_t> m_fixedVars;

    bool m_useIdentityMetric = false;

    // Cached values for the mass matrix and its L2 norm
    // Mass matrix is recomputed each iteration; L2 norm is estimated only
    // once across the entire solve.
    mutable std::unique_ptr<SuiteSparseMatrix> m_cachedHessian, m_cachedMetric;
    mutable bool m_cachedHessianUpToDate = false;
    mutable bool m_cachedMetricUpToDate = false;
    mutable bool m_sparsityPatternFactorizationUpToDate = false;
    mutable Real m_metricL2Norm = -1;
};

struct MESHFEM_EXPORT WorkingSet {
    WorkingSet(const NewtonProblem &problem) : m_prob(problem), m_contains(problem.numBoundConstraints(), false), m_varFixed(problem.numVars(), false) { }
    WorkingSet(const WorkingSet &ws) : m_prob(ws.m_prob), m_count(ws.m_count), m_contains(ws.m_contains), m_varFixed(ws.m_varFixed) { }

    // Check whether the working set contains a particular constraint
    bool contains(size_t idx) const { return m_contains[idx]; }
    bool fixesVariable(size_t vidx) const { return m_varFixed[vidx]; }

    // Returns true if the index was actually newly added to the set.
    bool add(size_t idx) {
        if (contains(idx)) return false;

        const size_t vidx = m_prob.boundConstraint(idx).idx;
        if (m_varFixed[vidx]) throw std::runtime_error("Only one active bound on a variable is supported (don't impose equality constraints with bounds!)");

        m_varFixed[vidx] = true;
        m_contains[idx] = true;
        ++m_count;

        return true;
    }

    // Return "true" if entries are removed.
    template<class Predicate>
    bool remove_if(const Predicate &p) {
        const size_t nbc = m_contains.size();
        size_t old_count = m_count;
        for (size_t bci = 0; bci < nbc; ++bci) {
            if (m_contains[bci] && p(bci)) {
                m_contains[bci] = false;
                const size_t vidx = m_prob.boundConstraint(bci).idx;
                assert(m_varFixed[vidx]);
                m_varFixed[vidx] = false;
                --m_count;
            }
        }
        return m_count < old_count;
    }

    size_t size() const { return m_count; }

    void validateStep(const Eigen::VectorXd &s) const {
        for (size_t vidx = 0; vidx < m_varFixed.size(); ++vidx) {
            if (m_varFixed[vidx] && (s[vidx] != 0.0)) {
                std::cerr << "Working set not enforced properly";
                throw std::logic_error("Working set not enforced properly");
            }
        }
    }

    // Zero out the components for variables fixed by the working set. E.g., if "g" is the gradient,
    // compute the gradient with respect to the "free" variables (without resizing)
    void getFreeComponentInPlace(Eigen::Ref<Eigen::VectorXd> g) const {
        if (size_t(g.size()) != m_varFixed.size()) throw std::runtime_error("Gradient size mismatch");
        for (size_t vidx = 0; vidx < m_varFixed.size(); ++vidx)
            if (m_varFixed[vidx]) g[vidx] = 0.0;
    }

    Eigen::VectorXd getFreeComponent(Eigen::VectorXd g /* copy modified inside */) const {
        getFreeComponentInPlace(g);
        return g;
    }

    std::unique_ptr<WorkingSet> clone() const { return std::make_unique<WorkingSet>(*this); }

    const NewtonProblem &problem() const { return m_prob; }

    void report(const Eigen::VectorXd &vars, const Eigen::VectorXd &g) const {
        for (size_t bci = 0; bci < m_prob.numBoundConstraints(); ++bci) {
            if (contains(bci)) m_prob.boundConstraint(bci).report(vars, g);
        }
    }

private:
    const NewtonProblem &m_prob;
    size_t m_count = 0;
    std::vector<char> m_contains; // Whether a particular constraint is in the working set
    std::vector<char> m_varFixed; // Whether a variable is fixed by one of the constraints in the working set
};

struct NewtonOptimizerOptionsBase {
    Real gradTol = 2e-8,
         beta = 1e-8;
    bool hessianScaledBeta = true;
    size_t niter = 100;                        // Maximum number of newton iterations
    bool useIdentityMetric = false;            // Whether to force the use of the identity matrix for Hessian modification (instead of the problem's custom metric)
    bool useNegativeCurvatureDirection = true; // Whether to compute and move in negative curvature directions to escape from saddle points.
    bool feasibilitySolve = true;              // Whether to solve for a feasible starting point or rely on the problem to jump to feasible parameters.
    int verbose = 1;
    bool verboseNonPosDef = false;             // Print CHOLMOD warning for non-pos-def matrices
    int stdoutFlushInterval = 1;               // How often to flush stdout (e.g., for immediate updates in Jupyter notebook or for reduced disk i/o when redirecting to a file in a HPC setting)
    bool writeIterateFiles = false;
    // Warning: the following fields are NOT serialized for reasons of backwards compatibility
    size_t nbacktrack_iter = 25;               // Number of backtracking iterations to run before giving up on the linesearch
    size_t ngd_fallback_steps = 3;             // Total number of "fall-backs iterations" trying the neg gradient instead of the Newton direction
    int  verboseWorkingSet = 0;                // Whether to report changes to the working set (>0) and the contents of nonempty working sets upon termination (>1).
};

// The part of the optimizer interface that is not trivially copyable.
struct MESHFEM_EXPORT NewtonOptimizerOptions : public NewtonOptimizerOptionsBase {
    NewtonOptimizerOptions() = default;
    NewtonOptimizerOptions(const NewtonOptimizerOptions &b)
        : NewtonOptimizerOptionsBase(b),
          m_hessianProjectionController(b.m_hessianProjectionController->clone()),
          m_hessianUpdateController(b.m_hessianUpdateController->clone())
    { }

    NewtonOptimizerOptions &operator=(const NewtonOptimizerOptions &b) {
        NewtonOptimizerOptionsBase::operator=(b);
        m_hessianProjectionController = b.m_hessianProjectionController->clone();
        m_hessianUpdateController     = b.m_hessianUpdateController->clone();
        return *this;
    }

    HessianProjectionController &getHessianProjectionController() const { return *m_hessianProjectionController; }
    void setHessianProjectionController(const HessianProjectionController &hpc) { m_hessianProjectionController = hpc.clone(); }

    HessianUpdateController &getHessianUpdateController() const { return *m_hessianUpdateController; }
    void setHessianUpdateController(const HessianUpdateController &huc) { m_hessianUpdateController = huc.clone(); }

    ////////////////////////////////////////////////////////////////////////////
    // Serialization + cloning support (for pickling)
    ////////////////////////////////////////////////////////////////////////////
    using State = std::tuple<Real, Real, bool, size_t, bool, bool, bool, int, bool, bool, std::shared_ptr<HessianProjectionController>, std::shared_ptr<HessianUpdateController>, size_t, size_t>;
    using StateBackwardCompat = std::tuple<Real, Real, bool, size_t, bool, bool, bool, int, bool, bool, std::shared_ptr<HessianProjectionController>, std::shared_ptr<HessianUpdateController>>; // before nbacktrack_iter and ngd_fallback_steps were added
    static State serialize(const NewtonOptimizerOptions &opts) {
        return std::make_tuple(opts.gradTol,  opts.beta,
                               opts.hessianScaledBeta, opts.niter, opts.useIdentityMetric,
                               opts.useNegativeCurvatureDirection, opts.feasibilitySolve,
                               opts.verbose, opts.writeIterateFiles, opts.verboseNonPosDef,
                               opts.m_hessianProjectionController, opts.m_hessianUpdateController,
                               opts.nbacktrack_iter, opts.ngd_fallback_steps);
    }
    template<typename State_>
    static std::unique_ptr<NewtonOptimizerOptions> deserialize_(const State_ &state) {
        auto opts = std::make_unique<NewtonOptimizerOptions>();
        opts->gradTol                       = std::get<0 >(state);
        opts->beta                          = std::get<1 >(state);
        opts->hessianScaledBeta             = std::get<2 >(state);
        opts->niter                         = std::get<3 >(state);
        opts->useIdentityMetric             = std::get<4 >(state);
        opts->useNegativeCurvatureDirection = std::get<5 >(state);
        opts->feasibilitySolve              = std::get<6 >(state);
        opts->verbose                       = std::get<7 >(state);
        opts->writeIterateFiles             = std::get<8 >(state);
        opts->verboseNonPosDef              = std::get<9 >(state);
        opts->m_hessianProjectionController = std::get<10>(state);
        opts->m_hessianUpdateController     = std::get<11>(state);
        return opts;
    }
    static std::unique_ptr<NewtonOptimizerOptions> deserialize(const StateBackwardCompat &state) { return deserialize_(state); }
    static std::unique_ptr<NewtonOptimizerOptions> deserialize(const State &state) {
        auto opts = deserialize_(state);
        opts->nbacktrack_iter    = std::get<12>(state);
        opts->ngd_fallback_steps = std::get<13>(state);
        return opts;
    }
    std::unique_ptr<NewtonOptimizerOptions> clone() { return deserialize(serialize(*this)); }

protected:
    // `shared_ptr` to support pickling
    std::shared_ptr<HessianProjectionController> m_hessianProjectionController = std::make_shared<HessianProjectionAlways>();
    std::shared_ptr<HessianUpdateController>     m_hessianUpdateController     = std::make_shared<HessianUpdateAlways>();
};

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

struct MESHFEM_EXPORT NewtonOptimizer {
    NewtonOptimizer(std::unique_ptr<NewtonProblem> &&p) {
        prob = std::move(p);
        updateSymbolicFactorization(prob->hessianReducedSparsityPattern());
        const std::vector<size_t> fixedVars = prob->fixedVars();
        isFixed.assign(prob->numVars(), false);
        for (size_t fv : fixedVars) isFixed[fv] = true;
    }

    void setFixedVars(const std::vector<size_t> &fixedVars) {
        prob->setFixedVars(fixedVars);
        isFixed.assign(prob->numVars(), false);
        for (size_t fv : fixedVars) isFixed[fv] = true;
        m_solver->updateSymbolicFactorization(prob->hessianReducedSparsityPattern());
    }

    ConvergenceReport optimize();
    ConvergenceReport optimize(WorkingSet &ws);

    Real newton_step(Eigen::VectorXd &step, const Eigen::VectorXd &g, const WorkingSet &ws, Real &beta, const Real betaMin, const bool feasibility = false);

    // Calculate a Newton step with empty working set and default beta/betaMin.
    Real newton_step(Eigen::VectorXd &step, const Eigen::VectorXd &g) {
        Real beta = options.beta;
        const Real betaMin = std::min(beta, 1e-6);
        WorkingSet ws(*prob);
        return newton_step(step, g, ws, beta, betaMin);
    }

    // Update the factorizations of the Hessian/KKT system with the current
    // iterate's Hessian. This is necessary for sensitivity analysis after
    // optimize() has been called: when optimization terminates either because
    // the problem is solved or the iteration limit is reached, solver/kkt_solver
    // hold values from the previous iteration (before the final linesearch
    // step).
    void update_factorizations(const WorkingSet &ws) {
        // Computing a Newton step updates the Cholesky factorization in
        // "solver" and (if applicable) the kkt_solver as a side-effect.
        Eigen::VectorXd dummy;
        newton_step(dummy, Eigen::VectorXd::Zero(prob->numVars()), ws, options.beta, std::min(options.beta, 1e-6));
    }

    void updateSymbolicFactorization(const SuiteSparseMatrix &H) {
        m_solver = std::make_unique<CholmodFactorizer>(H);
        m_solver->factorizeSymbolic();
        prob->sparsityPatternFactorizationUpToDate(true);
    }

    CholmodFactorizer &solver() { 
        if (!m_solver) {
            m_solver = std::make_unique<CholmodFactorizer>(prob->hessianReducedSparsityPattern());
            m_solver->factorizeSymbolic();
        }
        return *m_solver;
    }

    void update_factorizations() { update_factorizations(WorkingSet(*prob)); }

    Real tauScale() const { return (options.hessianScaledBeta ? m_cachedHessianL2Norm.get(*prob) : 1.0) / prob->metricL2Norm(); }

    const NewtonProblem &get_problem() const { return *prob; }
          NewtonProblem &get_problem()       { return *prob; }

    // Construct a vector of reduced components by removing the entries of "x" corresponding
    // to fixed variables. This is a (partial) inverse of extractFullSolution.
    void removeFixedEntriesInPlace(Eigen::VectorXd &x) const {
        int back = 0;
        for (int i = 0; i < x.size(); ++i)
            if (!isFixed[i]) x[back++] = x[i];
        x.conservativeResize(back);
    }
    Eigen::VectorXd removeFixedEntries(const Eigen::VectorXd &x) const {
        auto result = x;
        removeFixedEntriesInPlace(result);
        return result;
    }

    // Extract the full linear system solution vector "x" from the reduced linear
    // system solution "xReduced" (which was solved by removing the rows/columns for fixed variables).
    void extractFullSolution(const Eigen::VectorXd &xReduced, Eigen::VectorXd &x) const {
        int back = 0;
        for (int i = 0; i < x.size(); ++i) {
            if (!isFixed[i]) x[i] = xReduced[back++];
            else             x[i] = 0.0;
        }
        assert(back == xReduced.size());
    }

    Eigen::VectorXd extractFullSolution(const Eigen::VectorXd &xReduced) const {
        Eigen::VectorXd x(prob->numVars());
        extractFullSolution(xReduced, x);
        return x;
    }

    NewtonOptimizerOptions options;
    KKTSolver kkt_solver;
    // We fix variables by constraining the newton step to have zeros for these entries
    std::vector<char> isFixed;
    mutable CachedHessianL2Norm m_cachedHessianL2Norm;

private:
    std::unique_ptr<NewtonProblem> prob;
    std::unique_ptr<CholmodFactorizer> m_solver;
};

#endif /* end of include guard: NEWTON_OPTIMIZER_HH */
