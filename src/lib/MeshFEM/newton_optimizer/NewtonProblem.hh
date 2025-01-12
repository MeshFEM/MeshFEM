////////////////////////////////////////////////////////////////////////////////
// NewtonProblem.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Base class for problems that can be solved with `NewtonOptimizer`.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
*///////////////////////////////////////////////////////////////////////////////
#ifndef NEWTONPROBLEM_HH
#define NEWTONPROBLEM_HH

#include <MeshFEM/SparseMatrices.hh>
#include "ConvergenceReport.hh"
#include <MeshFEM/Eigensolver.hh>
#include "NewtonHessian.hh"

struct MESHFEM_EXPORT NewtonProblem {
    using VXd = Eigen::VectorXd;
    virtual void setVars(const VXd &vars) = 0;
    virtual VXd getVars() const = 0;
    virtual size_t numVars() const = 0;

    // Called at the start of each new iteration (after line search has been performed)
    bool iterationCallback(size_t i) {
        bool earlyExit = m_iterationCallback(i);
        m_clearCache(); // This must happen after `m_iterationCallback` in case
                        // the user's callback calls `hessian` with the wrong
                        // `projectionMask` or with incorrect variables.
        return earlyExit;
    }

    virtual Real objective() const = 0;
    // freshIterate: whether the gradient is being called immediately
    // after an iteration callback (without any change to the variables in between) instead
    // of, e.g., during the line search.
    // For some problems, a less expensive gradient expression can be used in this case.
    virtual VXd gradient(bool freshIterate = false) const = 0;

    const NewtonHessian &hessian(bool projectionMask = true) const {
        if (!m_cachedHessian) { m_cachedHessian = std::make_unique<NewtonHessian>(hessianSparsityPattern()); }
        if (disableCaching || !m_cachedHessianUpToDate) {
            m_evalHessian(*m_cachedHessian, projectionMask);
            m_cachedHessianUpToDate = true;
        }
        return *m_cachedHessian;
    }

    virtual bool providesMetric() const { return false; }

    // Positive definite matrix defining the metric used to define trust regions.
    // For efficiency, it must have the same sparsity pattern as the Hessian.
    // (This matrix is added to indefinite Hessians to produce a positive definite modified Hessian.)
    const SuiteSparseMatrix &metric() const {
        if (m_useIdentityMetric) {
            if (!m_identityMetric) {
                m_identityMetric = std::make_unique<SuiteSparseMatrix>(hessianSparsityPattern().toScalar());
                m_identityMetric->setIdentity(true);
            }
            return *m_identityMetric;
        }
        if (disableCaching || !m_cachedMetric) {
            m_cachedMetric = std::make_unique<SuiteSparseMatrix>(hessianSparsityPattern().toScalar());
            m_evalMetric(*m_cachedMetric);
        }
        return *m_cachedMetric;
    }

    Real hessianL2Norm() const { return largestMagnitudeEigenvalue(hessian(), 1e-1); }

    // Since computing the L2 norm is slightly expensive, we assume that it remains
    // constant throughout the solve. This is exactly true for ElasticRods, and should be
    // a good approximation for RodLinkages under mild deformation.
    // Also, an exact result should not be necessary since it's only used to determine a reasonable
    // initial guess for the Hessian modification magnitude.
    Real metricL2Norm() const {
        if (!providesMetric() || m_useIdentityMetric) return 1.0;
        if (m_metricL2Norm <= 0) m_metricL2Norm = largestMagnitudeEigenvalue(metric(), 1e-1);
        return m_metricL2Norm;
    }
    void setUseIdentityMetric(bool useIdentityMetric) { m_useIdentityMetric = useIdentityMetric; }

    // A compressed column sparse matrix with nonzero placeholders wherever the Hessian can ever have nonzero entries.
    NewtonHessian hessianSparsityPattern() const { updateSparsityPattern(); return m_getHessianSparsityPattern(); }

    void updateSparsityPattern() const {
        if (!m_updateSparsityPattern()) return; // No change

        m_cachedHessianUpToDate = false;
        m_cachedHessian.reset(); m_cachedMetric.reset(); // Cached matrices must be thrown out so they're reconstructed from scratch with the correct sparsity pattern!
        ++m_sparsityPatternID;
    }

    // Identifier used to determine whether a symbolic factorization has been
    // invalidated by a sparsity pattern change; this ID increments whenever the
    // sparsity pattern updates.
    size_t sparsityPatternID() const { return m_sparsityPatternID; }

    // A **sorted, unique** list of indices of variables that are fixed in this problem.
    const std::vector<size_t> &fixedVars() const { return m_fixedVars; }
    size_t numFixedVars() const { return fixedVars().size(); }
    size_t numReducedVars() const { return numVars() - fixedVars().size(); } // number of remaining variables after fixing fixedVars

    // WARNING: updating the fixed variables *after* constructing a
    // NewtonOptimizer from this problem won't work; then you must call
    // NewtonOptimizer::setFixedVars.
    void setFixedVars(std::vector<size_t> fv) { // Pass-by-value is intentional due to copy assignment below
        m_fixedVars = fv;
        std::sort(std::begin(m_fixedVars), std::end(m_fixedVars));
        m_fixedVars.erase(std::unique(std::begin(m_fixedVars), std::end(m_fixedVars)), std::end(m_fixedVars));
    }

    void addFixedVariables(const std::vector<size_t> &fv) {
        std::vector<size_t> fvNew;
        fvNew.reserve(fixedVars().size() + fv.size());
        fvNew.insert(fvNew.end(), fixedVars().begin(), fixedVars().end());
        fvNew.insert(fvNew.end(), fv.begin(), fv.end());
        setFixedVars(fvNew);
    }

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
        bool shouldRemoveFromWorkingSet(const VXd &neg_g, Real g_free_norm) const {
            if (type == Type::UPPER) { return neg_g[idx] <  10 * g_free_norm; }
            if (type == Type::LOWER) { return neg_g[idx] > -10 * g_free_norm; }
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
    std::pair<Real, size_t> feasibleStepLength(const VXd &vars, const VXd &step) const {
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
    void invalidateCachedHessian() { m_cachedHessianUpToDate = false; }

    // Allow the derived problem to update its Hessian sparsity pattern.
    // This will be called between last point the problem state can change (the
    // custom user callback) and the point where the Hessian is evaluated.
    // If the sparsity pattern changes, `hesianSparsityPatternChanged` should be called.
    virtual bool detectSparsityPatternUpdates() { return false; }

    // Allow subclasses to impose an upper bound on the step size (e.g., to
    // enforce interpenetration-free steps).
    virtual Real customFeasibleStepLength(const VXd &vars, const VXd &step) const { return std::numeric_limits<Real>::max(); }

    // End of line search notification
    virtual void lineSearchTerminated() const { }

    // When nonzero, the matrix `H + hessianShift I` is factorized at each
    // Newton step rather than `H` itself. This is intended for problems
    // with a Hessian nullspace due to, e.g., rigid motion, that can be
    // removed by a *small* shift.
    // Note: if `H + hessianShift I` is indefinite, then the
    // Hessian modification `H + sigma M` *replaces* this shift rather than
    // adding to it.
    Real hessianShift = 0.0;

protected:
    // Clear the cached per-iterate quantities
    void m_clearCache() { m_cachedHessianUpToDate = false, m_cachedMetric.reset(); /* TODO: decide if we want this: m_metricL2Norm = -1; */ }
    // Called at the start of each new iteration (after line search has been performed)
    // Returns true to exit early.
    virtual bool m_iterationCallback(size_t /* i */) { return false; }

    virtual NewtonHessian m_getHessianSparsityPattern() const = 0;
    virtual void m_evalHessian(NewtonHessian &result, bool projectionMask) const = 0;
    virtual void m_evalMetric (SuiteSparseMatrix &result) const = 0;
    // Ask subclass to update its sparsity pattern if needed; returns `true` if the pattern changed.
    virtual bool m_updateSparsityPattern() const = 0;

    std::vector<BoundConstraint> m_boundConstraints;
    std::vector<size_t> m_fixedVars;

    bool m_useIdentityMetric = false;

    // Cached values for the mass matrix and its L2 norm
    // Mass matrix is recomputed each iteration; L2 norm is estimated only
    // once across the entire solve.
    mutable std::unique_ptr<SuiteSparseMatrix> m_cachedMetric, m_identityMetric;
    mutable std::unique_ptr<NewtonHessian> m_cachedHessian;
    mutable bool m_cachedHessianUpToDate = false;
    mutable Real m_metricL2Norm = -1;

    mutable size_t m_sparsityPatternID = 0;
};

#endif /* end of include guard: NEWTONPROBLEM_HH */
