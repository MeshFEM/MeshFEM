////////////////////////////////////////////////////////////////////////////////
// MultiobjectiveProblem.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Base class hierarchy and implementation for a multiobjective optimization
//  problem composed of a number of NewtonObjectiveTerm instances, each
//  a function of the optimization variables described by `NewtonVars`.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
*///////////////////////////////////////////////////////////////////////////////
#ifndef MULTIOBJECTIVEPROBLEM_HH
#define MULTIOBJECTIVEPROBLEM_HH

#include "NewtonProblem.hh"
#include <memory>

////////////////////////////////////////////////////////////////////////////////
// Term increase limiting functionality:
////////////////////////////////////////////////////////////////////////////////
// To prevent one strong objective term from forcing the system into a bad
// state, we optionally limit the amount any given objective term can
// increase in a single Newton step. This is useful, e.g., for deployable
// structures actuated from a rest configuration (with zero elastic forces)
// using a high deployment force.
// 
// Increasing to below `valueLimitingThreshold` is always permitted.
// After exceeding that threshold, we only allow values a factor of
// `valueIncreaseFactorLimit` above the old value.
struct ObjectiveIncreaseLimiter {
    static constexpr Real NO_LIMIT = safe_numeric_limits<Real>::max();
    static constexpr Real INFTY    = safe_numeric_limits<Real>::infinity();
    Real factor = NO_LIMIT;
    Real threshold = 1e-6;
    Real previousValue; // the value as of the last Newton step

    bool valueExceedsLimit(Real v) const {
        return ((factor != NO_LIMIT) && v > factor * std::max(previousValue, threshold));
    }
};

////////////////////////////////////////////////////////////////////////////////
// Interface for a class holding the variables of a Newton optimization.
////////////////////////////////////////////////////////////////////////////////
struct NewtonVarsBase {
    using VXd = Eigen::VectorXd;

    virtual void setVars(const VXd &vars) = 0;
    virtual VXd getVars() const = 0; // TODO: avoid copy here for managers that return by reference?
    virtual size_t numVars() const = 0;

    virtual Real   approxLinfVelocity(const VXd & /* d */) const { return -1.0; }
    virtual Real characteristicLength() const { return  1.0; }

    // Called once after each Newton iteration, allowing the current
    // configuration to be reparametrized.
    virtual void updateParametrization() { }

    virtual ~NewtonVarsBase() { }
};

// Default implementation: store the variables in an Eigen array.
struct NewtonVars : public NewtonVarsBase {
    using VXd = Eigen::VectorXd;

    NewtonVars(size_t n) : m_x(n) { }
    NewtonVars(const VXd &v) : m_x(v) { }

    virtual void setVars(const VXd &vars) override { m_x = vars; }
    virtual VXd getVars() const override { return m_x; }
    virtual size_t numVars() const override { return m_x.size(); }

    // For storage-backed variables, we can return by reference.
    const VXd &vars() const { return m_x; }
protected:
    VXd m_x;
};

struct NewtonObjectiveTerm {
    using VXd = Eigen::VectorXd;

    // Inform the multiobjective class about how dynamic the sparsity pattern is:
    //      NEVER     -- Sparsity pattern is constant
    //      ALWAYS    -- Sparsity pattern changes essentially every time variables change
    //      SOMETIMES -- Sparsity pattern changes only occasionally
    enum class SparsityUpdateFrequency { NEVER, ALWAYS, SOMETIMES };

    virtual Real objective() const = 0;
    virtual void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const = 0;
    virtual void accumulateHessian(Real weight, SuiteSparseMatrix &result, bool projectionMask = false) const = 0;

    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0) const = 0;
    virtual SparsityUpdateFrequency sparsityUpdateFrequency() const { return SparsityUpdateFrequency::NEVER; }

    virtual ~NewtonObjectiveTerm() { }

    ////////////////////////////////////////////////////////////////////////////
    // Notifications
    ////////////////////////////////////////////////////////////////////////////
    virtual void varsUpdated() { }

    ////////////////////////////////////////////////////////////////////////////
    // Convenience methods
    ////////////////////////////////////////////////////////////////////////////
    SuiteSparseMatrix hessian(bool projectionMask = false) const {
        SuiteSparseMatrix H(hessianSparsityPattern());
        accumulateHessian(1.0, H, projectionMask);
        return H;
    }

    ObjectiveIncreaseLimiter increaseLimiter;

    // Option to ignore the sparsity pattern contributed by this term
    // (e.g., if we know it is a subset of the sparsity patterns of the other terms).
    bool suppressSparsity = false;
};


////////////////////////////////////////////////////////////////////////////////
// A generic multiobjective optimization composed of a "variable manager"
// (responsible for the setVars/getVars/etc. part of the NewtonProblem
// interface) and a number of NewtonObjectiveTerm instances.
////////////////////////////////////////////////////////////////////////////////
struct NewtonMultiobjectiveProblem : public NewtonProblem {
    using CallbackFunction = std::function<bool(NewtonProblem &, size_t)>;

    using TermPtr = std::shared_ptr<NewtonObjectiveTerm>;
    using NVMPtr  = std::shared_ptr<NewtonVarsBase>;

    NewtonMultiobjectiveProblem(NVMPtr vars, std::vector<TermPtr> terms)
        : m_vars(vars) {
        setTerms(terms);
    }

    virtual Real objective() const override {
        Real result = 0;

        for (size_t ti = 0; ti < numTerms(); ++ti) {
            const auto &t = term(ti);
            Real w = weight(ti);
            if (w == 0) continue;
            Real o = t.objective();

            // Bail early if any term exceeds its increase limit
            if (t.increaseLimiter.valueExceedsLimit(o))
                return ObjectiveIncreaseLimiter::INFTY;

            result += w * o;
        }
        return result;
    }

    VXd gradient(bool freshIterate = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonMultiobjectiveProblem.gradient");
        VXd g = VXd::Zero(numVars());
        for (size_t ti = 0; ti < numTerms(); ++ti)
            term(ti).accumulateGradient(weight(ti), g, freshIterate);
        
        return g;
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }

    size_t numTerms() const { return m_terms.size(); }

    size_t numVars() const override { return m_vars->numVars(); }
    VXd    getVars() const override { return m_vars->getVars(); }
    void   setVars(const VXd &vars) override {
        if (size_t(vars.size()) != numVars()) throw std::runtime_error("Incorrect variable size");
        m_vars->setVars(vars);

        for (auto &t : m_terms) t->varsUpdated();

        // TODO: update sparsity patterns
    }

    Real weight(size_t i) const { return m_weights[i]; }

    void setTerms(std::vector<TermPtr> terms) {
        m_terms = terms;
        m_termsAddedOrRemoved();
        m_weights.resize(numTerms(), 1.0);
    }

    void setWeights(std::vector<Real> weights) {
        if (m_weights.size() != numTerms()) throw std::runtime_error("Must have one weight per term");
        m_weights = weights;
    }

    const NewtonObjectiveTerm &term(size_t i) const { return *m_terms[i]; }
          NewtonObjectiveTerm &term(size_t i)       { return *m_terms[i]; }

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Useful for determining reasonable step lengths to take when the Newton step is not possible.)
    Real characteristicDistance(const Eigen::VectorXd &d) const override {
        return m_vars->characteristicLength() / m_vars->approxLinfVelocity(d);
    }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

private:
    NVMPtr m_vars;
    std::vector<TermPtr> m_terms;
    std::vector<Real> m_weights;

    SuiteSparseMatrix m_hessianSparsity, m_staticSparsityPattern;
    CallbackFunction m_customCallback;

    void m_termsAddedOrRemoved() {
        if (numTerms() == 0) throw std::runtime_error("Must have at least one term");

        // Note: empty sparsity patterns simply get replaced by `addWithDistinctSparsityPattern`
        m_hessianSparsity.m = m_hessianSparsity.n = numVars();
        m_hessianSparsity.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;

        for (size_t i = 0; i < numTerms(); ++i) {
            const auto &t = term(i);
            if (t.suppressSparsity) continue;
            m_hessianSparsity.addWithDistinctSparsityPattern(t.hessianSparsityPattern());
        }
        m_hessianSparsity.fill(1.0);
    }

    virtual void m_evalHessian(SuiteSparseMatrix &result, bool projectionMask) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonMultiobjectiveProblem.hessian");
        result.data().setZero();
        for (size_t ti = 0; ti < numTerms(); ++ti)
            term(ti).accumulateHessian(weight(ti), result, projectionMask);
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        throw std::runtime_error("No metric implemented for this problem.");
    }

    virtual bool m_iterationCallback(size_t i) override {
        for (size_t ti = 0; ti < numTerms(); ++ti) {
            auto &t = *m_terms[ti];
            auto &limit = t.increaseLimiter;
            if (limit.factor == ObjectiveIncreaseLimiter::NO_LIMIT)
                limit.previousValue = safe_numeric_limits<Real>::max();
            else limit.previousValue = t.objective();
        }

        m_vars->updateParametrization();

        if (m_customCallback) return m_customCallback(*this, i);
        return false; // don't exit early
    }
};

#endif /* end of include guard: MULTIOBJECTIVEPROBLEM_HH */
