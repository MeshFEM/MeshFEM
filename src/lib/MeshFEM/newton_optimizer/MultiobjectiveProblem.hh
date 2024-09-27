////////////////////////////////////////////////////////////////////////////////
// MultiobjectiveProblem.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Base class hierarchy and implementation for a multiobjective optimization
//  problem composed of a number of `NewtonObjectiveTermBase` instances, each
//  a function of the optimization variables described by `NewtonVarsBase`.
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
struct MESHFEM_EXPORT NewtonVarsBase {
    using VXd = Eigen::VectorXd;
    using NotificationCB = std::function<void()>;

    virtual size_t numVars() const = 0;
    virtual VXd getVars() const = 0; // TODO: avoid copy here for managers that return by reference?
    void setVars(const VXd &vars) {
        if (size_t(vars.size()) != numVars()) throw std::runtime_error("Variable size mismatch");
        m_setVarsImpl(vars);
        m_issueNotifications(VarType::Variable);
    }

    virtual size_t numParameters() const = 0;
    virtual VXd getParameters() const = 0;
    void setParameters(const VXd &params) {
        if (size_t(params.size()) != numParameters()) throw std::runtime_error("Parameter size mismatch");
        m_setParametersImpl(params);
        m_issueNotifications(VarType::Parameter);
    }

    virtual Real   approxLinfVelocity(const VXd & /* d */) const { return -1.0; }
    virtual Real characteristicLength() const { return  1.0; }

    // Called once after each Newton iteration, allowing the current
    // configuration to be reparametrized.
    virtual void updateParametrization() { }

    // Support rolling back to a previous parametrization to perfectly restore
    // a past configuration.
    virtual void setParametrizationState(const VXd &/* state */) { }

    ////////////////////////////////////////////////////////////////////////////////
    // Variable update notification mechanism
    // (Allow objective terms to register for automatic updates when variables
    // or parameters change).
    ////////////////////////////////////////////////////////////////////////////////
    // Distinguish between the optimization variables
    // (e.g., deformed state of an equilibrium problem) and
    // parameters of the objective function (e.g., the rest state).
    enum class VarType { Variable, Parameter };

    int registerUpdateCallback(VarType type, const NotificationCB &cb) const {
        if ((type != VarType::Variable) && (type != VarType::Parameter))
            throw std::runtime_error("`type` must be VarType::Variable or VarType::Parameter");
        int id;
        while (m_updateCBs.count(id = rand()));
        m_updateCBs.emplace(id, CBRecord{type, cb});
        return id;
    }

    void deregisterUpdateCallback(int id) const {
        auto it = m_updateCBs.find(id);
        if (it == m_updateCBs.end()) throw std::runtime_error("Attempted to deregister nonexistent callback");
        m_updateCBs.erase(it);
    }

    virtual ~NewtonVarsBase();
protected:
    void m_issueNotifications(VarType type) const {
        for (const auto &it : m_updateCBs) {
            const CBRecord &record = it.second;
            if (record.type == type)
                record.cb();
        }
    }

private:
    // State update notifications
    struct CBRecord {
        VarType type;
        NotificationCB cb;
    };

    virtual void m_setVarsImpl(const VXd &vars) = 0;
    virtual void m_setParametersImpl(const VXd &params) = 0;

    mutable std::map<int, CBRecord> m_updateCBs;
};

// Default implementation: store the variables in an Eigen array.
struct MESHFEM_EXPORT NewtonVars : public NewtonVarsBase {
    using VXd = Eigen::VectorXd;

    NewtonVars(size_t n, size_t numParams = 0) : m_x(n), m_p(numParams) { }
    NewtonVars(const VXd &v) : m_x(v) { }

    virtual size_t numVars() const override { return m_x.size(); }
    virtual VXd getVars() const override { return m_x; }

    virtual size_t numParameters() const override { return m_p.size(); }
    virtual VXd getParameters() const override { return m_p; }

    // For storage-backed variables, we can return by reference.
    const VXd &  vars() const { return m_x; }
    const VXd &params() const { return m_p; }

    ~NewtonVars();
protected:
    VXd m_x, m_p;
private:
    virtual void m_setVarsImpl(const VXd &vars) override { m_x = vars; }
    virtual void m_setParametersImpl(const VXd &params) override { m_p = params; }
};

// The main objective term interface (but without any storage/access to the
// optimizaton variables). Most objective terms will instead want to derive from
// `NewtonObjectiveTerm`, which will allows access to variables and supports
// notifications of variable changes.
// This base class is appropriate for classes like `ElasticObject` that
// implement both the `NewtonVarsBase` and `NewtonObjectiveTermBase` interfaces
// (and so can keep track of variable updates themselves).
struct MESHFEM_EXPORT NewtonObjectiveTermBase {
    using VXd = Eigen::VectorXd;
    using VT = NewtonVarsBase::VarType;

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

    virtual ~NewtonObjectiveTermBase();

    ////////////////////////////////////////////////////////////////////////////
    // Convenience methods
    ////////////////////////////////////////////////////////////////////////////
    virtual size_t       numVars() const { throw std::runtime_error(      "numVars must be implemented by subclass of NewtonObjectiveTermBase"); }
    virtual size_t numParameters() const { throw std::runtime_error("numParameters must be implemented by subclass of NewtonObjectiveTermBase"); }

    SuiteSparseMatrix hessian(bool projectionMask = false) const {
        SuiteSparseMatrix H(hessianSparsityPattern());
        accumulateHessian(1.0, H, projectionMask);
        return H;
    }

    virtual VXd gradient(Real weight = 1.0, bool freshIterate = false) const {
        VXd g;
        g.setZero(numVars());
        accumulateGradient(weight, g, freshIterate);
        return g;
    }

    ObjectiveIncreaseLimiter increaseLimiter;

    // Option to ignore the sparsity pattern contributed by this term
    // (e.g., if we know it is a subset of the sparsity patterns of the other terms).
    bool suppressSparsity = false;
};

struct MESHFEM_EXPORT NewtonObjectiveTerm : public NewtonObjectiveTermBase {
    using VXd = Eigen::VectorXd;
    using NVStorageType = std::weak_ptr<const NewtonVarsBase>;
    using VT = NewtonVarsBase::VarType;

    NewtonObjectiveTerm(const NVStorageType &nvars)
        : m_nvars(nvars)
    {
        m_variablesUpdateCallbackID = getNVars().registerUpdateCallback(VT::Variable,  [this]() {   varsUpdated(); });
        m_parameterUpdateCallbackID = getNVars().registerUpdateCallback(VT::Parameter, [this]() { paramsUpdated(); });
    }
    const NewtonVarsBase &getNVars() const {
        if (auto v = m_nvars.lock()) return *v;
        throw std::runtime_error("NewtonVars were destroyed");
    }

    virtual size_t numVars()       const override { return getNVars().numVars(); }
    virtual size_t numParameters() const override { return getNVars().numParameters(); }

    std::shared_ptr<const NewtonVarsBase> getNVarsPtr() const { return m_nvars.lock(); }

    virtual ~NewtonObjectiveTerm();

    ////////////////////////////////////////////////////////////////////////////
    // Notifications
    ////////////////////////////////////////////////////////////////////////////
    virtual void   varsUpdated() { }
    virtual void paramsUpdated() { }
private:
    int m_variablesUpdateCallbackID, m_parameterUpdateCallbackID;
    NVStorageType m_nvars;
};

template<class TermType>
struct MultiObjective {
    using TermPtr = std::shared_ptr<TermType>;
    using VXd = typename TermType::VXd;

    MultiObjective(std::vector<TermPtr> terms) { setTerms(terms); }

    size_t numTerms() const { return m_terms.size(); }

    void setTerms(std::vector<TermPtr> terms) {
        m_terms = terms;
        m_termsAddedOrRemoved();
        m_weights.resize(numTerms(), 1.0);

        m_names.resize(numTerms());
        for (size_t i = 0; i < numTerms(); ++i)
            m_names[i] = "Term " + std::to_string(i);
    }

    void setTermNames(std::vector<std::string> names) {
        if (names.size() != m_terms.size()) throw std::runtime_error("Term count mismatch");
        m_names = names;
    }

    const std::string &termName(size_t i) const { return m_names.at(i); }
    const std::vector<std::string> &getTermNames() const { return m_names; }

    void setWeights(const std::vector<Real> &weights) {
        if (weights.size() != numTerms()) throw std::runtime_error("Must have one weight per term (" + std::to_string(numTerms()) + " terms, " + std::to_string(weights.size()) + " weights)");
        m_weights = weights;
    }

    const std::vector<Real> &getWeights() const { return m_weights; }

    Real weight(size_t i) const { return m_weights[i]; }
    const NewtonObjectiveTermBase &term(size_t i) const { return *m_terms[i]; }
          NewtonObjectiveTermBase &term(size_t i)       { return *m_terms[i]; }

    // Accessing terms by name
    size_t termIndex(const std::string &name) const {
        auto it = std::find(m_names.begin(), m_names.end(), name);
        if (it == m_names.end()) throw std::runtime_error("Term not found: " + name);
        return std::distance(m_names.begin(), it);
    }

    Real weight(const std::string &name) const { return weight(termIndex(name)); }

    const NewtonObjectiveTermBase &term(const std::string &name) const { return term(termIndex(name)); }
          NewtonObjectiveTermBase &term(const std::string &name)       { return term(termIndex(name)); }

    std::map<std::string, Real> termObjectives() const {
        std::map<std::string, Real> result;
        for (size_t i = 0; i < numTerms(); ++i)
            result[m_names[i]] = term(i).objective();
        return result;
    }

    std::map<std::string, VXd> termGradients() const {
        std::map<std::string, VXd> result;
        for (size_t i = 0; i < numTerms(); ++i)
            result[m_names[i]] = term(i).gradient();
        return result;
    }

    virtual ~MultiObjective() = default;

protected:
    std::vector<TermPtr> m_terms;
    std::vector<Real> m_weights;
    std::vector<std::string> m_names;

    virtual void m_termsAddedOrRemoved() = 0;
};

////////////////////////////////////////////////////////////////////////////////
// A generic multiobjective optimization composed of a "variable manager"
// (responsible for the setVars/getVars/etc. part of the NewtonProblem
// interface) and a number of NewtonObjectiveTermBase instances.
////////////////////////////////////////////////////////////////////////////////
struct MESHFEM_EXPORT NewtonMultiobjectiveProblem : public NewtonProblem, public MultiObjective<NewtonObjectiveTermBase> {
    using CallbackFunction = std::function<bool(NewtonProblem &, size_t)>;

    using MO      = MultiObjective<NewtonObjectiveTermBase>;
    using TermPtr = typename MO::TermPtr;
    using NVMPtr  = std::shared_ptr<NewtonVarsBase>;

    NewtonMultiobjectiveProblem(NVMPtr vars, std::vector<TermPtr> terms)
        : MO(terms), m_vars(vars) {
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

    size_t numVars() const override { return m_vars->numVars(); }
    VXd    getVars() const override { return m_vars->getVars(); }
    void   setVars(const VXd &vars) override {
        if (size_t(vars.size()) != numVars()) throw std::runtime_error("Incorrect variable vector size");
        m_vars->setVars(vars);
    }

    size_t numParameters() const { return m_vars->numParameters(); }
    VXd    getParameters() const { return m_vars->getParameters(); }
    void   setParameters(const VXd &p) {
        if (size_t(p.size()) != numParameters()) throw std::runtime_error("Incorrect parameter vector size");
        m_vars->setParameters(p);
    }

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Useful for determining reasonable step lengths to take when the Newton step is not possible.)
    Real characteristicDistance(const Eigen::VectorXd &d) const override {
        return m_vars->characteristicLength() / m_vars->approxLinfVelocity(d);
    }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

    virtual ~NewtonMultiobjectiveProblem();

private:
    NVMPtr m_vars;

    SuiteSparseMatrix m_hessianSparsity, m_staticSparsityPattern;
    CallbackFunction m_customCallback;

    void m_termsAddedOrRemoved() override {
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
