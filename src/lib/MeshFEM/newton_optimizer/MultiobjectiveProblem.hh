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
#include "FeasibleStepLengthComputer.hh"
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/SparsityLRU.hh>
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
    Real previousValue = INFTY; // the value as of the last Newton step

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
    virtual    VXd getVars() const = 0; // TODO: avoid copy here for managers that return by reference?

    bool   hasAssembler() const { return bool(m_assembler); }
    size_t numBlockVars() const { return assembler().numBlockVars(); }

    void setVars(const VXd &vars) {
        if (size_t(vars.size()) != numVars()) throw std::runtime_error("Variable size mismatch");
        m_setVarsImpl(vars);
        m_issueNotifications(VarType::Variable);
    }

    // Set x to a trivial starting point for solving the optimization problem.
    // E.g., for an elasticity simulation, this should apply the identity deformation.
    virtual void applyTrivialInitialGuess() { setVars(VXd::Zero(numVars())); }

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

    // Support rolling back to a previous parametrization to perfectly restore a past configuration.
    virtual VXd  getParametrizationState() const { return VXd(); }
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

    virtual const SystemAssemblerBase &assembler() const {
        if (!m_assembler) throw std::runtime_error("System assembler was not set");
        return *m_assembler;
    }

protected:
    void m_issueNotifications(VarType type) const {
        for (const auto &it : m_updateCBs) {
            const CBRecord &record = it.second;
            if (record.type == type)
                record.cb();
        }
    }

    std::unique_ptr<SystemAssemblerBase> m_assembler;

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
    NewtonVars(size_t nvars = 0, size_t numParams = 0) : m_x(nvars), m_p(numParams) { }
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

struct NewtonMultiobjectiveProblem; // Forward declaration for friend-ing below.

// Sparsity pattern change detection: has a term's sparsity pattern changed
// since the last rebuild of the full problem's sparsity pattern?
// Terms will set this flag when they detect a change, and the flag will be
// cleared only by NewtonMultiobjectiveProblem::m_rebuildSparsityPattern.
struct SparsityPatternChangeFlag {
    operator bool() const { return m_sparsityPatternChanged; }
    void set() { m_sparsityPatternChanged =  true; }
private:
    friend struct NewtonMultiobjectiveProblem; // Only NewtonMultiobjectiveProblem can clear the flag!
    void clear() { m_sparsityPatternChanged = false; }
    bool m_sparsityPatternChanged = true; // Ensure that the first call to `m_updateSparsityPattern` triggers an initial build.
};

// The main objective term interface (but without any storage/access to the
// optimizaton variables). Most objective terms will instead want to derive from
// `NewtonObjectiveTerm`, which allows access to variables and supports
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
    virtual void accumulateHessian(Real weight, NewtonHessian &result, bool projectionMask = false) const = 0;

    virtual NewtonHessian hessianSparsityPattern()            const { throw std::runtime_error("hessianSparsityPattern not implemented by" + std::string(typeid(*this).name())); }
    virtual SparsityUpdateFrequency sparsityUpdateFrequency() const { return SparsityUpdateFrequency::NEVER; }

    // Sensitivity analysis support
    // Accumulate the matvec: result_accum += weight * (d2E / dpdx) * adjoint_state
    virtual void contract_d2E_dpdx(Real weight, const VXd &adjoint_state, VXd &result_accum) const { throw std::runtime_error("contract_d2E_dpdx unimplemented"); }

    // Allow subclasses to impose an upper bound on the step size `alpha`
    // before starting the line search (e.g., to enforce interpenetration-free steps).
    // This method is called on each term in a `NewtonMultiobjectiveProblem`
    // in order. The step length imposed by/feasible for all previous terms is
    // passed as `initialAlpha`. The current value of the full
    // `NewtonMultiobjectiveProblem` objective is also passed in case
    // the term wishes to implement an energy-based step length filter
    // (e.g., if all objective terms are known to be positive, then
    //  backtracking is necessary if a single term already exceeds the current
    //  objective value).
    virtual Real feasibleStepLength(const VXd &/* vars */, const VXd &/* step */, Real initialAlpha = 1.0, Real /* currentObjectiveValue */ = std::numeric_limits<Real>::max()) const { return initialAlpha; }

    ////////////////////////////////////////////////////////////////////////////
    // Convenience methods
    ////////////////////////////////////////////////////////////////////////////
    virtual size_t       numVars() const { throw std::runtime_error(      "numVars must be implemented by subclass of NewtonObjectiveTermBase"); }
    virtual size_t numParameters() const { throw std::runtime_error("numParameters must be implemented by subclass of NewtonObjectiveTermBase"); }

    ////////////////////////////////////////////////////////////////////////////
    // Notifications
    ////////////////////////////////////////////////////////////////////////////
    virtual void newtonIterationBegan() { }
    virtual void lineSearchTerminated() { }

    NewtonHessian hessian(bool projectionMask = false) const {
        NewtonHessian H(hessianSparsityPattern());
        H.insertSparsityPatternDiagonalBlocksIfNeeded();
        H.setZero();
        accumulateHessian(1.0, H, projectionMask);
        return H;
    }

    virtual VXd gradient(Real weight = 1.0, bool freshIterate = false) const {
        VXd g;
        g.setZero(numVars());
        accumulateGradient(weight, g, freshIterate);
        return g;
    }

    virtual ~NewtonObjectiveTermBase();

    ObjectiveIncreaseLimiter increaseLimiter;

    // Option to ignore the sparsity pattern contributed by this term
    // (e.g., if we know it is a subset of the sparsity patterns of the other terms).
    bool suppressSparsity = false;

    // Notify the term that it should check now for sparsity pattern changes
    // (unless it does so automatically on each variable change).
    // To help the term detect changes, it is passed the current sparsity
    // pattern that the multiobjective problem has on file.
    virtual bool detectSparsityPatternChange(const NewtonHessian &oldHsp) const { return sparsityPatternChanged; }

    mutable SparsityPatternChangeFlag sparsityPatternChanged;
};

struct MESHFEM_EXPORT NewtonObjectiveTerm : public NewtonObjectiveTermBase {
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

    size_t numTerms() const { return m_terms.size(); }

    void setTerms(std::vector<TermPtr> terms) {
        m_terms = terms;
        m_termsAddedOrRemoved();
        m_weights.resize(numTerms(), 1.0);

        m_names.resize(numTerms());
        for (size_t i = 0; i < numTerms(); ++i)
            m_names[i] = "Term " + std::to_string(i);
    }

    const std::vector<TermPtr> &getTerms() const { return m_terms; }

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

    void validateTermIndex(size_t i) const { if (i >= numTerms()) throw std::runtime_error("Term index " + std::to_string(i) + " out of bounds"); }

    Real weight(size_t i) const { validateTermIndex(i); return m_weights[i]; }
    const NewtonObjectiveTermBase &term(size_t i) const { validateTermIndex(i); return *m_terms[i]; }
          NewtonObjectiveTermBase &term(size_t i)       { validateTermIndex(i); return *m_terms[i]; }

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

    std::map<std::string, Eigen::VectorXd> termGradients() const {
        std::map<std::string, Eigen::VectorXd> result;
        for (size_t i = 0; i < numTerms(); ++i)
            result[m_names[i]] = term(i).gradient();
        return result;
    }

    virtual ~MultiObjective() = default;

protected:
    std::vector<TermPtr> m_terms;
    std::vector<Real> m_weights;
    std::vector<std::string> m_names;

    virtual void m_termsAddedOrRemoved() { }
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

    size_t numVars() const override { return m_vars->numVars(); }
    VXd    getVars() const override { return m_vars->getVars(); }
    void   setVars(const VXd &vars) override {
        if (size_t(vars.size()) != numVars()) throw std::runtime_error("Incorrect variable vector size");
        m_vars->setVars(vars);
    }

    void applyTrivialInitialGuess() { m_vars->applyTrivialInitialGuess(); }

    VXd  getParametrizationState() const           { return m_vars->getParametrizationState(); }
    void setParametrizationState(const VXd &state) { m_vars->setParametrizationState(state); }

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

    // The user can attach a FeasibleStepLengthComputer that determines an
    // initial upper bound for the feasible step length before each term is
    // queryed for its own feasible step length. An example use case is in
    // injective parameterization, where we seek a step that prevents elements
    // from inverting.
    std::shared_ptr<FeasibleStepLengthComputer> initialFeasibleStepLengthComputer;

    // Allow subclasses to impose an upper bound on the step size (e.g., to
    // enforce interpenetration-free steps).
    virtual Real customFeasibleStepLength(const VXd &vars, const VXd &step) const override {
        Real stepLength = 1.0;
        if (initialFeasibleStepLengthComputer)
            stepLength = std::min(stepLength, initialFeasibleStepLengthComputer->eval(vars, step));

        for (const auto &term : m_terms)
            stepLength = std::min(term->feasibleStepLength(vars, step, stepLength), stepLength); // Note that the `min` here shouldn't be necessary because the term should never grow the step length. But we keep it just in case...
        return stepLength;
    }

    virtual void lineSearchTerminated() const override {
        for (auto &term: m_terms)
            term->lineSearchTerminated();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Sensitivity analysis support
    ////////////////////////////////////////////////////////////////////////////
    VXd contract_d2E_dpdx(const VXd &adjoint_state) const {
        VXd result;
        result.setZero(numParameters());

        for (size_t ti = 0; ti < numTerms(); ++ti)
            term(ti).contract_d2E_dpdx(weight(ti), adjoint_state, result);

        return result;
    }

    VXd dirichletSensitivityTerm(const VXd &dJ_dx, const VXd &adjoint_state) const { throw std::runtime_error("Unimplemented"); }

    virtual ~NewtonMultiobjectiveProblem();

private:
    NVMPtr m_vars;

    mutable bool m_fullSparsityRebuildNeeded = true; // Whether all cached sparsity information has been invalidated (e.g., if the list of terms has changed)

    // The full Hessian sparsity pattern is composed of three parts:
    // a "static part", a "dynamic part", and a "semistatic part";
    // these are contributed to by terms with an update frequency of NEVER,
    // ALWAYS, and SOMETIMES, respectively.
    // We cache the full static part as well as the per-term sparsity pattern
    // for each semi-static term; the lattern is helpful for detecting
    // sparsity pattern changes (e.g., for IPCObjectiveTerm).
    // The dynamic part is rebuilt with every call to m_updateSparsityPattern().
    mutable NewtonHessian m_hessianSparsity, m_hessianSparsityStaticPart;
    mutable std::vector<NewtonHessian> m_hessianSparsityForSemistaticTerms;
    mutable std::unique_ptr<SparsityLRU> m_sparsityLRU; // Nonzero caching/retaining mechanism for minimizing Symbolic refactorizations

    CallbackFunction m_customCallback;

    using SUF = NewtonObjectiveTermBase::SparsityUpdateFrequency;
    bool m_updateSparsityPattern() const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonMultiobjectiveProblem.m_updateSparsityPattern");
        NewtonHessian dynamicSparsity;
        const bool force = m_fullSparsityRebuildNeeded;
        if (force) {
            m_hessianSparsity = NewtonHessian();
            m_hessianSparsityStaticPart = NewtonHessian();
            m_hessianSparsityForSemistaticTerms.assign(numTerms(), NewtonHessian());
        }

        bool changed = force;
        bool staticOnly = true;

        for (size_t i = 0; i < numTerms(); ++i) {
            const auto &t = term(i);
            if (t.suppressSparsity) continue;
            if (t.sparsityUpdateFrequency() == SUF::NEVER) {
                // Only rebuild the "static" part when the terms might have been invalidated.
                if (force) { m_hessianSparsityStaticPart.mergeSparsityPattern(t.hessianSparsityPattern()); std::cout << "Building static term sparsity pattern" << std::endl; }
                else if (t.sparsityPatternChanged) throw std::logic_error("Term with a sparsity pattern update frequency of NEVER reported a change.");
            }
            else if (t.sparsityUpdateFrequency() == SUF::SOMETIMES) {
                // Automatically rebuild the sparsity pattern for this term if
                // we're forcing a rebuild or if the term has already flagged a
                // change. Otherwise, we ask the term to detect a change
                // wrt. the currently incorporated version of its sparsity pattern.
                if (force || t.sparsityPatternChanged || t.detectSparsityPatternChange(m_hessianSparsityForSemistaticTerms[i])) {
                    // std::cout << "Building semistatic term sparsity pattern" << std::endl;
                    m_hessianSparsityForSemistaticTerms[i] = t.hessianSparsityPattern();
                    changed = true;
                    staticOnly = false;
                }
            }
            else {
                dynamicSparsity.mergeSparsityPattern(t.hessianSparsityPattern());
                changed = true;
                staticOnly = false;
            }
        }

        if (changed) {
            if (staticOnly) m_hessianSparsity = std::move(m_hessianSparsityStaticPart);
            else {
                if (!m_sparsityLRU && m_hessianSparsityStaticPart.H_ss && (m_hessianSparsityStaticPart.H_ss->nnz() > 0)) {
                    m_sparsityLRU = std::make_unique<SparsityLRU>(*(m_hessianSparsityStaticPart.H_ss));
                    m_hessianSparsity = std::move(m_hessianSparsityStaticPart.H_ss);
                }

                // TODO: what about the dense parts?
                NewtonHessian nonstaticPart = std::move(dynamicSparsity);
                for (const auto &t : m_hessianSparsityForSemistaticTerms)
                    nonstaticPart.mergeSparsityPattern(t);

                if (m_sparsityLRU) {
                    changed = m_sparsityLRU->update(*(nonstaticPart.H_ss));
                    changed |= force; // Ensure the static part rebuild takes effect.
                    if (changed) {
                        if (!m_hessianSparsity.H_ss) throw std::logic_error("NewtonMultiobjectiveProblem::m_updateSparsityPattern: m_hessianSparsity not initialized"); // This should never happen since `m_sparsityLRU` is only created when the static part is nonempty...
                        m_hessianSparsity.H_ss->Ap = (*m_sparsityLRU)->Ap;
                        m_hessianSparsity.H_ss->Ai = (*m_sparsityLRU)->Ai;
                        m_hessianSparsity.H_ss->nz = (*m_sparsityLRU)->nz;
                    }
                }
                else {
                    m_hessianSparsity = m_hessianSparsityStaticPart;
                    m_hessianSparsity.mergeSparsityPattern(nonstaticPart);
                }
            }

            m_hessianSparsity.finalize();
        }
        else if (m_sparsityLRU) {
            // Still notify the cache of the sparsity pattern update in case
            // it triggers a refactorization due to entry expiration.
            if (m_sparsityLRU->increaseAgeOfOldEntries()) {
                if (!m_hessianSparsity.H_ss) throw std::logic_error("NewtonMultiobjectiveProblem::m_updateSparsityPattern: m_hessianSparsity not initialized"); // This should never happen since `m_sparsityLRU` is only created when the static part is nonempty...
                m_hessianSparsity.H_ss->Ap = (*m_sparsityLRU)->Ap;
                m_hessianSparsity.H_ss->Ai = (*m_sparsityLRU)->Ai;
                m_hessianSparsity.H_ss->nz = (*m_sparsityLRU)->nz;

                m_hessianSparsity.finalize();

                changed = true;
            }
        }

        m_fullSparsityRebuildNeeded = false;

        // All terms' latest sparsity patterns are now incorporated.
        for (size_t i = 0; i < numTerms(); ++i)
            term(i).sparsityPatternChanged.clear();

        return changed;
    }

    virtual void m_termsAddedOrRemoved() override {
        m_fullSparsityRebuildNeeded = true;
    }

    virtual NewtonHessian m_getHessianSparsityPattern() const override { return m_hessianSparsity; }

    virtual void m_evalHessian(NewtonHessian &result, bool projectionMask) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonMultiobjectiveProblem.hessian");

        result.setZero();
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
            term(ti).newtonIterationBegan();
        }

        m_vars->updateParametrization();

        if (m_customCallback) return m_customCallback(*this, i);
        return false; // don't exit early
    }
};

#endif /* end of include guard: MULTIOBJECTIVEPROBLEM_HH */
