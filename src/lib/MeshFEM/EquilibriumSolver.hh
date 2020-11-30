////////////////////////////////////////////////////////////////////////////////
// EquilibriumSolver.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Compute the static equilibrium of a conservative system by minimizing its
//  total potential energy with a Newton-based solver.
//
//  The system whose equilibrium we compute must be represented by
//  an object with the following methods:
//      setVars/getVars/numVars
//      energy
//      gradient
//      hessian/hessianSparsityPattern
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/31/2020 20:07:51
////////////////////////////////////////////////////////////////////////////////
#ifndef EQUILIBRIUMSOLVER_HH
#define EQUILIBRIUMSOLVER_HH
#include "newton_optimizer/newton_optimizer.hh"
#include "Loads/Load.hh"

#include <memory>
#include <functional>
#include <utility>

using CallbackFunction = std::function<void(NewtonProblem &, size_t)>;

////////////////////////////////////////////////////////////////////////////////
// "Guarded" implementation calls:
// There are optional parts of the ElasticObject interface to support certain
// objects whose gradient methods want to know if they are called on a "fresh
// iterate" (i.e., not within a line search) or who need up update their DoF
// definitions at the end of each line search.
// The following `guarded` calls only use these optional features when
// available. We use a hack to disambiguate the calls when more than
// one is valid (e.g., when the system provides both a `gradient()` and a
// `gradient(bool)`: we pass an integer as the last parameter which prefers
// the overload accepting an `int` but that will still match the one accepting
// a `long` if the `int` overload if invalid.
////////////////////////////////////////////////////////////////////////////////
template<class EQSystem> auto guardedGradientCall(const EQSystem &sys, bool freshIterate,        int     /* PREFERRED */) -> decltype(sys.gradient(freshIterate)) { return sys.gradient(freshIterate); }
template<class EQSystem> auto guardedGradientCall(const EQSystem &sys, bool /* freshIterate */, long /* NON-PREFERRED */) -> decltype(sys.gradient())             { return sys.gradient(); }

template<class EQSystem> auto guardedParametrizationUpdate(EQSystem &sys, int      /* PREFERRED */) -> decltype(sys.updateParametrization()) { return sys.updateParametrization(); }
template<class EQSystem> void guardedParametrizationUpdate(EQSystem &   , long /* NON-PREFERRED */) { /* NOP */ } 

template<class EQSystem>
using LoadCollection = std::vector<std::shared_ptr<Loads::Load<EQSystem::N, typename EQSystem::Real>>>;

template<class EQSystem>
struct EquilibriumProblem : public NewtonProblem {
    static constexpr size_t N = EQSystem::N;
    using Real = typename EQSystem::Real;
    using LC = LoadCollection<EQSystem>;

    EquilibriumProblem(EQSystem &sys, const LC &lc = LC())
        : m_sys(sys), m_loads(lc),
          m_hessianSparsity(sys.hessianSparsityPattern()) {
        for (const auto &l : m_loads)
            m_hessianSparsity.addWithDistinctSparsityPattern(l->hessianSparsityPattern(1.0));
        m_hessianSparsity.fill(1.0);
    }

    virtual void setVars(const VXd &vars) override {
        m_sys.setVars(vars.cast<typename EQSystem::Real>());
    }
    virtual const VXd getVars() const override { return m_sys.getVars().template cast<double>(); }
    virtual size_t numVars() const override { return m_sys.numVars(); }

    virtual Real energy() const override {
        Real result = m_sys.energy();
        for (const auto &l : m_loads)
            result += l->energy();
        return result;
    }

    virtual VXd gradient(bool freshIterate = false) const override {
        auto result = guardedGradientCall(m_sys, freshIterate, 0/* disambiguation hack to ensure`freshIterate` is passed when possible */);
        for (const auto &l : m_loads)
            result += l->grad_x();

        return result.template cast<double>();
    }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

    // Note: we can modify the settings of each load through this method, but
    // not add/remove loads since this would alter the Hessian sparsity pattern.
    const LC &loads() { return m_loads; }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }

protected:
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool projectionMask) const override {
        result.setZero();
        m_sys.hessian(result, projectionMask);
        for (const auto &l : m_loads)
            l->hessian(result, projectionMask);
    }
    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        // TODO: mass matrix?
        result.setIdentity(true);
    }

    virtual void m_iterationCallback(size_t i) override {
        guardedParametrizationUpdate(m_sys, 0 /* disambiguation hack to ensure the `updateParametrization` call is made when it is available */);
        if (m_customCallback) m_customCallback(*this, i);
    }

    CallbackFunction m_customCallback;

    EQSystem &m_sys;
    LC m_loads;

    mutable SuiteSparseMatrix m_hessianSparsity;
};

template<class EQSys>
std::unique_ptr<NewtonOptimizer> get_equilibrium_optimizer(EQSys &sys, const LoadCollection<EQSys> &loads,
                                                           const std::vector<size_t> &fixedVars,
                                                           const NewtonOptimizerOptions &opts, CallbackFunction customCallback) {
    auto problem = std::make_unique<EquilibriumProblem<EQSys>>(sys, loads);
    problem->addFixedVariables(fixedVars);
    problem->setCustomIterationCallback(customCallback);
    auto opt = std::make_unique<NewtonOptimizer>(std::move(problem));
    opt->options = opts;
    return opt;
}

template<class EQSys>
ConvergenceReport equilibrium_newton(EQSys &sys, const LoadCollection<EQSys> &loads,
                                     const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction customCallback) {
    return get_equilibrium_optimizer(sys, loads, fixedVars, opts, customCallback)->optimize();
}

#endif /* end of include guard: EQUILIBRIUMSOLVER_HH */
