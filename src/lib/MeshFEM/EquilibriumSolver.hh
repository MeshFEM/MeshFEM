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
using CallbackFunction = std::function<void(NewtonProblem &, size_t)>;

// Some systems want to know whether the gradient is being evaluated at a
// "fresh iterate" (i.e., not within a line search) while others do not accept
// this information...
template<class EQSystem, std::enable_if_t<std::is_integral<typename function_traits<decltype(&EQSystem::gradient)>::template arg<0>>::value, int> = 0>
auto guardedGradientCall(const EQSystem &sys, bool freshIterate) -> decltype(sys.gradient(freshIterate)) {
    return sys.gradient(freshIterate);
}

// No arguments
template<class EQSystem, std::enable_if_t<function_traits<decltype(&EQSystem::gradient)>::arity == 0, int> = 0>
auto guardedGradientCall(const EQSystem &sys, bool /* freshIterate */) -> decltype(sys.gradient()) {
    return sys.gradient();
}

// First argument exists but is not boolean (doesn't look like it accepts a freshIterate flag).
template<class EQSystem, std::enable_if_t<(function_traits<decltype(&EQSystem::gradient)>::arity > 0) && (!std::is_integral<typename function_traits<decltype(&EQSystem::gradient)>::template arg<0>>::value), int> = 0>
auto guardedGradientCall(const EQSystem &sys, bool /* freshIterate */) -> decltype(sys.gradient()) {
    return sys.gradient();
}

template<class EQSystem>
void guardedParametrizationUpdate(EQSystem &/* sys */, long /* NON-PREFERRED */) { /* NOP */ }

template<class EQSystem>
auto guardedParametrizationUpdate(EQSystem &sys, int /* PREFERRED */) -> decltype(sys.updateParametrization()) {
    return sys.updateParametrization();
}

template<class EQSystem>
using LoadCollection = std::vector<std::shared_ptr<Loads::Load<EQSystem::N, typename EQSystem::Real>>>;

template<class EQSystem>
struct EquilibriumProblem : public NewtonProblem {
    static constexpr size_t N = EQSystem::N;
    using Real = typename EQSystem::Real;
    using LC = LoadCollection<EQSystem>;

    EquilibriumProblem(EQSystem &sys, const LC &l = LC())
        : m_sys(sys), m_loads(l),
          m_hessianSparsity(sys.hessianSparsityPattern()) {
        for (const auto &l : m_loads)
            m_hessianSparsity.addWithDistinctSparsityPattern(l->hessianSparsityPattern(1.0));
        m_hessianSparsity.fill(1.0);
    }

    virtual void setVars(const VXd &vars) override {
        m_sys.setVars(vars.cast<typename EQSystem::Real>());
        for (auto &l : m_loads)
            l->deformedStateUpdated();
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
        auto result = guardedGradientCall(m_sys, freshIterate);
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
    virtual void m_evalHessian(SuiteSparseMatrix &result) const override {
        result.setZero();
        m_sys.hessian(result);
        for (const auto &l : m_loads)
            l->hessian(result);
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
