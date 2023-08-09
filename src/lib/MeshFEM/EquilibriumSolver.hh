////////////////////////////////////////////////////////////////////////////////
// EquilibriumSolver.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Compute the static equilibrium of a conservative system (whose elastic
//  energy term derives from the `ElasticObject` base class)
//  by minimizing its total potential energy with a Newton-based solver.
//
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/31/2020 20:07:51
*///////////////////////////////////////////////////////////////////////////////
#ifndef EQUILIBRIUMSOLVER_HH
#define EQUILIBRIUMSOLVER_HH
#include "newton_optimizer/newton_optimizer.hh"
#include "Loads/Load.hh"
#include "ElasticObject.hh"

#include <memory>
#include <functional>
#include <utility>

// Callback to be called like `bool earlyExit = cb(problem, iter)`
using CallbackFunction = std::function<bool(NewtonProblem &, size_t)>;

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
template<typename _Real>
using LoadCollection = std::vector<std::shared_ptr<Loads::Load<_Real>>>;

template<typename _Real>
struct EquilibriumProblem : public NewtonProblem {
    using Real = _Real;
    using LC   = LoadCollection<Real>;
    using EO   = ElasticObject<Real>;

    EquilibriumProblem(ElasticObject<Real> &eo, const LC &lc = LC())
        : m_obj(eo), m_loads(lc),
          m_hessianSparsity(eo.hessianSparsityPattern()),
          m_characteristicLength(eo.characteristicLength()) {
        for (const auto &l : m_loads)
            m_hessianSparsity.addWithDistinctSparsityPattern(l->hessianSparsityPattern(1.0));
        m_hessianSparsity.fill(1.0);
        // TODO (Samara): Initialize bound constraints. 
    }

    virtual void setVars(const VXd &vars) override {
        m_obj.setVars(vars.cast<Real>());
    }
    virtual const VXd getVars() const override { return m_obj.getVars().template cast<double>(); }
    virtual size_t numVars() const override { return m_obj.numVars(); }

    virtual Real energy() const override {
        Real result = m_obj.energy();
        if (result > systemEnergyIncreaseFactorLimit * std::max(m_currSystemEnergy, energyLimitingThreshold))
             return safe_numeric_limits<Real>::max();
        for (const auto &l : m_loads)
            result += l->energy();
        return result;
    }

    virtual VXd gradient(bool freshIterate = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("EquilibriumProblem.gradient");
        auto result = m_obj.gradient(freshIterate);
        for (const auto &l : m_loads)
            result += l->grad_x();

        return result.template cast<double>();
    }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

    // Note: we can modify the settings of each load through this method, but
    // not add/remove loads since this would alter the Hessian sparsity pattern.
    const LC &loads() { return m_loads; }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Useful for determining reasonable step lengths to take when the Newton step is not possible.)
    virtual Real characteristicDistance(const Eigen::VectorXd &d) const override {
        return m_characteristicLength / m_obj.approxLinfVelocity(d);
    }

    // The maximum factor by which we allow the elastic energy to increase in a single
    // Newton iteration; limiting this prevents large deployment forces from
    // severly deforming the umbrella mesh into a bad configuration.
    Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max();
    Real energyLimitingThreshold = 1e-6;

protected:
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool projectionMask) const override {
        result.data().setZero();
        m_obj.hessian(result, projectionMask);
        for (const auto &l : m_loads)
            l->hessian(result, projectionMask);
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("EquilibriumSolver.m_evalMetric");
        m_obj.massMatrix(result, true, true);
    }

    virtual bool m_iterationCallback(size_t i) override {
        if (systemEnergyIncreaseFactorLimit == safe_numeric_limits<Real>::max())
            m_currSystemEnergy = safe_numeric_limits<Real>::max();
        else m_currSystemEnergy = m_obj.energy();
        m_obj.updateParametrization();
        if (m_customCallback) return m_customCallback(*this, i);
        return false; // don't exit early
    }

    CallbackFunction m_customCallback;

    EO &m_obj;
    LC m_loads;

    mutable SuiteSparseMatrix m_hessianSparsity;

    Real m_characteristicLength = 1.0;
    Real m_currSystemEnergy = safe_numeric_limits<Real>::max();
};

template<typename _Real>
std::unique_ptr<NewtonOptimizer> get_equilibrium_optimizer(ElasticObject<_Real> &obj, const LoadCollection<_Real> &loads,
                                                           const std::vector<size_t> &fixedVars,
                                                           const NewtonOptimizerOptions &opts, CallbackFunction customCallback,
                                                           Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max(), Real energyLimitingThreshold = 1e-6,
                                                           Real hessianShift = 0.0) {
    auto problem = std::make_unique<EquilibriumProblem<_Real>>(obj, loads);
    problem->addFixedVariables(fixedVars);
    problem->setCustomIterationCallback(customCallback);
    problem->systemEnergyIncreaseFactorLimit = systemEnergyIncreaseFactorLimit;
    problem->energyLimitingThreshold = energyLimitingThreshold;
    problem->hessianShift = hessianShift;
    auto opt = std::make_unique<NewtonOptimizer>(std::move(problem));
    opt->options = opts;
    return opt;
}

template<typename _Real>
ConvergenceReport equilibrium_newton(ElasticObject<_Real> &obj, const LoadCollection<_Real> &loads,
                                     const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction customCallback, Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max(), Real energyLimitingThreshold = 1e-6, Real hessianShift = 0.0) {
    return get_equilibrium_optimizer(obj, loads, fixedVars, opts, customCallback, systemEnergyIncreaseFactorLimit, energyLimitingThreshold, hessianShift)->optimize();
}

#endif /* end of include guard: EQUILIBRIUMSOLVER_HH */
