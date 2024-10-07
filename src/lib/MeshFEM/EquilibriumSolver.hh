////////////////////////////////////////////////////////////////////////////////
// EquilibriumSolver.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Compute the static equilibrium of a conservative system (whose elastic
//  energy term derives from the `ElasticObject` base class)
//  by minimizing its total potential energy with a Newton-based solver.
//
//  This is done via a small number of convenience functions that wrap the
//  `NewtonMultiobjectiveProblem` interface with equilibrium-specific
//  arguemnt names.
//
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/31/2020 20:07:51
*///////////////////////////////////////////////////////////////////////////////
#ifndef EQUILIBRIUMSOLVER_HH
#define EQUILIBRIUMSOLVER_HH
#include "newton_optimizer/newton_optimizer.hh"
#include "newton_optimizer/MultiobjectiveProblem.hh"
#include "Loads/Load.hh"
#include "ElasticObject.hh"

#include <memory>
#include <utility>

template<typename _Real>
using LoadCollection = std::vector<std::shared_ptr<Loads::Load<_Real>>>;

template<typename _Real>
std::unique_ptr<NewtonMultiobjectiveProblem> get_equilibrium_problem(std::shared_ptr<ElasticObject<_Real>> obj, const LoadCollection<_Real> &loads) {
    // `obj` plays two roles: `NewtonVarsBase` and `NewtonObjectiveTermBase`
    std::vector<std::shared_ptr<NewtonObjectiveTermBase>> terms;
    terms.push_back(obj);
    terms.insert(terms.end(), loads.begin(), loads.end());
    return std::make_unique<NewtonMultiobjectiveProblem>(obj, terms);
}

template<typename _Real>
std::unique_ptr<NewtonOptimizer> get_equilibrium_optimizer(std::shared_ptr<ElasticObject<_Real>> obj, const LoadCollection<_Real> &loads,
                                                           const std::vector<size_t> &fixedVars,
                                                           const NewtonOptimizerOptions &opts, NewtonMultiobjectiveProblem::CallbackFunction customCallback,
                                                           Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max(), Real energyLimitingThreshold = 1e-6,
                                                           Real hessianShift = 0.0) {
    auto problem = get_equilibrium_problem(obj, loads);
    problem->addFixedVariables(fixedVars);
    problem->setCustomIterationCallback(customCallback);
    problem->term(0).increaseLimiter.factor = systemEnergyIncreaseFactorLimit;
    problem->term(0).increaseLimiter.threshold = energyLimitingThreshold;
    problem->hessianShift = hessianShift;
    auto opt = std::make_unique<NewtonOptimizer>(std::move(problem));
    opt->options = opts;
    return opt;
}

template<typename _Real>
ConvergenceReport equilibrium_newton(std::shared_ptr<ElasticObject<_Real>> obj, const LoadCollection<_Real> &loads,
                                     const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, NewtonMultiobjectiveProblem::CallbackFunction customCallback, Real systemEnergyIncreaseFactorLimit = safe_numeric_limits<Real>::max(), Real energyLimitingThreshold = 1e-6, Real hessianShift = 0.0) {
    return get_equilibrium_optimizer(obj, loads, fixedVars, opts, customCallback, systemEnergyIncreaseFactorLimit, energyLimitingThreshold, hessianShift)->optimize();
}

#endif /* end of include guard: EQUILIBRIUMSOLVER_HH */
