#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11; // NOLINT (workaround clang-tidy bug)

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>
#include "BindingUtils.hh"

#include "CallbackWrapper.hh"

template<class DerivedController, class BaseController>
auto bindController(py::module &m, const char *name) {
    py::class_<DerivedController, BaseController, std::shared_ptr<DerivedController>> binding(m, name);
    binding.def(py::init<>());
    addSerializationBindings<DerivedController>(binding);

    return binding;
}

PYBIND11_MODULE(py_newton_optimizer, m) {
    m.doc() = "Wrapper for Newton optimizer's types";

    py::module::import("sparse_matrices");
    py::module::import("block_sparse_hessian");

    ////////////////////////////////////////////////////////////////////////////////
    // "Controllers" for customizing solver behavior
    // (accessed through NewtonOptimizerOptions)
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<HessianProjectionController, std::shared_ptr<HessianProjectionController>> pyHPC(m, "HessianProjectionController");
    pyHPC.def("shouldUseProjection", &HessianProjectionController::shouldUseProjection)
         .def("notifyDefiniteness",  &HessianProjectionController::notifyDefiniteness,  py::arg("isIndefinite"))
         .def("notifyStep",          &HessianProjectionController::notifyStep,          py::arg("step"))
         .def("notifyDirectionalDerivative", &HessianProjectionController::notifyDirectionalDerivative, py::arg("directionalDerivative"))
         .def("reset", &HessianProjectionController::reset, "Reset the controller to its initial state (e.g., automatically called at the start of each Newton optimization).")
        ;

    bindController<HessianProjectionNever,    HessianProjectionController>(m, "HessianProjectionNever"   );
    bindController<HessianProjectionAlways,   HessianProjectionController>(m, "HessianProjectionAlways"  );
    bindController<HessianProjectionAdaptive, HessianProjectionController>(m, "HessianProjectionAdaptive")
        .def_readwrite("numProjectionStepsBeforeDisable",           &HessianProjectionAdaptive::numProjectionStepsBeforeDisable,           "Number of Hessian-projected steps to take before trying un-projected Hessian")
        .def_readwrite("numConsecutiveIndefiniteStepsBeforeEnable", &HessianProjectionAdaptive::numConsecutiveIndefiniteStepsBeforeEnable, "Number of indefinite Hessians to allow before switching to applying the Hessian projection")
        .def_readwrite("stepLengthThresholdForDisable",             &HessianProjectionAdaptive::stepLengthThresholdForDisable,             "Disable projection if step length falls below this threshold")
        .def_readwrite("directionalDerivativeThresholdForDisable",  &HessianProjectionAdaptive::directionalDerivativeThresholdForDisable,  "Disable projection if directional derivative exceeds (becomes less negative than) this threshold")
        .def_readwrite("projectionActive",                          &HessianProjectionAdaptive::projectionActive,                          "(internal state for switching logic)")
        .def_readwrite("switchCounter",                             &HessianProjectionAdaptive::projectionActive,                          "(internal state for switching logic)")
        .def_readwrite("startWithProjectionActive",                 &HessianProjectionAdaptive::startWithProjectionActive,                 "Whether to start the optimization with projection active")
        ;

    py::class_<HessianUpdateController, std::shared_ptr<HessianUpdateController>>(m, "HessianUpdateController")
        .def("needsUpdate", &HessianUpdateController::needsUpdate)
        .def("newHessian",  &HessianUpdateController::newHessian,  py::arg("isIndefinite"))
        .def("reusedHessian",  &HessianUpdateController::reusedHessian)
        ;

    bindController<HessianUpdateNever,    HessianUpdateController>(m, "HessianUpdateNever"   );
    bindController<HessianUpdateAlways,   HessianUpdateController>(m, "HessianUpdateAlways"  );
    bindController<HessianUpdatePeriodic, HessianUpdateController>(m, "HessianUpdatePeriodic")
        .def_readwrite("period", &HessianUpdatePeriodic::period, "Number of times to reuse a Hessian factorization before computing a new one.")
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Newton solver options/convergence report
    ////////////////////////////////////////////////////////////////////////////////
    using PyNOO = py::class_<NewtonOptimizerOptions, std::shared_ptr<NewtonOptimizerOptions>>;
    PyNOO pyNewtonOptimizerOptions(m, "NewtonOptimizerOptions");
    pyNewtonOptimizerOptions
        .def(py::init<>())
        .def_readwrite("gradTol",                       &NewtonOptimizerOptions::gradTol)
        .def_readwrite("beta",                          &NewtonOptimizerOptions::beta)
        .def_readwrite("hessianScaledBeta",             &NewtonOptimizerOptions::hessianScaledBeta)
        .def_readwrite("niter",                         &NewtonOptimizerOptions::niter)
        .def_readwrite("useIdentityMetric",             &NewtonOptimizerOptions::useIdentityMetric)
        .def_readwrite("useNegativeCurvatureDirection", &NewtonOptimizerOptions::useNegativeCurvatureDirection)
        .def_readwrite("feasibilitySolve",              &NewtonOptimizerOptions::feasibilitySolve)
        .def_readwrite("verbose",                       &NewtonOptimizerOptions::verbose)
        .def_readwrite("verboseNonPosDef",              &NewtonOptimizerOptions::verboseNonPosDef)
        .def_readwrite("verboseWorkingSet",             &NewtonOptimizerOptions::verboseWorkingSet)
        .def_readwrite("stdoutFlushInterval",           &NewtonOptimizerOptions::stdoutFlushInterval)
        .def_readwrite("nbacktrack_iter",               &NewtonOptimizerOptions::nbacktrack_iter)
        .def_readwrite("ngd_fallback_steps",            &NewtonOptimizerOptions::ngd_fallback_steps)
        .def_readwrite("armijo_c1",                     &NewtonOptimizerOptions::armijo_c1)
        .def_readwrite("backtrack_shrink_factor",       &NewtonOptimizerOptions::backtrack_shrink_factor)
        .def_readwrite("factorizer",                    &NewtonOptimizerOptions::factorizer)
        .def_property("hessianProjectionController", [](const NewtonOptimizerOptions &opts) -> HessianProjectionController & { return opts.getHessianProjectionController(); },
                                                     [](      NewtonOptimizerOptions &opts, const HessianProjectionController &h) { opts.setHessianProjectionController(h); },
                                                     py::return_value_policy::reference_internal)
        .def_property("hessianUpdateController",     [](const NewtonOptimizerOptions &opts) -> HessianUpdateController & { return opts.getHessianUpdateController(); },
                                                     [](      NewtonOptimizerOptions &opts, const HessianUpdateController &h) { opts.setHessianUpdateController(h); },
                                                     py::return_value_policy::reference_internal)
        ;
    addSerializationBindings<NewtonOptimizerOptions, PyNOO, NewtonOptimizerOptions::StateBackwardCompat, NewtonOptimizerOptions::StateBackwardCompat2>(pyNewtonOptimizerOptions);

    py::class_<ConvergenceReport>(m, "ConvergenceReport")
        .def_readonly("success",          &ConvergenceReport::success)
        .def         ("numIters",         &ConvergenceReport::numIters)
        .def_readonly("energy",           &ConvergenceReport::energy)
        .def_readonly("freeGradientNorm", &ConvergenceReport::freeGradientNorm)
        .def_readonly("stepLength",       &ConvergenceReport::stepLength)
        .def_readonly("indefinite",       &ConvergenceReport::indefinite)
        .def_readonly("customData",       &ConvergenceReport::customData)
        ;

    using BC = NewtonProblem::BoundConstraint;
    py::class_<NewtonProblem::BoundConstraint>(m, "BoundConstraint")
        .def_readwrite("idx",      &BC::idx)
        .def_readwrite("val",      &BC::val)
        .def_readwrite("type",     &BC::type)
        .def("active",             &BC::active,             py::arg("vars"), py::arg("g"), py::arg("tol") = 1e-8)
        .def("feasible",           &BC::feasible,           py::arg("vars"))
        .def("apply",              &BC::apply,              py::arg("vars"))
        .def("feasibleStepLength", &BC::feasibleStepLength, py::arg("vars"), py::arg("step"))
        ;

    py::class_<NewtonProblem, std::shared_ptr<NewtonProblem>>(m, "NewtonProblem")
        .def("energy",                 &NewtonProblem::objective)
        .def("objective",              &NewtonProblem::objective)
        .def("gradient",               &NewtonProblem::gradient, py::arg("freshIterate") = false)
        .def("hessian",                &NewtonProblem::hessian,  py::arg("projectionMask") = false)
        .def("hessianSparsityPattern", &NewtonProblem::hessianSparsityPattern, py::arg("needsUpdate") = true)
        .def("metric",                 &NewtonProblem::metric)
        .def("fixedVars",              &NewtonProblem::fixedVars)
        .def("addFixedVariables",      &NewtonProblem::addFixedVariables)
        .def("setFixedVars",           &NewtonProblem::setFixedVars)
        .def("getVars",                &NewtonProblem::getVars)
        .def("setVars",                &NewtonProblem::setVars)
        .def("numVars",                &NewtonProblem::numVars)
        .def("applyBoundConstraints",  &NewtonProblem::applyBoundConstraints)
        .def("activeBoundConstraints", &NewtonProblem::activeBoundConstraints, py::arg("vars"), py::arg("g"), py::arg("tol") = 1e-8)
        .def("boundConstraints",       &NewtonProblem::boundConstraints, py::return_value_policy::reference_internal)
        .def("feasible",               &NewtonProblem::feasible)
        .def("feasibleStepLength",     py::overload_cast<const Eigen::VectorXd &>(&NewtonProblem::feasibleStepLength, py::const_))
        .def("characteristicDistance", &NewtonProblem::characteristicDistance, py::arg("d"))

        .def("customFeasibleStepLength", &NewtonProblem::customFeasibleStepLength, py::arg("vars"), py::arg("step"))
        .def("lineSearchTerminated",     &NewtonProblem::lineSearchTerminated, "Notify the problem that a line search has terminated (e.g., called from NewtonOptimizer::optimize).")

        .def_readwrite("hessianShift",            &NewtonProblem::hessianShift)
        .def_readwrite("useRelativeHessianShift", &NewtonProblem::useRelativeHessianShift)

        .def_property_readonly("hessianWasProjected",             &NewtonProblem::hessianWasProjected,             "Whether a projected Hessian was requested in the last call to `hessian()`")
        .def_property_readonly("lastFactorizationShiftMagnitude", &NewtonProblem::lastFactorizationShiftMagnitude, "The last `tau` parameter that was used to make the Hessian positive definite during newton_step")

        .def("optimizer", [](std::shared_ptr<NewtonProblem> prob) { return std::make_shared<NewtonOptimizer>(prob); })

        .def_readwrite("disableCaching", &NewtonProblem::disableCaching)
        .def("invalidateCachedHessian",  &NewtonProblem::invalidateCachedHessian)

        .def("setCustomLineSearchBeganCallback",
                [](NewtonProblem &prob, const GenericPyCallbackFunction<NewtonProblem, void, const Eigen::VectorXd &, double> &pcb) {
                    prob.setCustomLineSearchBeganCallback(callbackWrapper<NewtonProblem, void, const Eigen::VectorXd &, double>(pcb));
                }, py::arg("cb"))
        ;

    py::class_<WorkingSet>(m, "WorkingSet")
        .def(py::init<NewtonProblem &>())
        .def("contains", &WorkingSet::contains)
        .def("fixesVariable", &WorkingSet::fixesVariable)
        .def("size", &WorkingSet::size)
        .def("getFreeComponent", &WorkingSet::getFreeComponent)
        ;

    using NVB = NewtonVarsBase;
    py::class_<NVB, std::shared_ptr<NVB>>(m, "NewonVars")
        .def("getVars", &NVB::getVars)
        .def("setVars", &NVB::setVars)
        .def("numVars", &NVB::numVars)
        .def("updateParametrization", &NVB::updateParametrization)

        .def("characteristicLength",  &NVB::characteristicLength)
        .def("approxLinfVelocity",    &NVB::approxLinfVelocity, py::arg("d"))
        ;

    py::class_<ObjectiveIncreaseLimiter, std::shared_ptr<ObjectiveIncreaseLimiter>>(m, "ObjectiveIncreaseLimiter")
        .def_readwrite("factor",        &ObjectiveIncreaseLimiter::factor)
        .def_readwrite("threshold",     &ObjectiveIncreaseLimiter::threshold)
        .def_readwrite("previousValue", &ObjectiveIncreaseLimiter::previousValue)
        .def("valueExceedsLimit",       &ObjectiveIncreaseLimiter::valueExceedsLimit)
        ;

    using NOT = NewtonObjectiveTermBase;

    py::enum_<NOT::SparsityUpdateFrequency>(m, "SparsityUpdateFrequency")
        .value("NEVER",     NOT::SparsityUpdateFrequency::NEVER)
        .value("ALWAYS",    NOT::SparsityUpdateFrequency::ALWAYS)
        .value("SOMETIMES", NOT::SparsityUpdateFrequency::SOMETIMES)
        ;

    py::class_<NOT, std::shared_ptr<NOT>>(m, "NewtonObjectiveTerm")
        .def("objective", &NOT::objective)
        .def("gradient",  &NOT::gradient, py::arg("weight") = 1.0, py::arg("freshIterate") = false)
        .def("hessian",   &NOT::hessian, py::arg("projectionMask") = false)
        .def("hessianSparsityPattern", &NOT::hessianSparsityPattern)
        .def_readwrite("suppressSparsity", &NOT::suppressSparsity, "Suppress sparsity pattern contributions from this term")
        .def_property_readonly("sparsityUpdateFrequency", &NOT::sparsityUpdateFrequency)
        .def_readonly("increaseLimiter", &NOT::increaseLimiter, py::return_value_policy::reference_internal)

        .def("objectiveAtVars", &NOT::objectiveAtVars, py::arg("x"))
        ;

    py::class_<FeasibleStepLengthComputer, std::shared_ptr<FeasibleStepLengthComputer>>(m, "FeasibleStepLengthComputer")
        .def("eval", &FeasibleStepLengthComputer::eval, py::arg("vars"), py::arg("step"),
             "Evaluate the feasible step length for the given variables and step. "
             "Returns a positive value if the step is feasible, or a negative value if it is not.")
        ;

    py::class_<NewtonMultiobjectiveProblem, NewtonProblem, std::shared_ptr<NewtonMultiobjectiveProblem>>(m, "NewtonMultiobjectiveProblem")
        .def(py::init<std::shared_ptr<NVB>, std::vector<std::shared_ptr<NOT>>>(), py::arg("vars"), py::arg("terms"))
        .def("numTerms",   &NewtonMultiobjectiveProblem::numTerms)
        .def("setTerms",   &NewtonMultiobjectiveProblem::setTerms)
        .def("setWeights", &NewtonMultiobjectiveProblem::setWeights)
        .def("getWeights", &NewtonMultiobjectiveProblem::getWeights)

        .def("setTermNames", &NewtonMultiobjectiveProblem::setTermNames, py::arg("names"))
        .def("getTermNames", &NewtonMultiobjectiveProblem::getTermNames)
        .def("termName",     &NewtonMultiobjectiveProblem::termName, py::arg("idx"))

        .def("weight",       [](NewtonMultiobjectiveProblem &prob,                size_t i) { return prob.weight(i);    })
        .def("weight",       [](NewtonMultiobjectiveProblem &prob, const std::string &name) { return prob.weight(name); })
        .def("term",       [](NewtonMultiobjectiveProblem &prob,                size_t i) -> NOT & { return prob.term(i);    }, py::return_value_policy::reference_internal)
        .def("term",       [](NewtonMultiobjectiveProblem &prob, const std::string &name) -> NOT & { return prob.term(name); }, py::return_value_policy::reference_internal)
        .def_property_readonly("terms", &NewtonMultiobjectiveProblem::getTerms)
        .def_property_readonly("sparsityLRU", &NewtonMultiobjectiveProblem::sparsityLRUPtr)

        .def("termObjectives", &NewtonMultiobjectiveProblem::termObjectives)
        .def("termGradients",  &NewtonMultiobjectiveProblem::termGradients)

        .def("objectiveAtVars", &NewtonMultiobjectiveProblem::objectiveAtVars, py::arg("x"))

        .def_readwrite("initialFeasibleStepLengthComputer",
                &NewtonMultiobjectiveProblem::initialFeasibleStepLengthComputer,
                "A user-defined functor that computes an initial upper bound for the feasible step length before each term is queried for its own feasible step length. An example use case is in injective parameterization, where we seek a step that prevents elements from inverting.")

        .def("setCustomIterationCallback",
                [](NewtonMultiobjectiveProblem &prob, const PyCallbackFunction<NewtonProblem> &pcb) {
                    prob.setCustomIterationCallback(callbackWrapper<NewtonProblem>(pcb));
                }, py::arg("cb"))
        ;

    py::class_<NewtonOptimizer, std::shared_ptr<NewtonOptimizer>>(m, "NewtonOptimizer")
        .def("optimize", [](NewtonOptimizer &nopt) {
                  py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
                  py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
                  return nopt.optimize();
              })
        // For debugging the Newton step. TODO: support nonempty working sets, different betas
        .def("newton_step", [](NewtonOptimizer &opt, const bool feasibility) {
                Eigen::VectorXd step;
                auto &prob = opt.get_problem();
                prob.setVars(prob.applyBoundConstraints(prob.getVars()));
                WorkingSet workingSet(prob);

                Real beta = opt.options.beta;
                const Real betaMin = std::min(beta, 1e-6); // Initial shift "tau" to use when an indefinite matrix is detected.

                opt.newton_step(step, -prob.gradient(false), workingSet, beta, betaMin, feasibility);
                return step;
            }, py::arg("feasibility") = false)
        .def("get_problem", py::overload_cast<>(&NewtonOptimizer::get_problem), py::return_value_policy::reference_internal)

        .def("update_factorizations", [](NewtonOptimizer &opt) { opt.update_factorizations(); })
        .def_property_readonly("hessian_factorization", [](NewtonOptimizer &opt) -> NewtonHessianFactorization & { return opt.hessianFactorization(); }, py::return_value_policy::reference_internal)

        .def_readwrite("options", &NewtonOptimizer::options)
        ;
}
