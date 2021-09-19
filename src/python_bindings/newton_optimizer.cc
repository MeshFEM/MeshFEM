#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include "BindingUtils.hh"

template<class DerivedController, class BaseController>
auto bindController(py::module &m, const char *name) {
    py::class_<DerivedController, BaseController, std::shared_ptr<DerivedController>> binding(m, name);
    binding.def(py::init<>());
    addSerializationBindings<DerivedController>(binding);

    return binding;
}

PYBIND11_MODULE(py_newton_optimizer, m) {
    m.doc() = "Wrapper for Newton optimizer's types";

    ////////////////////////////////////////////////////////////////////////////////
    // "Controllers" for customizing solver behavior
    // (accessed through NewtonOptimizerOptions)
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<HessianProjectionController, std::shared_ptr<HessianProjectionController>> pyHPC(m, "HessianProjectionController");
    pyHPC.def("shouldUseProjection", &HessianProjectionController::shouldUseProjection)
         .def("notifyDefiniteness",  &HessianProjectionController::notifyDefiniteness,  py::arg("isIndefinite"))
        ;

    bindController<HessianProjectionNever,    HessianProjectionController>(m, "HessianProjectionNever"   );
    bindController<HessianProjectionAlways,   HessianProjectionController>(m, "HessianProjectionAlways"  );
    bindController<HessianProjectionAdaptive, HessianProjectionController>(m, "HessianProjectionAdaptive")
        .def_readwrite("numProjectionStepsBeforeSwitch",            &HessianProjectionAdaptive::numProjectionStepsBeforeSwitch,            "Number of Hessian-projected steps to take before trying un-projected Hessian")
        .def_readwrite("numConsecutiveIndefiniteStepsBeforeSwitch", &HessianProjectionAdaptive::numConsecutiveIndefiniteStepsBeforeSwitch, "Number of indefinite Hessians to allow before switching to applying the Hessian projection")
        .def_readwrite("projectionActive",                          &HessianProjectionAdaptive::projectionActive,                          "(internal state for switching logic)")
        .def_readwrite("switchCounter",                             &HessianProjectionAdaptive::projectionActive,                          "(internal state for switching logic)")
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
        .def_readwrite("stdoutFlushInterval",           &NewtonOptimizerOptions::stdoutFlushInterval)
        .def_readwrite("nbacktrack_iter",               &NewtonOptimizerOptions::nbacktrack_iter)
        .def_readwrite("ngd_fallback_steps",            &NewtonOptimizerOptions::ngd_fallback_steps)
        .def_property("hessianProjectionController", [](const NewtonOptimizerOptions &opts) -> HessianProjectionController & { return opts.getHessianProjectionController(); },
                                                     [](      NewtonOptimizerOptions &opts, const HessianProjectionController &h) { opts.setHessianProjectionController(h); },
                                                     py::return_value_policy::reference)
        .def_property("hessianUpdateController",     [](const NewtonOptimizerOptions &opts) -> HessianUpdateController & { return opts.getHessianUpdateController(); },
                                                     [](      NewtonOptimizerOptions &opts, const HessianUpdateController &h) { opts.setHessianUpdateController(h); },
                                                     py::return_value_policy::reference)
        ;
    addSerializationBindings<NewtonOptimizerOptions, PyNOO, NewtonOptimizerOptions::StateBackwardCompat>(pyNewtonOptimizerOptions);

    py::class_<ConvergenceReport>(m, "ConvergenceReport")
        .def_readonly("success",          &ConvergenceReport::success)
        .def         ("numIters",         &ConvergenceReport::numIters)
        .def_readonly("energy",           &ConvergenceReport::energy)
        .def_readonly("gradientNorm",     &ConvergenceReport::gradientNorm)
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

    py::class_<NewtonProblem>(m, "NewtonProblem")
        .def("energy",                 &NewtonProblem::energy)
        .def("gradient",               &NewtonProblem::gradient, py::arg("freshIterate") = false)
        .def("hessian",                &NewtonProblem::hessian,  py::arg("projectionMask") = true)
        .def("hessianSparsityPattern", &NewtonProblem::hessianSparsityPattern)
        .def("metric",                 &NewtonProblem::metric)
        .def("fixedVars",              &NewtonProblem::fixedVars)
        .def("addFixedVariables",      &NewtonProblem::addFixedVariables)
        .def("setFixedVars",           &NewtonProblem::setFixedVars)
        .def("getVars",                &NewtonProblem::getVars)
        .def("setVars",                &NewtonProblem::setVars)
        .def("numVars",                &NewtonProblem::numVars)
        .def("applyBoundConstraints",  &NewtonProblem::applyBoundConstraints)
        .def("activeBoundConstraints", &NewtonProblem::activeBoundConstraints)
        .def("boundConstraints",       &NewtonProblem::boundConstraints, py::return_value_policy::reference)
        .def("feasible",               &NewtonProblem::feasible)
        .def("feasibleStepLength",     py::overload_cast<const Eigen::VectorXd &>(&NewtonProblem::feasibleStepLength, py::const_))
        .def("iterationCallback",      &NewtonProblem::iterationCallback)
        .def_readwrite("disableCaching", &NewtonProblem::disableCaching)
        ;

    py::class_<WorkingSet>(m, "WorkingSet")
        .def(py::init<NewtonProblem &>())
        .def("contains", &WorkingSet::contains)
        .def("fixesVariable", &WorkingSet::fixesVariable)
        .def("size", &WorkingSet::size)
        .def("getFreeComponent", &WorkingSet::getFreeComponent)
        ;

    py::class_<NewtonOptimizer>(m, "NewtonOptimizer")
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

                opt.newton_step(step, prob.gradient(false), workingSet, beta, betaMin, feasibility);
                return step;
            }, py::arg("feasibility") = false)
        .def("get_problem", py::overload_cast<>(&NewtonOptimizer::get_problem), py::return_value_policy::reference)
        .def("setFixedVars", &NewtonOptimizer::setFixedVars, py::arg("fixedVars"))
        .def_readwrite("options", &NewtonOptimizer::options)
        ;
}
