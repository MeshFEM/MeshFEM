#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>

PYBIND11_MODULE(py_newton_optimizer, m) {
    m.doc() = "Wrapper for Newton optimizer's types";

    ////////////////////////////////////////////////////////////////////////////////
    // Newton solver options/convergence report
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<NewtonOptimizerOptions>(m, "NewtonOptimizerOptions")
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
        ;

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
        .def("hessian",                &NewtonProblem::hessian)
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
