#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (workaround clang-tidy bug)

#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/EquilibriumSolver.hh>
#include <MeshFEM/DynamicSimulator.hh>

#include "CallbackWrapper.hh"

using namespace MeshFEM;

template<typename Real_>
void bind(py::module &m, py::module &detail_module) {
    using EO = ElasticObject<Real_>;
    using VXd = typename EO::VXd;

    py::module::import("elastic_object");

    using DS = DynamicSimulator<Real>;

    using TSM = TimesteppingMethod;
    py::enum_<TSM>(detail_module, "TimesteppingMethod")
        .value("ImplicitNewmark", TSM::ImplicitNewmark)
        .value("BackwardEuler",   TSM::BackwardEuler)
        .export_values()
        ;

    py::class_<DS, std::shared_ptr<DS>>(detail_module, ("DynamicProblem" + floatingPointTypeSuffix<Real>()).c_str())
        .def("run", &DS::run, py::arg("t0") = 0.0, py::arg("time") = 1.0)
        .def_property_readonly("problem",     &DS::getProblem)
        .def_property_readonly("inertiaLoad", [](const DS &ds) -> const Loads::Load<Real_> & { return ds.inertiaLoad(); })
        .def_readwrite("method", &DS::method)
        .def_readwrite("v",      &DS::v)
        .def_property_readonly("kineticEnergies",   &DS::kineticEnergies)
        .def_property_readonly("potentialEnergies", &DS::potentialEnergies)
        .def("getVars", &DS::getVars)

        .def_readwrite("beta",  &DS::beta,  "beta parameter for Newmark time stepping")
        .def_readwrite("gamma", &DS::gamma, "gamma parameter for Newmark time stepping")

        .def("setInitVelocity", &DS::setInitVelocity)
        .def("setXhat", &DS::setXhat)
        .def("getXhat", &DS::getXhat)

        .def("setPostTimestepCallback", [](DS &ds, const PyCallbackFunction<           DS> &pcb) { ds.setPostTimestepCallback(callbackWrapper<           DS>(pcb)); }, py::arg("cb"))
        .def("setPreTimestepCallback",  [](DS &ds, const PyCallbackFunction<           DS> &pcb) { ds. setPreTimestepCallback(callbackWrapper<           DS>(pcb)); }, py::arg("cb"))
        .def("setNewtonCallback",       [](DS &ds, const PyCallbackFunction<NewtonProblem> &pcb) { ds.      setNewtonCallback(callbackWrapper<NewtonProblem>(pcb)); }, py::arg("cb"))

        .def_property("fixedVars", &DS::fixedVars, &DS::setFixedVars)

        .def_readwrite("dt", &DS::dt)

        .def_property_readonly("optimizer", [](const DS &ds) -> NewtonOptimizer & { return ds.getOptimizer(); }, py::return_value_policy::reference_internal)
        .def("configureInertiaTerm", &DS::configureInertiaForTimeStep)
        ;

    m.def("DynamicSimulator",
            [](const std::shared_ptr<EO> &obj, std::vector<std::shared_ptr<NewtonObjectiveTermBase>> &terms, bool useLumpedMass, const double dt) {
                return std::make_shared<DS>(obj, terms, useLumpedMass, dt); },
            py::arg("obj"), py::arg("terms") = nullptr, py::arg("useLumpedMass") = false, py::arg("dt") = 1.0);
}

PYBIND11_MODULE(dynamic_simulator, m)
{
    m.doc() = "Bindings for the DynamicSimulator class";

    py::module detail_module = m.def_submodule("detail");

#if MESHFEM_BIND_LONG_DOUBLE
    bind<long double>(m, detail_module);
#endif
    bind<double>(m, detail_module);
}
