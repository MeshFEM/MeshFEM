#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (workaround clang-tidy bug)

#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/EquilibriumSolver.hh>
#include <MeshFEM/DynamicSimulator.hh>
#include <MeshFEM/IPCObjectiveTerm.hh>
#include "EquilibriumBinding.hh"

template<typename Real_>
void bind(py::module &m, py::module &detail_module) {
    using EO = ElasticObject<Real_>;
    using VXd = typename EO::VXd;

    py::module::import("elastic_object");

    // Bind DynamicSimulator
    using DS = DynamicSimulator<Real>;
    py::class_<DS, std::shared_ptr<DS>>(detail_module, ("DynamicProblem" + floatingPointTypeSuffix<Real>()).c_str())
    .def("run", [](DS &self, const std::vector<size_t> &fixedVars, const TimestepCallback &cb, const TimestepCallback &pre_cb, const PyCallbackFunction &cb_newton, const double finalTime) {
            return self.run(fixedVars, cb, pre_cb, callbackWrapper(cb_newton), finalTime);
        },
        py::arg("fixedVars") = std::vector<size_t>(), py::arg("tcb") = nullptr, py::arg("pre_tcb") = nullptr, py::arg("ncb") = nullptr, py::arg("finalTime") = 1.0,
                                py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def_property_readonly("problem",     &DS::getProblem)
        .def_property_readonly("inertiaLoad", [](const DS &ds) -> const Loads::Load<Real_> & { return ds.inertiaLoad(); })
        .def_readwrite("method", &DS::method)
        .def_readwrite("v",      &DS::v)
        .def_readonly("tIter", &DS::tIter)
        .def_property_readonly("kineticEnergies",   &DS::kineticEnergies)
        .def_property_readonly("potentialEnergies", &DS::potentialEnergies)
        .def("getVars", &DS::getVars)
        .def("setBata", &DS::setBeta)
        .def("setGamma", &DS::setGamma)
        .def("setInitVelocity", &DS::setInitVelocity)
        .def("setXhat", &DS::setXhat)
        .def("getXhat", &DS::getXhat)
        .def_property_readonly("optimizer", [](const DS &ds) -> NewtonOptimizer & { return ds.getOptimizer(); }, py::return_value_policy::reference_internal)
        .def("configureInertiaTerm", &DS::configureInertiaForTimeStep)
        ;

    m.def("DynamicSimulator",
            [](const std::shared_ptr<EO> &obj, std::vector<std::shared_ptr<NewtonObjectiveTermBase>> &terms, const NewtonOptimizerOptions &opts, bool useLumpedMass, const double dt) {
                return std::make_shared<DS>(obj, terms, opts, useLumpedMass, dt); },
            py::arg("obj"), py::arg("terms") = nullptr, py::arg("opts") = NewtonOptimizerOptions(), py::arg("useLumpedMass") = false, py::arg("dt") = 1.0);
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
