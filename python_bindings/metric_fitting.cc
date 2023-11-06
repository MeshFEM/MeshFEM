#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/BindMembraneMaterial.hh>
#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>

#include "../MetricFitter.hh"

PYBIND11_MODULE(metric_fitting, m)
{
    py::module::import("mesh_energy");
    py::module detail_module = m.def_submodule("detail");

    using Mesh = MetricFitter::Mesh;

    py::class_<MetricFitter, std::shared_ptr<MetricFitter>, NewtonMultiobjectiveProblem>(m, "MetricFitter")
        .def(py::init<std::shared_ptr<MetricFitter::Mesh>>(), py::arg("m"))
        .def("getFB",                 &MetricFitter::getFB)
        .def("setTargetMetric",       &MetricFitter::setTargetMetric, py::arg("ei"), py::arg("G"), py::arg("relativeCollapsePreventionThreshold") = 0.25)
        .def("programCurrentMetric",  &MetricFitter::programCurrentMetric)
        .def("metricDistSq",          &MetricFitter::metricDistSq)
        .def_property("bendingStiffness", &MetricFitter::bendingStiffness, &MetricFitter::setBendingStiffness)
        ;

    using  MF = CompositeEnergyDensity<MetricFittingEnergy<double, 2>, CollapsePreventionEnergyDet<double, 2>>;
    using MMF = MembraneMaterial<MF>;
    bindMembraneMaterial<MF>(m, detail_module)
        .def_property_readonly("fitting",            [](const MMF &mmf) { return mmf.psi.psi1; }, py::return_value_policy::reference_internal)
        .def_property_readonly("collapsePrevention", [](const MMF &mmf) { return mmf.psi.psi2; }, py::return_value_policy::reference_internal)
        ;

    bindMeshEnergy<MetricFittingMeshEnergy>("MetricFitting", m, detail_module);
}
