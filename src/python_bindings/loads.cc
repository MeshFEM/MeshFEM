#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "loads.hh"

using APC = Loads::AttachmentPointCoordinate<double>;

PYBIND11_MODULE(loads, m)
{
    bind<2>(m);
    bind<3>(m);

    py::module detail_module = m.def_submodule("detail");
    generateElasticObjectBindings(m, detail_module, LoadBinder());

    py::class_<APC>(m, "AttachmentPointCoordinate")
        .def(py::init<Eigen::Ref<const typename APC::VXi>, Eigen::Ref<const typename APC::VXd>>(), py::arg("varIndices"), py::arg("coefficients"), "Material attachment point coordinate")
        .def(py::init<typename APC::Real                                                      >(), py::arg("coordinate"),                          "Fixed anchor point coordinate")
        .def("isFixedAnchor", &APC::isFixedAnchor)
        .def("getPosition",   &APC::getPosition, py::arg("vars"))
        .def_readwrite("varIndices",   &APC::varIndices)
        .def_readwrite("coefficients", &APC::coefficients)
        ;
}
