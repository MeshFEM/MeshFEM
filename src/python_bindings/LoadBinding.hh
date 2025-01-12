#ifndef LOADBINDING_HH
#define LOADBINDING_HH

#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/Loads/Inertia.hh>

#include <pybind11/pybind11.h>

namespace py = pybind11; // NOLINT (work around clang-tidy bug)

template<class Object>
static void bindGravity(py::module &m, py::module &detail_module) {
    using Load = Loads::Load<double>;
    using GLoad = Loads::Gravity<Object>;
    py::class_<GLoad, Load, std::shared_ptr<GLoad>>(detail_module, ("Gravity" + NameMangler<Object>::name()).c_str())
        .def_property("g", &GLoad::get_g, &GLoad::set_g, "Gravitational acceleration vector")
       ;

    m.def("Gravity", [&](const std::shared_ptr<Object> &obj, const typename GLoad::VNd &g) {
            return std::make_shared<GLoad>(obj, g);
        }, py::arg("obj"), py::arg("g") = GLoad::default_gravity());
}

template<class Object>
static void bindInertia(py::module &m, py::module &detail_module) {
    using Load = Loads::Load<double>;
    using ILoad = Loads::Inertia<Object>;
    py::class_<ILoad, Load, std::shared_ptr<ILoad>>(detail_module, ("Inertia" + NameMangler<Object>::name()).c_str())
        .def_readonly("xhat", &ILoad::xhat)
        .def_readonly("weight", &ILoad::xhat)
        .def_readonly("M",      &ILoad::M, py::return_value_policy::reference_internal)
       ;

    m.def("Inertia", [&](const std::shared_ptr<Object> &obj, bool lumpedMass) {
                return std::make_shared<ILoad>(obj, lumpedMass);
            }, py::arg("obj"), py::arg("lumpedMass") = true);
}

#endif /* end of include guard: LOADBINDING_HH */
