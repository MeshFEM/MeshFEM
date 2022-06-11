#ifndef LOADBINDING_HH
#define LOADBINDING_HH

#include "BindingInstantiations.hh"

#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/Gravity.hh>


template<class Object>
static std::enable_if_t<Object::N == 3> bindGravity(py::module &module, py::module &detail_module, const char* name) {
    using Load = Loads::Load<double>;

    ////////////////////////////////////////////////////////////////////////
    // Gravity
    ////////////////////////////////////////////////////////////////////////
    using GLoad = Loads::Gravity<Object>;
    py::class_<GLoad, Load, std::shared_ptr<GLoad>>(detail_module, name)
       .def_property("rho", &GLoad::get_rho, &GLoad::set_rho)
       ;

    using V3d = Eigen::Vector3d;
    module.def("Gravity", [&](const std::shared_ptr<Object> &obj, double rho, const V3d &g) {
                return std::make_shared<GLoad>(obj, rho, g);
            }, py::arg("obj"), py::arg("rho"), py::arg("g") = V3d(0.0, 0.0, 9.80635 * 1e3))
         ;
}

#endif /* end of include guard: LOADBINDING_HH */
