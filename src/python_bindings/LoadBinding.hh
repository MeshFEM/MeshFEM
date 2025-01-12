#ifndef LOADBINDING_HH
#define LOADBINDING_HH

#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/Gravity.hh>

#include <pybind11/pybind11.h>

namespace py = pybind11; // NOLINT (work around clang-tidy bug)

template<class Object>
static void bindGravity(py::module &module, py::module &detail_module, const char* name) {
    using Load = Loads::Load<double>;

    ////////////////////////////////////////////////////////////////////////
    // Gravity
    ////////////////////////////////////////////////////////////////////////
    using GLoad = Loads::Gravity<Object>;
    py::class_<GLoad, Load, std::shared_ptr<GLoad>> pyG(detail_module, name)
       ;

    using V3d = Eigen::Vector3d;
    module.def("Gravity", [&](const std::shared_ptr<Object> &obj, const V3d &g) {
                return std::make_shared<GLoad>(obj, g);
            }, py::arg("obj"), py::arg("g") = V3d(0.0, 0.0, 9.80635 * 1e3))
         ;
}

#endif /* end of include guard: LOADBINDING_HH */
