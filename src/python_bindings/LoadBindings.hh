#ifndef LOADBINDINGS_HH
#define LOADBINDINGS_HH

#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/Loads/Spreaders.hh>

namespace impl {
    template<class L>
    void addCommonLoadBindings(py::class_<L> &pyLoad, py::module &detail_module) {
        pyLoad.def("energy",  &L::energy);
        pyLoad.def("deformedStateUpdated", &L::deformedStateUpdated);
        pyLoad.def("restStateUpdated",     &L::restStateUpdated);
        pyLoad.def("grad_x",               &L::grad_x);
        pyLoad.def("grad_X",               &L::grad_X);
        pyLoad.def("hessian", [](const L &l) { auto H = l.hessianSparsityPattern(0.0); l.hessian(H); return H; })
        ;
    }
}

template<class Object, class PyObj, typename std::enable_if<Object::N == 3, int>::type = 0>
void addLoadBindings(PyObj &pyObj, py::module &detail_module, const std::string &objectName) {
    using GLoad = Loads::Gravity<Object>;
    using V3d = Eigen::Vector3d;
    py::class_<GLoad> gravity(detail_module, ("Gravity" + objectName).c_str());
    gravity// .def(py::init<const Object &, double, const V3d &>(), py::arg("obj"), py::arg("rho"), py::arg("g") = V3d(0.0, 0.0, 9.80635))
           .def_property("rho", &GLoad::get_rho, &GLoad::set_rho)
           ;
    impl::addCommonLoadBindings<GLoad>(gravity, detail_module);
    pyObj.def("GravityLoad", [&](const Object &obj, double rho, const V3d &g) {
                return std::make_unique<GLoad>(obj, rho, g);
            }, py::arg("rho"), py::arg("g") = V3d(0.0, 0.0, 9.80635));

    using SLoad = Loads::Spreaders<Object>;
    using MX2i = Eigen::MatrixX2i;
    using VXi  = Eigen::VectorXi;
    py::class_<SLoad> spreader(detail_module, ("Spreaders" + objectName).c_str());
    impl::addCommonLoadBindings<SLoad>(spreader, detail_module);
    pyObj.def("SpreadersLoad", [&](const Object &obj, const std::vector<VXi> &clusterVtxs,
                                   const MX2i &connectivity, Real force, bool disableHessian) {
                return std::make_unique<SLoad>(obj, clusterVtxs, connectivity, force, disableHessian);
            }, py::arg("clusterVtxs"), py::arg("connectivity"), py::arg("force"), py::arg("disableHessian") = false);
}

template<class Object, class PyObj, typename std::enable_if<Object::N != 3, int>::type = 0>
void addLoadBindings(PyObj &pyObj, py::module &detail_module, const std::string &objectName) {
    // No loads are defined yet for 2D...
}

#endif /* end of include guard: LOADBINDINGS_HH */
