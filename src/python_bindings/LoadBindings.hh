#ifndef LOADBINDINGS_HH
#define LOADBINDINGS_HH

#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/Loads/Spreaders.hh>

template<class Object, class PyObj, typename std::enable_if<Object::N == 3, int>::type = 0>
void addLoadBindings(PyObj &pyObj, py::module &detail_module, const std::string &objectName) {
    using Load = Loads::Load<3, double>;
    using GLoad = Loads::Gravity<Object>;
    using V3d = Eigen::Vector3d;
    py::class_<GLoad, Load, std::shared_ptr<GLoad>>(detail_module, ("Gravity" + objectName).c_str())
           .def_property("rho", &GLoad::get_rho, &GLoad::set_rho)
           ;
    pyObj.def("GravityLoad", [&](const Object &obj, double rho, const V3d &g) {
                return std::make_shared<GLoad>(obj, rho, g);
            }, py::arg("rho"), py::arg("g") = V3d(0.0, 0.0, 9.80635))
         ;

    using SLoad = Loads::Spreaders<Object>;
    using MX2i = Eigen::MatrixX2i;
    using VXi  = Eigen::VectorXi;
    py::class_<SLoad, Load, std::shared_ptr<SLoad>>(detail_module, ("Spreaders" + objectName).c_str())
         .def_property("magnitude", &SLoad::getMagnitude, &SLoad::setMagnitude)
         ;
    pyObj.def("SpreadersLoad", [&](const Object &obj, const std::vector<VXi> &clusterVtxs,
                                   const MX2i &connectivity, Real force, bool disableHessian) {
                return std::make_shared<SLoad>(obj, clusterVtxs, connectivity, force, disableHessian);
            }, py::arg("clusterVtxs"), py::arg("connectivity"), py::arg("force"), py::arg("disableHessian") = false)
         ;
}

template<class Object, class PyObj, typename std::enable_if<Object::N != 3, int>::type = 0>
void addLoadBindings(PyObj &pyObj, py::module &detail_module, const std::string &objectName) {
    // No loads are defined yet for 2D...
}

#endif /* end of include guard: LOADBINDINGS_HH */
