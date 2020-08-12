#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/Utilities/NameMangling.hh>
#include "BindingInstantiations.hh"

#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/Loads/Spreaders.hh>

template<size_t N>
void bind(py::module &m) {
    using Load = Loads::Load<N, double>;
    py::class_<Load, std::shared_ptr<Load>>(m, ("Load" + std::to_string(N)).c_str())
        .def("energy",               &Load::energy)
        .def("deformedStateUpdated", &Load::deformedStateUpdated)
        .def("restStateUpdated",     &Load::restStateUpdated)
        .def("grad_x",               &Load::grad_x)
        .def("grad_X",               &Load::grad_X)
        .def("hessian", [](const Load &l) { auto H = l.hessianSparsityPattern(0.0); l.hessian(H); return H; })
        ;
}

struct LoadBinder {
    // Bind loads for a particular elastic structure type `Object`
    template<class Object>
    static std::enable_if_t<Object::N == 3> bind(py::module &module, py::module &detail_module) {
        using Load = Loads::Load<3, double>;

        ////////////////////////////////////////////////////////////////////////
        // Gravity
        ////////////////////////////////////////////////////////////////////////
        using GLoad = Loads::Gravity<Object>;
        py::class_<GLoad, Load, std::shared_ptr<GLoad>>(detail_module, ("Gravity" + NameMangler<Object>::name()).c_str())
           .def_property("rho", &GLoad::get_rho, &GLoad::set_rho)
           ;

        using V3d = Eigen::Vector3d;
        module.def("Gravity", [&](const Object &obj, double rho, const V3d &g) {
                    return std::make_shared<GLoad>(obj, rho, g);
                }, py::arg("obj"), py::arg("rho"), py::arg("g") = V3d(0.0, 0.0, 9.80635))
             ;

        ////////////////////////////////////////////////////////////////////////
        // Spreaders
        ////////////////////////////////////////////////////////////////////////
        using SLoad = Loads::Spreaders<Object>;
        using MX2i = Eigen::MatrixX2i;
        using VXi  = Eigen::VectorXi;
        py::class_<SLoad, Load, std::shared_ptr<SLoad>>(detail_module, ("Spreaders" + NameMangler<Object>::name()).c_str())
             .def_property("magnitude", &SLoad::getMagnitude, &SLoad::setMagnitude)
             ;
        module.def("Spreaders", [&](const Object &obj, const std::vector<VXi> &clusterVtxs,
                                   const MX2i &connectivity, Real force, bool disableHessian) {
                    return std::make_shared<SLoad>(obj, clusterVtxs, connectivity, force, disableHessian);
                }, py::arg("obj"), py::arg("clusterVtxs"), py::arg("connectivity"), py::arg("force"), py::arg("disableHessian") = false)
             ;
    }

    template<class Object>
    static std::enable_if_t<Object::N == 2> bind(py::module &/* module */, py::module &/* detail_module */) {
        // No loads are defined for 2D yet
    }
};

PYBIND11_MODULE(loads, m)
{
    bind<2>(m);
    bind<3>(m);

    py::module detail_module = m.def_submodule("detail");
    generateElasticObjectBindings(m, detail_module, LoadBinder());
}
