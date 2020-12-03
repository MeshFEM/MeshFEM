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
#include <MeshFEM/Loads/Springs.hh>
#include <MeshFEM/Loads/Traction.hh>

using APC = Loads::AttachmentPointCoordinate<double>;

template<size_t N>
void bind(py::module &m) {
    using Load = Loads::Load<N, double>;
    py::class_<Load, std::shared_ptr<Load>>(m, ("Load" + std::to_string(N)).c_str())
        .def("energy",               &Load::energy)
        .def("grad_x",               &Load::grad_x)
        .def("grad_X",               &Load::grad_X)
        .def("hessian",                [](const Load &l) { auto H = l.hessianSparsityPattern(0.0); l.hessian(H); return H; })
        .def("hessianSparsityPattern", [](const Load &l) { return l.hessianSparsityPattern(1.0); })
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
        module.def("Gravity", [&](const std::shared_ptr<Object> &obj, double rho, const V3d &g) {
                    return std::make_shared<GLoad>(obj, rho, g);
                }, py::arg("obj"), py::arg("rho"), py::arg("g") = V3d(0.0, 0.0, 9.80635))
             ;

        ////////////////////////////////////////////////////////////////////////
        // Traction
        ////////////////////////////////////////////////////////////////////////
        using TLoad = Loads::Traction<Object>;
        py::class_<TLoad, Load, std::shared_ptr<TLoad>>(detail_module, ("Traction" + NameMangler<Object>::name()).c_str())
           .def_property("boundaryTractions", &TLoad::getBoundaryTractions, &TLoad::setBoundaryTractions)
           ;

        module.def("Traction", [&](const std::shared_ptr<Object> &obj) {
                    return std::make_shared<TLoad>(obj);
                }, py::arg("obj"))
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
        module.def("Spreaders", [&](const std::shared_ptr<Object> &obj, const std::vector<VXi> &clusterVtxs,
                                   const MX2i &connectivity, Real force, bool disableHessian) {
                    return std::make_shared<SLoad>(obj, clusterVtxs, connectivity, force, disableHessian);
                }, py::arg("obj"), py::arg("clusterVtxs"), py::arg("connectivity"), py::arg("force"), py::arg("disableHessian") = false)
              .def("Spreaders", [&](const std::shared_ptr<Object> &obj, const SuiteSparseMatrix &S,
                                   const MX2i &connectivity, Real force, bool disableHessian) {
                    return std::make_shared<SLoad>(obj, S, connectivity, force, disableHessian);
                }, py::arg("obj"), py::arg("deformationSamplerMatrix"), py::arg("connectivity"), py::arg("force"), py::arg("disableHessian") = false)
             ;

        ////////////////////////////////////////////////////////////////////////
        // Springs
        ////////////////////////////////////////////////////////////////////////
        using Springs = Loads::Springs<Object>;
        using VXd  = Eigen::VectorXd;
        py::class_<Springs, Load, std::shared_ptr<Springs>>(detail_module, ("Springs" + NameMangler<Object>::name()).c_str())
            .def("getStiffnesses", &Springs::getStiffnesses)
            .def("setStiffnesses", [](Springs &s, double     val ) { s.setStiffnesses(val ); }, py::arg("val"))
            .def("setStiffnesses", [](Springs &s, const VXd &vals) { s.setStiffnesses(vals); }, py::arg("vals"))
            ;
        module.def("Springs", [&](const std::shared_ptr<Object> &obj,
                                  const std::vector<APC> &coordsA,
                                  const std::vector<APC> &coordsB,
                                  Eigen::Ref<const VXd> stiffnesses) {
                    return std::make_shared<Springs>(obj, coordsA, coordsB, stiffnesses);
                }, py::arg("obj"), py::arg("coordsA"), py::arg("coordsB"), py::arg("stiffnesses"))
              .def("Springs", [&](const std::shared_ptr<Object> &obj,
                                  const std::vector<APC> &coordsA,
                                  const std::vector<APC> &coordsB,
                                  typename Springs::Real stiffness) {
                    return std::make_shared<Springs>(obj, coordsA, coordsB, stiffness);
                }, py::arg("obj"), py::arg("coordsA"), py::arg("coordsB"), py::arg("stiffness"))
              .def("Springs", [&](const std::shared_ptr<Object> &obj,
                                  const SuiteSparseMatrix &dsm,
                                  Eigen::Ref<const Eigen::VectorXd> tgt,
                                  Eigen::Ref<const VXd> stiffnesses) {
                    return std::make_shared<Springs>(obj, dsm, tgt, stiffnesses);
                }, py::arg("obj"), py::arg("deformationSamplerMatrix"),
                   py::arg("targetPositions"), py::arg("stiffnesses"))
              .def("Springs", [&](const std::shared_ptr<Object> &obj,
                                  const SuiteSparseMatrix &dsm,
                                  Eigen::Ref<const Eigen::VectorXd> tgt,
                                  typename Springs::Real stiffness) {
                    return std::make_shared<Springs>(obj, dsm, tgt, stiffness);
                }, py::arg("obj"), py::arg("deformationSamplerMatrix"),
                   py::arg("targetPositions"), py::arg("stiffness"))
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

    py::class_<APC>(m, "AttachmentPointCoordinate")
        .def(py::init<Eigen::Ref<const typename APC::VXi>, Eigen::Ref<const typename APC::VXd>>(), py::arg("varIndices"), py::arg("coefficients"), "Material attachment point coordinate")
        .def(py::init<typename APC::Real                                                      >(), py::arg("coordinate"),                          "Fixed anchor point coordinate")
        .def("isFixedAnchor", &APC::isFixedAnchor)
        .def("getPosition",   &APC::getPosition, py::arg("vars"))
        .def_readwrite("varIndices",   &APC::varIndices)
        .def_readwrite("coefficients", &APC::coefficients)
        ;
}
