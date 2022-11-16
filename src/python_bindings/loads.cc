#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/Loads/Spreaders.hh>
#include <MeshFEM/Loads/Springs.hh>
#include <MeshFEM/Loads/SphereFitter.hh>
#include <MeshFEM/Loads/CircumcenterBarrier.hh>
#include <MeshFEM/Loads/Traction.hh>
#include <MeshFEM/Loads/Inflation.hh>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "LoadBinding.hh"

using APC = Loads::AttachmentPointCoordinate<double>;

struct LoadBinder {
    // Bind loads for a particular elastic structure type `Object`
    template<class Object>
    static void bind_generic(py::module &module, py::module &detail_module) {
        using Real = typename Object::Real;
        using Load = Loads::Load<Real>;

        ////////////////////////////////////////////////////////////////////////
        // Gravity
        ////////////////////////////////////////////////////////////////////////
        bindGravity<Object>(module, detail_module, ("Gravity" + NameMangler<Object>::name()).c_str());

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
    static std::enable_if_t<(Object::N == 3) && (Object::K == 3)> bind(py::module &module, py::module &detail_module) {
        bind_generic<Object>(module, detail_module);

        ////////////////////////////////////////////////////////////////////////
        // Solid-specific load: SphereFitter, CircumcenterBarrier
        ////////////////////////////////////////////////////////////////////////
        using Real = typename Object::Real;
        using Load = Loads::Load<Real>;
        using SphereFitter = Loads::SphereFitter<Object>;
        py::class_<SphereFitter, Load, std::shared_ptr<SphereFitter>>(detail_module, ("SphereFitter" + NameMangler<Object>::name()).c_str())
            .def_readwrite("stiffness", &SphereFitter::stiffness)
            .def_readwrite("r_tgt",     &SphereFitter::r_tgt)
            ;
        module.def("SphereFitter", [&](const std::shared_ptr<Object> &obj, Real r_tgt, Real stiffness) {
                    return std::make_shared<SphereFitter>(obj, r_tgt, stiffness);
                }, py::arg("obj"), py::arg("r_tgt") = 1.0, py::arg("r_tgt") = 1.0)
        ;

        using CB = Loads::CircumcenterBarrier<Object>;
        py::class_<CB, Load, std::shared_ptr<CB>>(detail_module, ("CircumcenterBarrier" + NameMangler<Object>::name()).c_str())
            .def("subtets", &CB::subtets, py::arg("ei"), "for debugging")
            .def_property("activationThreshold", [](const CB &cb) { return cb.barrier.activationThreshold; },
                                                 [](CB &cb, Real v) { cb.barrier.activationThreshold = v; }, "value at which the barrier term kicks in")
            .def_property("barrierThreshold", [](const CB &cb) { return cb.barrier.barrierThreshold; },
                                              [](CB &cb, Real v) { cb.barrier.barrierThreshold = v; }, "value at which the barrier term becomes infinite")
            .def("minCircumcenterBC", &CB::minCircumcenterBC, "Get the smallest barycentric coordinate of any of the elements (or any of the sub-elements if `m_subdivisionBarrier` is `true`).")
            .def_readwrite("bc_min", &CB::bc_min)
            ;
        module.def("CircumcenterBarrier", [&](const std::shared_ptr<Object> &obj, Real bc_min, bool subdivisionBarrier) {
                    return std::make_shared<CB>(obj, bc_min, subdivisionBarrier);
                }, py::arg("obj"), py::arg("bc_min") = 0.0, py::arg("subdivisionBarrier") = false)
        ;
    }

    template<class Object>
    static std::enable_if_t<(Object::N == 3) && (Object::K == 2)> bind(py::module &module, py::module &detail_module) {
        bind_generic<Object>(module, detail_module);

        ////////////////////////////////////////////////////////////////////////
        // Sheet-specific load: Inflation
        ////////////////////////////////////////////////////////////////////////
        using Real = typename Object::Real;
        using Load = Loads::Load<Real>;
        using Inflation = Loads::Inflation<Object>;
        py::class_<Inflation, Load, std::shared_ptr<Inflation>>(detail_module, ("Inflation" + NameMangler<Object>::name()).c_str())
            .def("volume", &Inflation::volume)
            .def_readwrite("pressure", &Inflation::pressure)
            ;

        module.def("Inflation", [&](const std::shared_ptr<Object> &obj, Real pressure) {
                    return std::make_shared<Inflation>(obj, pressure);
                }, py::arg("sheet"), py::arg("pressure") = 1.0);

    }

    template<class Object>
    static std::enable_if_t<Object::N == 2> bind(py::module &/* module */, py::module &/* detail_module */) {
        // No loads are defined for 2D yet
    }
};

PYBIND11_MODULE(loads, m)
{
    using Load = Loads::Load<double>;
    py::class_<Load, std::shared_ptr<Load>>(m, "Load")
        .def("energy",               &Load::energy)
        .def("grad_x",               &Load::grad_x)
        .def("grad_X",               &Load::grad_X)
        .def("hessian",                [](const Load &l) { auto H = l.hessianSparsityPattern(0.0); l.hessian(H); return H; })
        .def("hessianSparsityPattern", [](const Load &l) { return l.hessianSparsityPattern(1.0); })
        ;

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

