#ifndef LOADBINDING_HH
#define LOADBINDING_HH

#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/BodyForce.hh>
#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/Loads/RegionNetForce.hh>
#include <MeshFEM/Loads/Traction.hh>
#include <MeshFEM/Loads/Inertia.hh>
#include <MeshFEM/Loads/Spreaders.hh>
#include <MeshFEM/Loads/SphereFitter.hh>
#include <MeshFEM/Loads/CircumcenterBarrier.hh>
#include <MeshFEM/Loads/Inflation.hh>

#include <pybind11/pybind11.h>

namespace py = pybind11; // NOLINT (work around clang-tidy bug)

namespace MeshFEM {

template<class Object>
static void bindBodyForces(py::module &m, py::module &detail_module) {
    using Load   = Loads::Load<double>;
    using GLoad  = Loads::Gravity<Object>;
    using BFLoad = Loads::BodyForce<Object>;
    using NFLoad = Loads::RegionNetForce<Object>;

    py::class_<BFLoad, Load, std::shared_ptr<BFLoad>>(detail_module, ("BodyForce" + NameMangler<Object>::name()).c_str())
       .def_property("nodalForceDensity", &BFLoad::getNodalForceDensity, &BFLoad::setNodalForceDensity)
       ;

    m.def("BodyForce", [&](const std::shared_ptr<Object> &obj, const Eigen::Ref<const typename BFLoad::MXd> &f) {
            return std::make_shared<BFLoad>(obj, f);
        }, py::arg("obj"), py::arg("f"));

    py::class_<GLoad, Load, std::shared_ptr<GLoad>>(detail_module, ("Gravity" + NameMangler<Object>::name()).c_str())
        .def_property("g", &GLoad::get_g, &GLoad::set_g, "Gravitational acceleration vector")
       ;

    m.def("Gravity", [&](const std::shared_ptr<Object> &obj, const typename GLoad::VNd &g) {
            return std::make_shared<GLoad>(obj, g);
        }, py::arg("obj"), py::arg("g") = GLoad::default_gravity());

    py::class_<NFLoad, Load, std::shared_ptr<NFLoad>>(detail_module, ("RegionNetForce" + NameMangler<Object>::name()).c_str())
       .def_property("indicatorField", &NFLoad::get_indicator_field, &NFLoad::set_indicator_field, "Indicator field defining the region over which the net force is applied")
       .def_property("f", &NFLoad::get_f, &NFLoad::set_f, "Net force applied by this load")
       ;

    m.def("NetForce", [&](const std::shared_ptr<Object> &obj, const typename NFLoad::VNd &f, const Eigen::VectorXd &indicatorField) {
            auto result = std::make_shared<NFLoad>(obj);
            result->set_f(f);
            if (indicatorField.size() > 0)
                result->set_indicator_field(indicatorField);
            return result;
        }, py::arg("obj"), py::arg("f"), py::arg("indicatorField") = Eigen::VectorXd());
}

template<class Object>
static void bindInertia(py::module &m, py::module &detail_module) {
    using Load = Loads::Load<double>;
    using ILoad = Loads::Inertia<Object>;
    py::class_<ILoad, Load, std::shared_ptr<ILoad>>(detail_module, ("Inertia" + NameMangler<Object>::name()).c_str())
        .def_readonly("xhat", &ILoad::xhat)
        .def_readonly("weight", &ILoad::xhat)
        .def_readonly("M_full",   &ILoad::M_full,   py::return_value_policy::reference_internal)
        .def_readonly("M_lumped", &ILoad::M_lumped, py::return_value_policy::reference_internal)
       ;

    m.def("Inertia", [&](const std::shared_ptr<Object> &obj, bool lumpedMass) {
                return std::make_shared<ILoad>(obj, lumpedMass);
            }, py::arg("obj"), py::arg("lumpedMass") = true);
}

struct LoadBinder {
    // Bind loads for a particular elastic structure type `Object`
    template<class Object>
    static void bind_generic(py::module &m, py::module &detail_module) {
        using Real = typename Object::Real;
        using Load = Loads::Load<Real>;

        bindBodyForces<Object>(m, detail_module);
        bindInertia<Object>(m, detail_module);

        ////////////////////////////////////////////////////////////////////////
        // Traction
        ////////////////////////////////////////////////////////////////////////
        using TLoad = Loads::Traction<Object>;
        py::class_<TLoad, Load, std::shared_ptr<TLoad>>(detail_module, ("Traction" + NameMangler<Object>::name()).c_str())
           .def_property("boundaryTractions", &TLoad::getBoundaryTractions, &TLoad::setBoundaryTractions)
           ;

        m.def("Traction", [&](const std::shared_ptr<Object> &obj) {
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
        m.def("Spreaders", [&](const std::shared_ptr<Object> &obj, const std::vector<VXi> &clusterVtxs,
                                   const MX2i &connectivity, Real force, bool disableHessian) {
                    return std::make_shared<SLoad>(obj, clusterVtxs, connectivity, force, disableHessian);
                }, py::arg("obj"), py::arg("clusterVtxs"), py::arg("connectivity"), py::arg("force"), py::arg("disableHessian") = false)
         .def("Spreaders", [&](const std::shared_ptr<Object> &obj, const SuiteSparseMatrix &S,
                              const MX2i &connectivity, Real force, bool disableHessian) {
               return std::make_shared<SLoad>(obj, S, connectivity, force, disableHessian);
           }, py::arg("obj"), py::arg("deformationSamplerMatrix"), py::arg("connectivity"), py::arg("force"), py::arg("disableHessian") = false)
         ;
    }

    template<class Object>
    static std::enable_if_t<(Object::N == 3) && (Object::K == 3)> bind(py::module &m, py::module &detail_module) {
        bind_generic<Object>(m, detail_module);

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
        m.def("SphereFitter", [&](const std::shared_ptr<Object> &obj, Real r_tgt, Real stiffness) {
                return std::make_shared<SphereFitter>(obj, r_tgt, stiffness);
            }, py::arg("obj"), py::arg("r_tgt") = 1.0, py::arg("r_tgt") = 1.0)
        ;

        if constexpr (Object::Deg == 1) {
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
            m.def("CircumcenterBarrier", [&](const std::shared_ptr<Object> &obj, Real bc_min, bool subdivisionBarrier) {
                    return std::make_shared<CB>(obj, bc_min, subdivisionBarrier);
                }, py::arg("obj"), py::arg("bc_min") = 0.0, py::arg("subdivisionBarrier") = false)
            ;
        }
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
    static std::enable_if_t<Object::N == 2> bind(py::module &m, py::module &detail_module) {
        bind_generic<Object>(m, detail_module);
    }
};

} // namespace MeshFEM

#endif /* end of include guard: LOADBINDING_HH */
