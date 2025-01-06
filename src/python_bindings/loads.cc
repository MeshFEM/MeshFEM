#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/Gravity.hh>
// #include <MeshFEM/Loads/Spreaders.hh>
// #include <MeshFEM/Loads/Springs.hh>
// #include <MeshFEM/Loads/ProjectedAttachmentPoint.hh>
#include <MeshFEM/Loads/SphereFitter.hh>
#include <MeshFEM/Loads/CircumcenterBarrier.hh>
#include <MeshFEM/Loads/Traction.hh>
#include <MeshFEM/Loads/Inflation.hh>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include "LoadBinding.hh"

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

#if 0
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
#endif
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
            module.def("CircumcenterBarrier", [&](const std::shared_ptr<Object> &obj, Real bc_min, bool subdivisionBarrier) {
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
    static std::enable_if_t<Object::N == 2> bind(py::module &/* module */, py::module &/* detail_module */) {
        // No loads are defined for 2D yet
    }
};

#if 0
using APC = Loads::AttachmentPointCoordinate<double>;
template<class Springs>
auto bindSprings(py::module &m, const std::string name) {
    using Load = Loads::Load<double>;
    using VXd  = Eigen::VectorXd;
    py::class_<Springs, Load, std::shared_ptr<Springs>> pySprings(m, name.c_str());
    pySprings
        .def("getStiffnesses", &Springs::getStiffnesses)
        .def("setStiffnesses", [](Springs &s, double     val ) { s.setStiffnesses(val ); }, py::arg("val"))
        .def("setStiffnesses", [](Springs &s, const VXd &vals) { s.setStiffnesses(vals); }, py::arg("vals"))
        .def("attachmentPointA", &Springs::attachmentPointA, py::arg("s"), py::return_value_policy::reference_internal)
        .def("attachmentPointB", &Springs::attachmentPointB, py::arg("s"), py::return_value_policy::reference_internal)
        .def("numSprings",       &Springs::numSprings)
        ;
    return pySprings;
}

template<size_t N>
void bindProjectedSprings(py::module &m, py::module &detail_module) {
    using Load = Loads::Load<double>;
    using VXd  = Eigen::VectorXd;
    using VNd  = VecN_T<double, N>;

    using  APC = Loads::AttachmentPointCoordinate<VNd>;
    using PAPC = Loads:: ProjectedAttachmentPoint<VNd>;
    using Springs = Loads::GenericSprings<APC, PAPC>;

    py::class_<PAPC>(detail_module, ("ProjectedAttachmentPoint" + std::to_string(N)).c_str())
        .def("d_dvar",      &PAPC::d_dvar, py::arg("vi"))
        .def("get_dp_dq",   &PAPC::get_dp_dq)
        .def("projector",   &PAPC::projector)

        .def_property_readonly("position", [](const PAPC &p) { return p.getPosition(); })
        .def_property_readonly("preprojectedPosition", [](const PAPC &p) { return p.getPreprojectedPoint(); })

        .def_readwrite("preprojectionAttachmentPoint", &PAPC::preprojectionAttachmentPoint)
        ;

    bindSprings<Springs>(detail_module, "ProjectedSprings" + std::to_string(N));

    m.def("ProjectedSprings", [](const std::shared_ptr<NewtonVarsBase> &obj,
                                 const SuiteSparseMatrix &dsm,
                                 std::shared_ptr<ClosestPointProjection<VNd>> proj,
                                 const VXd &stiffnesses) {
              return std::make_shared<Springs>(obj, APC::fromDeformationSamplerMatrix(dsm), PAPC::fromDeformationSamplerMatrix(dsm, proj), stiffnesses);
          }, py::arg("obj"), py::arg("deformationSamplerMatrix"), py::arg("closestPointProjector"), py::arg("stiffnesses"))
     ;

    m.def("ProjectedSprings", [](const std::shared_ptr<NewtonVarsBase> &obj,
                                 const SuiteSparseMatrix &dsm,
                                 std::shared_ptr<ClosestPointProjection<VNd>> proj,
                                 double stiffness) {
              return std::make_shared<Springs>(obj, APC::fromDeformationSamplerMatrix(dsm), PAPC::fromDeformationSamplerMatrix(dsm, proj), stiffness);
          }, py::arg("obj"), py::arg("deformationSamplerMatrix"), py::arg("closestPointProjector"), py::arg("stiffness") = 1.0)
     ;

    m.def("ProjectedSprings", [](const std::shared_ptr<NewtonVarsBase> &obj,
                                 const Eigen::VectorXi &blockVars,
                                 std::shared_ptr<ClosestPointProjection<VNd>> proj,
                                 const VXd &stiffnesses) {
              std::vector<APC> apc = APC::fromBlockVars(blockVars);
              return std::make_shared<Springs>(obj, apc, PAPC::fromAttachmentPoints(apc, proj), stiffnesses);
          }, py::arg("obj"), py::arg("blockVars"), py::arg("closestPointProjector"), py::arg("stiffnesses"))
     ;

    m.def("ProjectedSprings", [](const std::shared_ptr<NewtonVarsBase> &obj,
                                 const Eigen::VectorXi &blockVars,
                                 std::shared_ptr<ClosestPointProjection<VNd>> proj,
                                 double stiffnesses) {
              std::vector<APC> apc = APC::fromBlockVars(blockVars);
              return std::make_shared<Springs>(obj, apc, PAPC::fromAttachmentPoints(apc, proj), stiffnesses);
          }, py::arg("obj"), py::arg("blockVars"), py::arg("closestPointProjector"), py::arg("stiffness") = 1.0)
     ;
}
#endif

PYBIND11_MODULE(loads, m)
{
    py::module::import("py_newton_optimizer");
    py::module::import("closest_point_projection");

    using Load = Loads::Load<double>;
    py::class_<Load, NewtonObjectiveTermBase, std::shared_ptr<Load>>(m, "Load")
        .def("energy",               &Load::energy)
        .def("grad_x",               &Load::grad_x)
        .def("grad_X",               &Load::grad_X)
        ;

    py::module detail_module = m.def_submodule("detail");
    generateElasticObjectBindings(m, detail_module, LoadBinder());

#if 0
    py::class_<APC>(m, "AttachmentPointCoordinate")
        .def(py::init<Eigen::Ref<const typename APC::VXi>, Eigen::Ref<const typename APC::VXd>>(), py::arg("varIndices"), py::arg("coefficients"), "Material attachment point coordinate")
        .def(py::init<typename APC::Real                                                      >(), py::arg("coordinate"),                          "Fixed anchor point coordinate")
        .def("isFixedAnchor", &APC::isFixedAnchor)
        .def("getPosition",   [](const APC &apc, const APC::VXd &vars) { return apc.getPosition(vars); }, py::arg("vars"))
        .def_readwrite("varIndices",   &APC::varIndices)
        .def_readwrite("coefficients", &APC::coefficients)
        ;

    ////////////////////////////////////////////////////////////////////////////
    // Springs
    ////////////////////////////////////////////////////////////////////////////
    using Springs = Loads::Springs;
    using VXd  = Eigen::VectorXd;
    bindSprings<Springs>(m, "Springs")
        .def(py::init([&](const std::shared_ptr<NewtonVarsBase> &obj,
                          const std::vector<APC> &coordsA,
                          const std::vector<APC> &coordsB,
                          Eigen::Ref<const VXd> stiffnesses) {
              return std::make_shared<Springs>(obj, coordsA, coordsB, stiffnesses);
          }), py::arg("obj"), py::arg("coordsA"), py::arg("coordsB"), py::arg("stiffnesses"))
        .def(py::init([&](const std::shared_ptr<NewtonVarsBase> &obj,
                          const std::vector<APC> &coordsA,
                          const std::vector<APC> &coordsB,
                          typename Springs::Real stiffness) {
              return std::make_shared<Springs>(obj, coordsA, coordsB, stiffness);
          }), py::arg("obj"), py::arg("coordsA"), py::arg("coordsB"), py::arg("stiffness"))
        .def(py::init([&](const std::shared_ptr<NewtonVarsBase> &obj,
                          const SuiteSparseMatrix &dsm,
                          Eigen::Ref<const Eigen::VectorXd> tgt,
                          Eigen::Ref<const VXd> stiffnesses) {
              return std::make_shared<Springs>(obj, dsm, tgt, stiffnesses);
          }), py::arg("obj"), py::arg("deformationSamplerMatrix"),
              py::arg("targetPositions"), py::arg("stiffnesses"))
        .def(py::init([&](const std::shared_ptr<NewtonVarsBase> &obj,
                          const SuiteSparseMatrix &dsm,
                          Eigen::Ref<const Eigen::VectorXd> tgt,
                          typename Springs::Real stiffness) {
              return std::make_shared<Springs>(obj, dsm, tgt, stiffness);
          }), py::arg("obj"), py::arg("deformationSamplerMatrix"),
              py::arg("targetPositions"), py::arg("stiffness"))
       ;

    bindProjectedSprings<2>(m, detail_module);
    bindProjectedSprings<3>(m, detail_module);
#endif
}

