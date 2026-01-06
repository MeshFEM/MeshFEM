#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/Loads/Load.hh>
#include <MeshFEM/Loads/Springs.hh>
#include <MeshFEM/Loads/ProjectedAttachmentPoint.hh>
#include <MeshFEM/Loads/BodyForce.hh>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include "LoadBinding.hh"
#include "BindingInstantiations.hh"

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

PYBIND11_MODULE(loads, m)
{
    py::module::import("py_newton_optimizer");
    py::module::import("closest_point_projection");
    py::module::import("sparse_matrices");

    using Load = Loads::Load<double>;
    py::class_<Load, NewtonObjectiveTermBase, std::shared_ptr<Load>>(m, "Load")
        .def("energy",               &Load::energy)
        .def("grad_x",               &Load::grad_x)
        .def("grad_X",               &Load::grad_X)
        ;

    using BFLoad = Loads::BodyForce<double>;
    py::class_<BFLoad, Load, std::shared_ptr<BFLoad>>(m, "BodyForce")
       .def(py::init<std::shared_ptr<BFLoad::EO>>(), py::arg("obj"))
       .def(py::init([](const std::shared_ptr<BFLoad::EO> &obj, const Eigen::Ref<const BFLoad::MXd> &f) {
                auto bf = std::make_shared<BFLoad>(obj);
                bf->setNodalForceDensity(f);
                return bf;
            }), py::arg("obj"), py::arg("f"))
       .def_property("nodalForceDensity", &BFLoad::getNodalForceDensity, &BFLoad::setNodalForceDensity)
       ;

    py::module detail_module = m.def_submodule("detail");
    generateElasticObjectBindings(m, detail_module, LoadBinder());

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
}

