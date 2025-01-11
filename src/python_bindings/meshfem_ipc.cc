#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
namespace py = pybind11; // NOLINT (workaround clang-tidy bug)

#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/EquilibriumSolver.hh>
#include <MeshFEM/DynamicSimulator.hh>
#include <MeshFEM/IPCIntegration/IPCObjectiveTerm.hh>
#include <MeshFEM/IPCIntegration/Obstacle.hh>

template<typename Real_>
void bind(py::module &m, py::module &detail_module) {
    using EO = ElasticObject<Real_>;
    py::module::import("elastic_object");
#if MESHFEM_WITH_IPC_TOOLKIT
    // Bind IPCObjective
    using IPCO = IPCObjectiveTerm<Real_>;
    py::class_<IPCO, NewtonObjectiveTerm, std::shared_ptr<IPCO>>(detail_module, ("IPCObjectiveTerm" + floatingPointTypeSuffix<Real_>()).c_str())
        .def_property("useAdaptiveBarrier",   &IPCO::getUseAdaptiveBarrierStiffness, &IPCO::setUseAdaptiveBarrierStiffness)
        .def_property_readonly("object",      &IPCO::object)
        .def("getCollisionVertexPositions",   &IPCO::getCollisionVertexPositions)
        .def("getCollisionMeshFaces",         &IPCO::getCollisionMeshFaces)
        .def("getCollisionMeshEdges",         &IPCO::getCollisionMeshEdges)
        .def_property("barrierStiffness",     &IPCO::getBarrierStiffness, &IPCO::setBarrierStiffness)
        .def_property("dhat",                 &IPCO::get_dhat, &IPCO::set_dhat)
        .def("contactEnergy",                 &IPCO::contactPotentialEnergy)
        .def_readwrite("CCD",                 &IPCO::CCD)
        .def("contactGradient",               &IPCO::contactGradient)
        .def("numCollisionConstraints",       &IPCO::numCollisionConstraints)
        .def("CCDFeasibleStepLength",         &IPCO::CCDFeasibleStepLength, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("CCDStepSize",                   &IPCO::CCDStepSize, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        ;

    m.def("IPCObjectiveTerm", [](std::shared_ptr<EO> eo, const ObstaclesCollection &obstacles) { return std::make_shared<IPCO>(eo, obstacles); }, py::arg("eo"), py::arg("obstacles") = ObstaclesCollection(), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    using MXd = Eigen::MatrixXd;
    using MXi = Eigen::MatrixXi;
    using VXd = Eigen::VectorXd;
    
    py::class_<Obstacle, std::shared_ptr<Obstacle>> pyObst(m, "Obstacle");
    pyObst.def(py::init<const MXd, const MXi, const MXi, const Obstacle::xFunction>(), py::arg("vertices") = MXd(), py::arg("faces") = MXi(), py::arg("edges") = MXi(), py::arg("xfunction") = Obstacle::xFunction())
          .def("getPositions",      &Obstacle::getVertices)
          .def("getForce",          &Obstacle::getForce)
          .def("getEdges",          &Obstacle::getEdges);
#endif
}

PYBIND11_MODULE(meshfem_ipc, m)
{
    m.doc() = "Bindings for MeshFEM's IPC support";

    py::module detail_module = m.def_submodule("detail");

#if MESHFEM_BIND_LONG_DOUBLE
    bind<long double>(m, detail_module);
#endif
    bind<double>(m, detail_module);
}
