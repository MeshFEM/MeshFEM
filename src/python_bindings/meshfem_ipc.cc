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
#include <MeshFEM/IPCIntegration/CollisionMesh.hh>

#include "MeshBindings.hh"

using PyCM = py::class_<CollisionMesh, std::shared_ptr<CollisionMesh>>;

struct CollisionMeshBinder {
    CollisionMeshBinder(PyCM &pyCM) : m_pyCM(pyCM) { }

    template<class Mesh>
    void bind(py::module &m, py::module &/* detail_module */) {
        m_pyCM.def(py::init([](const Mesh &mesh, size_t N, bool forceLinear = false) {
            return CollisionMesh::constructForMesh<Mesh>(mesh, N, forceLinear); }),
            py::arg("mesh"),
            py::arg("embeddingDimension") = size_t(Mesh::EmbeddingSpace::RowsAtCompileTime),
            py::arg("forceLinear") = false);
    }

private:
    PyCM &m_pyCM;
};

template<typename Real_>
void bind(py::module &m, py::module &detail_module) {
    using EO = ElasticObject<Real_>;
    py::module::import("elastic_object");
    py::module::import("py_newton_optimizer");
    py::module::import("mesh");

    // Bind IPCObjective
    using IPCO = IPCObjectiveTerm<Real_>;
    py::class_<IPCO, NewtonObjectiveTermBase, std::shared_ptr<IPCO>>(detail_module, ("IPCObjectiveTerm" + floatingPointTypeSuffix<Real_>()).c_str())
        .def_readwrite("useAdaptiveBarrier",  &IPCO::useAdaptiveBarrier)
        .def("getCollisionVertexPositions",   &IPCO::getCollisionVertexPositions)
        .def("getCollisionMeshFaces",         &IPCO::getCollisionMeshFaces)
        .def("getCollisionMeshEdges",         &IPCO::getCollisionMeshEdges)
        .def_property("barrierStiffness",     &IPCO::getBarrierStiffness, &IPCO::setBarrierStiffness)
        .def_property("dhat",                 &IPCO::get_dhat, &IPCO::set_dhat)
        .def_property("ccdTol",               &IPCO::get_ccdTol, &IPCO::set_ccdTol)
        .def_property("ccdMaxIters",          &IPCO::get_ccdMaxIters, &IPCO::set_ccdMaxIters)
        .def("contactEnergy",                 &IPCO::contactPotentialEnergy)
        .def_readwrite("CCD",                 &IPCO::CCD)
        .def("contactGradient",               &IPCO::contactGradient, py::arg("includeObstVertices") = false)
        .def("initialBarrierStiffness",       [](IPCO &o, double w, const Eigen::VectorXd &ppg, double pom) { o.initialBarrierStiffness(w, ppg, pom); }, py::arg("weight"), py::arg("primaryPotentialGradient"), py::arg("primaryObjectMass"))
        .def("numCollisionConstraints",       &IPCO::numCollisionConstraints)
        .def("CCDFeasibleStepLength",         &IPCO::CCDFeasibleStepLength, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("CCDStepSize",                   &IPCO::CCDStepSize,           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def_readwrite("sparsityPatternUpdateThreshold", &IPCO::sparsityPatternUpdateThreshold)
        ;

    m.def("IPCObjectiveTerm", [](std::shared_ptr<NewtonVarsBase> vars, CollisionMesh cm, const ObstaclesCollection &obstacles, double dhat) { return std::make_shared<IPCO>(vars, cm, obstacles); }, py::arg("vars"), py::arg("collisionMesh"), py::arg("obstacles") = ObstaclesCollection(), py::arg("dhat") = 0.0, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    // Convenience method for constructing from an ElasticObject
    m.def("IPCObjectiveTerm", [](std::shared_ptr<EO> eo, const ObstaclesCollection &obstacles, double dhat) { return std::make_shared<IPCO>(eo, eo->getCollisionMesh(), obstacles); }, py::arg("eo"), py::arg("obstacles") = ObstaclesCollection(), py::arg("dhat") = 0.0, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    using MXd = Eigen::MatrixXd;
    using MXi = Eigen::MatrixXi;
    using VXd = Eigen::VectorXd;

    py::class_<Obstacle, std::shared_ptr<Obstacle>> pyObst(m, "Obstacle");
    pyObst.def(py::init<const MXd, const MXi, const MXi, const Obstacle::xFunction>(), py::arg("vertices") = MXd(), py::arg("faces") = MXi(), py::arg("edges") = MXi(), py::arg("xfunction") = Obstacle::xFunction())
          .def("getPositions",      &Obstacle::getVertices)
          .def("getForce",          &Obstacle::getForce)
          .def("getEdges",          &Obstacle::getEdges);
}

PYBIND11_MODULE(meshfem_ipc, m)
{
    m.doc() = "Bindings for MeshFEM's IPC support";

    py::module detail_module = m.def_submodule("detail");

#if MESHFEM_BIND_LONG_DOUBLE
    bind<long double>(m, detail_module);
#endif
    bind<double>(m, detail_module);

    PyCM pyCM(m, "CollisionMesh");
    pyCM.def_readwrite("edges", &CollisionMesh::edges)
        .def_readwrite("faces", &CollisionMesh::faces)
        .def_readwrite("nodeForCollisionMeshVertex", &CollisionMesh::nodeForCollisionMeshVertex)
        .def_readwrite("bbox", &CollisionMesh::bbox)
        .def_readwrite("fullModelBlockVars", &CollisionMesh::fullModelBlockVars)

        .def("numCollisionVertices", &CollisionMesh::numCollisionVertices)
        .def("extractVectorField",   &CollisionMesh::extractVectorField,   py::arg("vars"))
        ;

    generateMeshSpecificBindings(m, detail_module, CollisionMeshBinder(pyCM));
}
