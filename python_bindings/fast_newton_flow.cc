#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include <MeshFEM/EnergyDensities/SymmetricDirichlet.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>

#include "../FastNewtonFlow.hh"
#include "../RigidMotionFactorization.hh"

using namespace MeshFEM;

template<class NFME, class Factorization, class Class>
void bindFlowSolver(Class &cls) {
    cls
        .def("upgradeToDegree", &NFME::template upgradeToDegree<Factorization>, py::arg("hessianFactorization"), py::arg("targetDegree"))
        .def("computeTaylorCoefficients", py::overload_cast<const Factorization &, const typename NFME::VXd &, int, bool, bool>(&NFME::template computeTaylorCoefficients<Factorization>), py::arg("hessianFactorization"), py::arg("x1"), py::arg("degree") = 6, py::arg("arclen") = false, py::arg("projectHessian") = false)
        .def("computeTaylorCoefficients", py::overload_cast<const Factorization &, const typename NFME::VXd &, int, NewtonFlowParameterization, bool>(&NFME::template computeTaylorCoefficients<Factorization>),
             py::arg("hessianFactorization"), py::arg("x1"), py::arg("degree"), py::arg("parameterization"), py::arg("projectHessian") = false)
        ;
}

template<size_t Dim, size_t FEMDeg>
auto bindFastNewtonFlow(const std::string &name, py::module &m, py::module &detail) {
    using NFME = FastNewtonFlowMeshEnergy<Dim, FEMDeg>;
    auto cls = bindMeshEnergy<NFME>(name, m, detail)

        .def("setNDPartition", &NFME::setNDPartition)
        .def("clearNDPartition", &NFME::clearNDPartition)
        .def_property_readonly("hasNDPartition", &NFME::hasNDPartition)
        .def("refreshGeometryCache", &NFME::refreshGeometryCache)
        .def_property("eigenvalueClampTarget", &NFME::getEigenvalueClampTarget, &NFME::setEigenvalueClampTarget,
                      "Hessian eigenvalue clamp target. Changing it invalidates Taylor coefficients; reinitialize before upgrading.")
        .def_property_readonly("automaticProjectionMask", &NFME::automaticProjectionMask,
                               "Fresh boolean mask: minimum element energy-density Hessian eigenvalue < eigenvalueClampTarget at the current configuration. Ignores manual mask overrides.")
        .def("initCoefficients", py::overload_cast<const typename NFME::VXd &, bool, bool>(&NFME::initCoefficients), py::arg("d"), py::arg("arclen") = false, py::arg("projectHessian") = false)
        .def("initCoefficients", py::overload_cast<const typename NFME::VXd &, NewtonFlowParameterization, bool>(&NFME::initCoefficients),
             py::arg("d"), py::arg("parameterization"), py::arg("projectHessian") = false)
        .def("getCoefficient", []( NFME &me, int d) -> py::array {
            const auto &xd = me.getCoefficient(d);
            return py::array(xd.size(), xd.data());
        }, py::arg("d"), "Get the degree-d Taylor coefficient as a numpy array (note: this is a view into the internal storage of the energy, not a copy)")
        .def_property_readonly("lambdaCoefficients", &NFME::getLambdaCoefficients,
                               "Scalar Taylor coefficients (lambda for ConstantSpeed, its reciprocal for ConstantSpeedReciprocal), starting with 1. Empty for Native/GradientProgress or invalid expansions; degree-d positions give d scalar coefficients.")
        .def_readwrite("pk1ChunkSize", &NFME::pk1ChunkSize, "PK1/arclength graph chunk size; 0 selects degree-dependent automatic tuning")
        .def_readwrite("projectionChunkSize", &NFME::projectionChunkSize, "Projection graph chunk size; 0 selects automatic tuning for the projected-element count")
        .def_readonly("neg_delta_g", &NFME::neg_delta_g)
        ;
    bindFlowSolver<NFME, NewtonHessianFactorization>(cls);
    bindFlowSolver<NFME, RigidMotionFactorization>(cls);
    return cls;
}

PYBIND11_MODULE(fast_newton_flow, m)
{
    py::module::import("sparse_matrices");
    py::module::import("mesh_energy");
    py::module::import("py_newton_optimizer");
    py::module::import("rotation_strain_extrapolation");
    py::module detail = m.def_submodule("detail");

    py::enum_<NewtonFlowParameterization>(m, "Parameterization")
        .value("Native", NewtonFlowParameterization::Native)
        .value("ConstantSpeed", NewtonFlowParameterization::ConstantSpeed)
        .value("ConstantSpeedReciprocal", NewtonFlowParameterization::ConstantSpeedReciprocal)
        .value("GradientProgress", NewtonFlowParameterization::GradientProgress);

    py::enum_<RigidMotionConstraints>(m, "RigidMotionConstraints")
        .value("Translations", RigidMotionConstraints::Translations)
        .value("All", RigidMotionConstraints::All);
    py::class_<RigidMotionFactorization>(m, "RigidMotionFactorization",
        "Distributed planar rigid constraints, frozen at setGeometry. Requires an unshifted sparse Hessian and a positive pinned principal block.")
        .def(py::init<const Eigen::VectorXd &, RigidMotionConstraints, CholeskyProvider>(),
             py::arg("x"), py::arg("constraints"), py::arg("provider") = CholeskyProvider::CatamariNesdisParallel)
        .def("setGeometry", &RigidMotionFactorization::setGeometry)
        .def("factorizeSymbolic", &RigidMotionFactorization::factorizeSymbolic)
        .def("factorizeNumeric", &RigidMotionFactorization::factorizeNumeric)
        .def("solve", [](const RigidMotionFactorization &f, const Eigen::VectorXd &b) {
            Eigen::VectorXd x; f.solve(b, x); return x;
        });

    m.attr("ElementPartitionFromND") = py::module::import("sparse_matrices").attr("ElementPartitionFromND");
    bindFastNewtonFlow<2, 1>("symmetric_dirichlet", m, detail);
}
