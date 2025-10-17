#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include "../SimpleElasticSolid.hh"
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>

PYBIND11_MODULE(simple_elastic_solid, m)
{
    py::module::import("MeshFEM");
    py::module::import("mesh_energy");
    py::module::import("py_newton_optimizer");

    using SES = SimpleElasticSolid<3, 1, CommonNeoHookeanEnergy<double, 3>>;
    py::class_<SES, NewtonObjectiveTermBase, std::shared_ptr<SES>>(m, "SimpleElasticSolid")
        .def(py::init<const Eigen::MatrixXd &,
                      const Eigen::Matrix<int, Eigen::Dynamic, SES::NodesPerElement> &,
                      std::shared_ptr<NewtonVarsBase>>(),
             py::arg("V"), py::arg("T"), py::arg("vars"))
        ;
}
