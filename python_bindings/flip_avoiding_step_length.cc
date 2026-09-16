#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include "../FlipAvoidingStepLength.hh"

using namespace MeshFEM;

PYBIND11_MODULE(flip_avoiding_step_length, m)
{
    py::module::import("MeshFEM");
    py::module::import("py_newton_optimizer");

    py::class_<FlipAvoidingStepLength, FeasibleStepLengthComputer, std::shared_ptr<FlipAvoidingStepLength>>(m, "FlipAvoidingStepLength")
        .def(py::init<const Eigen::MatrixXi &>(), py::arg("F"))
        .def_readwrite("backoffFactor", &FlipAvoidingStepLength::backoffFactor)
        ;
}
