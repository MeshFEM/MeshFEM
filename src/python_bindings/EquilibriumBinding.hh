#ifndef EQUILIBRIUMBINDING_HH
#define EQUILIBRIUMBINDING_HH
#include <MeshFEM/EquilibriumSolver.hh>

// Hack around a limitation of pybind11 where we cannot specify argument passing policies and
// pybind11 tries to make a copy if the passed instance is not already registered:
//      https://github.com/pybind/pybind11/issues/1200
// We therefore make our Python callback interface use a raw pointer to forbid this copy (which
// causes an error since NewtonProblem is not copyable).
using PyCallbackFunction = std::function<void(NewtonProblem *, size_t)>;

CallbackFunction callbackWrapper(const PyCallbackFunction &pcb) {
    return [pcb](NewtonProblem &p, size_t i) -> void { if (pcb) pcb(&p, i); };
}

template<class EQSystem, class PYEs>
void addComputeEquilibriumBinding(PYEs &pyES) {
    pyES.def("computeEquilibrium",
              [](EQSystem &sys, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, PyCallbackFunction pcb = nullptr) {
                py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
                py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
                return equilibrium_newton(sys, fixedVars, opts, callbackWrapper(pcb));
          }, py::arg("fixedVars") = std::vector<size_t>(), py::arg("opts") = NewtonOptimizerOptions(), py::arg("cb") = nullptr);
}

#endif /* end of include guard: EQUILIBRIUMBINDING_HH */
