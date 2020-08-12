#ifndef EQUILIBRIUMBINDING_HH
#define EQUILIBRIUMBINDING_HH
#include <MeshFEM/EquilibriumSolver.hh>

#include <pybind11/iostream.h>
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
void addComputeEquilibriumBinding(PYEs &pyES, py::module &detail_module, const std::string &objectName) {
    using EQProb = EquilibriumProblem<EQSystem>;
    using LC = LoadCollection<EQSystem>;

    pyES
        .def("computeEquilibrium",
            [](EQSystem &sys, const LC &loads, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, PyCallbackFunction pcb = nullptr) {
                return equilibrium_newton(sys, loads, fixedVars, opts, callbackWrapper(pcb));
            },
            py::arg("loads") = LC(),
            py::arg("fixedVars") = std::vector<size_t>(), py::arg("opts") = NewtonOptimizerOptions(), py::arg("cb") = nullptr,
            py::call_guard<py::scoped_ostream_redirect,
                           py::scoped_estream_redirect>())
        .def("EquilibriumProblem",
            [](EQSystem &sys, const LC &loads) {
                return std::make_unique<EQProb>(sys, loads);
            },
            py::arg("loads") = LC())
        ;

    using EQProb = EquilibriumProblem<EQSystem>;
    py::class_<EQProb, NewtonProblem>(detail_module, ("EquilibriumProblem" + objectName).c_str())
        .def("loads", &EQProb::loads, py::return_value_policy::reference)
        ;
}

#endif /* end of include guard: EQUILIBRIUMBINDING_HH */
