#ifndef EQUILIBRIUMBINDING_HH
#define EQUILIBRIUMBINDING_HH
#include <MeshFEM/EquilibriumSolver.hh>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

namespace MeshFEM {

// Hack around a limitation of pybind11 where we cannot specify argument passing policies and
// pybind11 tries to make a copy if the passed instance is not already registered:
//      https://github.com/pybind/pybind11/issues/1200
// We therefore make our Python callback interface use a raw pointer to forbid this copy (which
// causes an error since NewtonProblem is not copyable).
using PyCallbackFunction = std::function<bool(NewtonProblem *, size_t)>;

inline NewtonMultiobjectiveProblem::CallbackFunction callbackWrapper(const PyCallbackFunction &pcb) {
    return [pcb](NewtonProblem &p, size_t i) -> bool { if (pcb) return pcb(&p, i); return false; };
}

template<class EQObj, class PYEs>
void addComputeEquilibriumBinding(PYEs &pyES) {
    using Real = typename EQObj::Real;
    using LC   = LoadCollection<Real>;

    pyES
        .def("computeEquilibrium",
            [](std::shared_ptr<EQObj> obj, const LC &loads, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, PyCallbackFunction pcb, Real systemEnergyIncreaseFactorLimit, Real energyLimitingThreshold, Real hessianShift) {
                return equilibrium_newton(obj, loads, fixedVars, opts, callbackWrapper(pcb), systemEnergyIncreaseFactorLimit, energyLimitingThreshold, hessianShift);
            },
            py::arg("loads") = LC(),
            py::arg("fixedVars") = std::vector<size_t>(), py::arg("opts") = NewtonOptimizerOptions(), py::arg("cb") = nullptr,
            py::arg("systemEnergyIncreaseFactorLimit") = safe_numeric_limits<Real>::max(), py::arg("energyLimitingThreshold") = 1e-6,
            py::arg("hessianShift") = 0.0,
            py::call_guard<py::scoped_ostream_redirect,
                           py::scoped_estream_redirect>())
        .def("equilibriumOptimizer",
            [](std::shared_ptr<EQObj> obj, const LC &loads, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, PyCallbackFunction pcb, Real systemEnergyIncreaseFactorLimit, Real energyLimitingThreshold, Real hessianShift) -> std::shared_ptr<NewtonOptimizer> {
                return get_equilibrium_optimizer(obj, loads, fixedVars, opts, callbackWrapper(pcb), systemEnergyIncreaseFactorLimit, energyLimitingThreshold, hessianShift);
            },
            py::arg("loads") = LC(),
            py::arg("fixedVars") = std::vector<size_t>(), py::arg("opts") = NewtonOptimizerOptions(), py::arg("cb") = nullptr,
            py::arg("systemEnergyIncreaseFactorLimit") = safe_numeric_limits<Real>::max(), py::arg("energyLimitingThreshold") = 1e-6,
            py::arg("hessianShift") = 0.0,
            py::call_guard<py::scoped_ostream_redirect,
                           py::scoped_estream_redirect>())
        .def("EquilibriumProblem",
            [](std::shared_ptr<EQObj> obj, const LC &loads) -> std::shared_ptr<NewtonMultiobjectiveProblem> {
                return get_equilibrium_problem(obj, loads);
            },
            py::arg("loads") = LC())
        ;
}

} // namespace MeshFEM

#endif /* end of include guard: EQUILIBRIUMBINDING_HH */
