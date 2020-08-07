#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/Loads/Load.hh>

template<size_t N>
void bind(py::module &m) {
    using Load = Loads::Load<N, double>;
    py::class_<Load, std::shared_ptr<Load>>(m, ("Load" + std::to_string(N)).c_str())
        .def("energy",               &Load::energy)
        .def("deformedStateUpdated", &Load::deformedStateUpdated)
        .def("restStateUpdated",     &Load::restStateUpdated)
        .def("grad_x",               &Load::grad_x)
        .def("grad_X",               &Load::grad_X)
        .def("hessian", [](const Load &l) { auto H = l.hessianSparsityPattern(0.0); l.hessian(H); return H; })
        ;
}

PYBIND11_MODULE(loads, m)
{
    bind<2>(m);
    bind<3>(m);
}
