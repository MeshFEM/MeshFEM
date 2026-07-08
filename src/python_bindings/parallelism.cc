#include <pybind11/pybind11.h>
#include <MeshFEMCore/Parallelism.hh>

using namespace MeshFEM;
namespace py = pybind11;

#if MESHFEM_WITH_TBB

PYBIND11_MODULE(_parallelism, m) {
    m.def("unset_max_num_tbb_threads",       &unset_max_num_tbb_threads);
    m.def("set_max_num_tbb_threads",           &set_max_num_tbb_threads,           py::arg("num_threads"));
    m.def("set_gradient_assembly_num_threads", &set_gradient_assembly_num_threads, py::arg("num_threads"));
    m.def("set_hessian_assembly_num_threads",  &set_hessian_assembly_num_threads,  py::arg("num_threads"));

    py::class_<PinningObserver>(m, "PinningObserver")
        .def(py::init<bool>(), py::arg("spread") = true);
    ;
}

#endif
