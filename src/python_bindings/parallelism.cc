#include <pybind11/pybind11.h>
#include <MeshFEM/Parallelism.hh>
namespace py = pybind11;

PYBIND11_MODULE(_parallelism, m) {

#ifdef MESHFEM_WITH_TBB

    m.def(  "set_max_num_tbb_threads",         &set_max_num_tbb_threads,           py::arg("num_threads"));
    m.def("unset_max_num_tbb_threads",       &unset_max_num_tbb_threads);
    m.def("set_max_num_tbb_threads",           &set_max_num_tbb_threads,           py::arg("num_threads"));
    m.def("set_gradient_assembly_num_threads", &set_gradient_assembly_num_threads, py::arg("num_threads"));
    m.def("set_hessian_assembly_num_threads",  &set_hessian_assembly_num_threads,  py::arg("num_threads"));

#endif

}
