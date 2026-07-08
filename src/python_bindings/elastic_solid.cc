#include "ElasticSolidBinding.hh"

using namespace MeshFEM;

PYBIND11_MODULE(elastic_solid, m)
{
    py::module detail_module = m.def_submodule("detail");

    py::module::import("mesh");
    py::module::import("energy");
    py::module::import("sparse_matrices");
    py::module::import("py_newton_optimizer");
    py::module::import("loads");
    py::module::import("elastic_object");

    generateElasticSolidBindings(m, detail_module, ElasticSolidBinder());
}
