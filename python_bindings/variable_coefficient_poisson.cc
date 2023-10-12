#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "../VariableCoefficientPoisson.hh"
#include <MeshFEM/../../python_bindings/BindingInstantiations.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

struct VPBinder {
    template<class Mesh>
    static void bind(py::module &m, py::module &detail_module) {
        using VP = VariableCoefficientPoisson<Mesh>;
        py::class_<VP>(detail_module, ("VariableCoefficientPoisson" + getMeshName<Mesh>()).c_str())
            .def_readonly("A", &VP::A)
            .def_readonly("b", &VP::b)
            ;

        using MXd = typename VP::MXd;
        using VXd = typename VP::VXd;
        using VXi = typename VP::VXi;
        m.def("construct", [&](const Mesh &m, const MXd &ks, const VXd &nodalF,
                               const VXi &neumannBoundaryElements, const VXd &neumannFluxes,
                               const VXi &dirichletNodes, const VXd &dirichletValues) {
                return std::make_unique<VP>(m, ks, nodalF, neumannBoundaryElements, neumannFluxes, dirichletNodes, dirichletValues);
            }, py::arg("m"), py::arg("ks"), py::arg("nodalF"),
            py::arg("neumannFluxes"), py::arg("neumannFluxes"),
            py::arg("dirichletNodes"), py::arg("dirichletValues"))
        ;
    }
};

PYBIND11_MODULE(variable_coefficient_poisson, m)
{
    py::module::import("MeshFEM");
    py::module::import("mesh");
    py::module::import("sparse_matrices");

    py::module detail_module = m.def_submodule("detail");

    generateMeshSpecificBindings(m, detail_module, VPBinder());
}
