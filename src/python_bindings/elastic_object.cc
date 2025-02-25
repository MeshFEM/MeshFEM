#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (workaround clang-tidy bug)

#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/EquilibriumSolver.hh>
#include "EquilibriumBinding.hh"

template<typename Real_>
void bind(py::module &m) {
    const std::string name = "ElasticObject" + floatingPointTypeSuffix<Real_>();

    using EO = ElasticObject<Real_>;
    using VXd = typename EO::VXd;
    using VM = VariableMask;

    py::module::import("py_newton_optimizer");
    py::module::import("block_sparse_hessian");

    py::class_<EO, NewtonVarsBase, NewtonObjectiveTermBase, std::shared_ptr<EO>> pyEO(m, name.c_str());

    py::enum_<VM>(pyEO, "VariableMask")
        .value("Defo", VM::Defo)
        .value("Rest", VM::Rest)
        .value( "All", VM::All)
        ;

    pyEO.def("setVars", [](EO &eo, const VXd &v, VM vm) { eo.setVars(v, vm); }, py::arg("vars"), py::arg("vmask") = VM::Defo)
        .def( "getVars", py::overload_cast<VM>(&EO::getVars, py::const_), py::arg("vmask") = VM::Defo)
        .def( "numVars", py::overload_cast<VM>(&EO::numVars, py::const_), py::arg("vmask") = VM::Defo)
        .def(  "energy", &EO::energy)
        .def("gradient", py::overload_cast<bool, VM>(&EO::gradient, py::const_), py::arg("updatedParametrization") = false, py::arg("vmask") = VM::Defo)

        .def("hessian", [](const EO &eo, bool projectionMask) { return eo.hessian(projectionMask); }, py::arg("projectionMask") = false)
        .def("hessianSparsityPattern",    [](const EO &eo, VM vm) { return eo.hessianSparsityPattern(vm); }, py::arg("vmask") = VM::Defo)
        .def("massMatrix",                [](const EO &eo, bool up) { return eo.massMatrix(up); }, py::arg("updatedParametrization") = false)
        .def("lumpedMass",                [](const EO &eo, bool up) { return eo.lumpedMass(up); }, py::arg("updatedParametrization") = false)
        .def("sobolevInnerProductMatrix", &EO::sobolevInnerProductMatrix, py::arg("Mscale") = 1.0)

        .def("updateParametrization",     &EO::updateParametrization)

        .def("referenceConfigSampler",    &EO::referenceConfigSampler)
        .def("deformationSamplerMatrix",  &EO::deformationSamplerMatrix, py::arg("pts"))

         .def("setIdentityDeformation",   &EO::setIdentityDeformation)

         .def("contract_d2E_dXdx",        &EO::contract_d2E_dXdx, py::arg("y"))

         .def_property("rho", &EO::getMassDensity, &EO::setMassDensity)
        ;
    addComputeEquilibriumBinding<EO>(pyEO);
}

PYBIND11_MODULE(elastic_object, m)
{
    m.doc() = "Bindings for the ElasticObject base class (not meant to be user-accessible)";
#if MESHFEM_BIND_LONG_DOUBLE
    bind<long double>(m);
#endif
    bind<double>(m);
}
