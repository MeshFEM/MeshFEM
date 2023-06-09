#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

#include <MeshFEM/EquilibriumSolver.hh>
#include "EquilibriumBinding.hh"

template<typename Real_>
void bind(py::module &m) {
    const std::string name = "ElasticObject" + floatingPointTypeSuffix<Real_>();

    using EO = ElasticObject<Real_>;

    py::class_<EO, std::shared_ptr<EO>> pyEO(m, name.c_str());

    using VM = typename EO::VariableMask;
    py::enum_<VM>(pyEO, "VariableMask")
        .value("Defo", VM::Defo)
        .value("Rest", VM::Rest)
        .value( "All", VM::All)
        ;

    pyEO.def( "setVars", &EO::setVars, py::arg("vars"), py::arg("vmask") = VM::Defo)
        .def( "getVars", &EO::getVars,                  py::arg("vmask") = VM::Defo)
        .def( "numVars", &EO::numVars,                  py::arg("vmask") = VM::Defo)
        .def(  "energy", &EO::energy)
        .def("gradient", &EO::gradient, py::arg("updatedParametrization") = false, py::arg("vmask") = VM::Defo)

        .def("hessian", [](const EO &eo, bool projectionMask) { return eo.hessian(projectionMask); }, py::arg("projectionMask") = false)
        .def("hessianSparsityPattern",    &EO::hessianSparsityPattern, py::arg("val") = 0, py::arg("vmask") = VM::Defo)
        .def("massMatrix",                [](const EO &eo, bool up, bool l) { return eo.massMatrix(up, l); }, py::arg("updatedParametrization") = false, py::arg("lumped") = false)
        .def("sobolevInnerProductMatrix", &EO::sobolevInnerProductMatrix, py::arg("Mscale") = 1.0)

        .def("updateParametrization",     &EO::updateParametrization)

        .def("characteristicLength",      &EO::characteristicLength)
        .def("approxLinfVelocity",        &EO::approxLinfVelocity, py::arg("d"))

        .def("referenceConfigSampler",    &EO::referenceConfigSampler)
        .def("deformationSamplerMatrix",  &EO::deformationSamplerMatrix, py::arg("pts"))

         .def("setIdentityDeformation",   &EO::setIdentityDeformation)
        ;
    addComputeEquilibriumBinding<EO>(pyEO);

    using EQProb = EquilibriumProblem<Real_>;
    py::class_<EQProb, NewtonProblem, std::shared_ptr<EQProb>>(m, ("EquilibriumProblem" + floatingPointTypeSuffix<Real_>()).c_str())
        .def("loads", &EQProb::loads, py::return_value_policy::reference)
        ;
}

PYBIND11_MODULE(elastic_object, m)
{
    m.doc() = "Bindings for the ElasticObject base class (not meant ot be user-accessible)";
#if MESHFEM_BIND_LONG_DOUBLE
    bind<long double>(m);
#endif
    bind<double>(m);
}
