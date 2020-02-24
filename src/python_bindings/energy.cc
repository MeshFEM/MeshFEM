#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

template<typename Energy>
void
bindEnergy(py::class_<Energy>& energy_binding)
{
    energy_binding
        .def(
            "setDeformationGradient", &Energy::setDeformationGradient, py::arg("deformation_gradient"))
        .def("energy", &Energy::energy)
        .def("denergy",
            py::overload_cast<const typename Energy::Matrix&>(&Energy::denergy, py::const_),
            py::arg("dF"))
        .def("d2energy",       &Energy::d2energy,       py::arg("dF_a"), py::arg("dF_b"))
        .def("delta2_denergy", &Energy::delta2_denergy, py::arg("dF_a"), py::arg("dF_b"))
        ;
}

template<size_t _Dimension>
void
bindLinearElasticEnergy(py::module& detail_module)
{
    using LEEnergy = LinearElasticEnergy<double, _Dimension>;

    py::class_<LEEnergy> linear_elastic_energy(detail_module, getLinearElasticEnergyName<_Dimension>().c_str());
    linear_elastic_energy.def(py::init<const typename LEEnergy::ETensor&>(),
        py::arg("elasticity_tensor"));

    bindEnergy(linear_elastic_energy);
}

template<size_t _Dimension>
void
bindNeoHookeanEnergy(py::module& detail_module)
{
    py::class_<NeoHookeanEnergy<double, _Dimension>> neo_hookean_energy(
        detail_module, getNeoHookeanEnergyName<_Dimension>().c_str());
    neo_hookean_energy.def(
        py::init<double, double, double>(),
        py::arg("first_lame_parameter"),
        py::arg("shear_modulus"),
        py::arg("finite_continuation_start") = -1);

    bindEnergy(neo_hookean_energy);
}

py::object constructNeoHookean(size_t dimension, double lambda, double mu, double finiteContinuationStart) {
    if (dimension == 2) return py::cast(new NeoHookeanEnergy<double, 2>(lambda, mu, finiteContinuationStart), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new NeoHookeanEnergy<double, 3>(lambda, mu, finiteContinuationStart), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

py::object constructIsotropicLinear(size_t dimension, double young, double poisson) {
    if (dimension == 2) return py::cast(new LinearElasticEnergy<double, 2>(ElasticityTensor<double, 2>(young, poisson)), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new LinearElasticEnergy<double, 3>(ElasticityTensor<double, 3>(young, poisson)), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

PYBIND11_MODULE(energy, m)
{
    py::module detail_module = m.def_submodule("detail");
    py::module::import("tensors");

    bindLinearElasticEnergy<2>(detail_module);
    bindLinearElasticEnergy<3>(detail_module);
    bindNeoHookeanEnergy<2>   (detail_module);
    bindNeoHookeanEnergy<3>   (detail_module);

    m.def("NeoHookean",    [](size_t dimension, double lambda, double mu, double finiteContinuationStart) {                                                                     return constructNeoHookean(dimension, lambda, mu, finiteContinuationStart); }, py::arg("dimension"), py::arg("lambda"), py::arg("mu"), py::arg("finiteContinuationStart") = -1.0);
    m.def("NeoHookean",    [](py::object mesh,  double lambda, double mu, double finiteContinuationStart) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructNeoHookean(dimension, lambda, mu, finiteContinuationStart); }, py::arg("mesh"),      py::arg("lambda"), py::arg("mu"), py::arg("finiteContinuationStart") = -1.0);

    m.def("LinearElastic", [](const ElasticityTensor<double, 3> &etensor) { return LinearElasticEnergy<double, 3>(etensor); }, py::arg("elasticity_tensor"));
    m.def("LinearElastic", [](const ElasticityTensor<double, 2> &etensor) { return LinearElasticEnergy<double, 2>(etensor); }, py::arg("elasticity_tensor"));

    m.def("IsotropicLinearElastic",    [](size_t dimension, double young, double poisson) {                                                                     return constructIsotropicLinear(dimension, young, poisson); }, py::arg("dimension"), py::arg("young"), py::arg("poisson"));
    m.def("IsotropicLinearElastic",    [](py::object mesh,  double young, double poisson) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructIsotropicLinear(dimension, young, poisson); }, py::arg("mesh"),      py::arg("young"), py::arg("poisson"));
}
