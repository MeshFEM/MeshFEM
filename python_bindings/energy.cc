#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

template<size_t _Dimension>
void
bindElasticityTensor(py::module& module)
{
    using ETensor = ElasticityTensor<double, _Dimension>;

    auto py_et = py::class_<ETensor>(module, getElasticityTensorName<_Dimension>().c_str())
        .def(py::init<>())
        .def(py::init([](const std::string& material_file) {
        return Materials::Constant<_Dimension>(material_file).getTensor();
    }))
        .def("setIsotropic", &ETensor::setIsotropic, py::arg("E"), py::arg("nu"))
        ;
    if (_Dimension == 3) {
        py_et.def("setOrthotropic",
            &ETensor::setOrthotropic3D,
            py::arg("Ex"),
            py::arg("Ey"),
            py::arg("Ez"),
            py::arg("nuYX"),
            py::arg("nuZX"),
            py::arg("nuZY"),
            py::arg("muYZ"),
            py::arg("myZX"),
            py::arg("muXY"));
    }
    if (_Dimension == 2) {
        py_et.def("setOrthotropic",
            &ETensor::setOrthotropic2D,
            py::arg("Ex"),
            py::arg("Ey"),
            py::arg("nuYX"),
            py::arg("muXY"));
    }
}

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
        .def("d2energy", &Energy::d2energy, py::arg("dF_lhs"), py::arg("dF_rhs"))
        .def_property_readonly_static("type",
            [](py::object) { return EnergyTraits<Energy>::type_v; });
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

PYBIND11_MODULE(energy, m)
{
    py::enum_<EnergyType>(m, "EnergyType")
        .value("LINEAR", EnergyType::LINEAR)
        .value("NEO_HOOKEAN", EnergyType::NEO_HOOKEAN);

    py::module detail_module = m.def_submodule("detail");

    bindElasticityTensor<2>   (m);
    bindElasticityTensor<3>   (m);
    bindLinearElasticEnergy<2>(detail_module);
    bindLinearElasticEnergy<3>(detail_module);
    bindNeoHookeanEnergy<2>   (detail_module);
    bindNeoHookeanEnergy<3>   (detail_module);

    m.def("NeoHookean",    [](size_t dimension, double lambda, double mu, double finiteContinuationStart) {                                                                     return constructNeoHookean(dimension, lambda, mu, finiteContinuationStart); }, py::arg("dimension"), py::arg("lambda"), py::arg("mu"), py::arg("finiteContinuationStart") = -1.0);
    m.def("NeoHookean",    [](py::object mesh,  double lambda, double mu, double finiteContinuationStart) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructNeoHookean(dimension, lambda, mu, finiteContinuationStart); }, py::arg("mesh"),      py::arg("lambda"), py::arg("mu"), py::arg("finiteContinuationStart") = -1.0);

    m.def("LinearElastic", [](const ElasticityTensor<double, 3> &etensor) { return LinearElasticEnergy<double, 3>(etensor); }, py::arg("elasticity_tensor"));
    m.def("LinearElastic", [](const ElasticityTensor<double, 2> &etensor) { return LinearElasticEnergy<double, 2>(etensor); }, py::arg("elasticity_tensor"));

    // Isotropic linear elastic versions...
}
