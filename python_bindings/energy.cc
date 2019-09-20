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
std::string
getLinearElasticEnergyName()
{
    return "LinearElasticEnergy" + std::to_string(_Dimension) + "D";
}

template<size_t _Dimension>
std::string
getNeoHookeanEnergyName()
{
    return "NeoHookeanEnergy" + std::to_string(_Dimension) + "D";
}

template<size_t _Dimension>
std::string
getElasticityTensorName()
{
    return "ElasticityTensor" + std::to_string(_Dimension) + "D";
}

template<size_t _Dimension>
void
bindElasticityTensor(py::module& module)
{
    using ETensor = ElasticityTensor<double, _Dimension>;

    py::class_<ETensor>(module, getElasticityTensorName<_Dimension>().c_str())
        .def(py::init<>())
        .def(py::init([](const std::string& material_file) {
        return Materials::Constant<_Dimension>(material_file).getTensor();
    }))
        .def("setIsotropic", &ETensor::setIsotropic, py::arg("E"), py::arg("nu"))
        .def("setOrthotropic3D",
            &ETensor::setOrthotropic3D,
            py::arg("Ex"),
            py::arg("Ey"),
            py::arg("Ez"),
            py::arg("nuYX"),
            py::arg("nuZX"),
            py::arg("nuZY"),
            py::arg("muYZ"),
            py::arg("myZX"),
            py::arg("muXY"))
        .def("setOrthotropic2D",
            &ETensor::setOrthotropic2D,
            py::arg("Ex"),
            py::arg("Ey"),
            py::arg("nuYX"),
            py::arg("muXY"));
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
bindLinearElasticEnergy(py::module& module)
{
    using LEEnergy = LinearElasticEnergy<double, _Dimension>;

    py::class_<LEEnergy> linear_elastic_energy(module,
        getLinearElasticEnergyName<_Dimension>().c_str());
    linear_elastic_energy.def(py::init<const typename LEEnergy::ETensor&>(),
        py::arg("elasticity_tensor"));

    bindEnergy(linear_elastic_energy);
}

template<size_t _Dimension>
void
bindNeoHookeanEnergy(py::module& module)
{
    py::class_<NeoHookeanEnergy<double, _Dimension>> neo_hookean_energy(
        module, getNeoHookeanEnergyName<_Dimension>().c_str());
    neo_hookean_energy.def(
        py::init<double, double, double>(),
        py::arg("first_lame_parameter"),
        py::arg("shear_modulus"),
        py::arg("finite_continuation_start") = -1);

    bindEnergy(neo_hookean_energy);
}

PYBIND11_MODULE(energy, m)
{
    py::enum_<EnergyType>(m, "EnergyType")
        .value("LINEAR", EnergyType::LINEAR)
        .value("NEO_HOOKEAN", EnergyType::NEO_HOOKEAN);

    bindElasticityTensor<2>(m);
    bindElasticityTensor<3>(m);
    bindLinearElasticEnergy<2>(m);
    bindLinearElasticEnergy<3>(m);
    bindNeoHookeanEnergy<2>(m);
    bindNeoHookeanEnergy<3>(m);
}
