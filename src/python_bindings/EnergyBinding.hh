#ifndef ENERGYBINDING_HH
#define ENERGYBINDING_HH

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <MeshFEM/EnergyDensities/StressToBiotStrain.hh>

namespace MeshFEM {

template<template<class Real_, size_t Dim> class Energy, typename... Args>
py::object constructDimensionSpecific(size_t dimension, Args... args) {
    if (dimension == 2) return py::cast(new Energy<double, 2>(std::forward<Args>(args)...), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new Energy<double, 3>(std::forward<Args>(args)...), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

template<template<class Real_, size_t Dim> class Energy, typename... Args>
py::object constructDimensionSpecificAP(size_t dimension, Args... args) {
    if (dimension == 2) return py::cast(new AutoHessianProjection<Energy<double, 2>>(std::forward<Args>(args)...), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new AutoHessianProjection<Energy<double, 3>>(std::forward<Args>(args)...), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

template<template<class Real_, size_t Dim> class Energy, typename... Args>
py::object constructDimensionSpecificConditionalAP(size_t dimension, bool autoproject, Args... args) {
    if (!autoproject) return constructDimensionSpecific<Energy>(dimension, std::forward<Args>(args)...);
    else            return constructDimensionSpecificAP<Energy>(dimension, std::forward<Args>(args)...);
}

template<typename T, typename = void>
struct has_member_projectionEnabled : std::false_type { };

template<typename T>
struct has_member_projectionEnabled<T, std::void_t<decltype(T::projectionEnabled)>> : std::true_type { };

template<typename T>
inline constexpr bool has_member_projectionEnabled_v = has_member_projectionEnabled<T>::value;

template<class Energy>
py::class_<Energy>
bindEnergyFBased(py::module &detail_module)
{
    py::class_<Energy> ebind(detail_module, getEnergyName<Energy>().c_str());
    using Mat = typename Energy::Matrix;
    ebind
        .def("setDeformationGradient", [](Energy &e, const Mat &F) { e.setDeformationGradient(F); }, py::arg("deformation_gradient"))
        .def("getDeformationGradient", &Energy::getDeformationGradient)
        .def("energy", &Energy::energy)
        .def("denergy", [](const Energy &e) { return e.denergy(); })
        .def("denergy", [](const Energy &e, const Mat &dF) { return e.denergy(dF); }, py::arg("dF"))
        .def("d2energy",       [](const Energy &e, const Mat &dF_a, const Mat &dF_b) { return e.d2energy      (dF_a, dF_b); }, py::arg("dF_a"), py::arg("dF_b"))
        .def("d2energy",       [](const Energy &e) { return evaluate_d2energy_dF2(e); }, "Get a column-major-flattened representation of Hessian psi''")
        .def("delta_denergy",  [](const Energy &e, const Mat &dF_a                 ) { return e. delta_denergy(dF_a      ); }, py::arg("dF_a"))
        .def("delta2_denergy", [](const Energy &e, const Mat &dF_a, const Mat &dF_b) { return e.delta2_denergy(dF_a, dF_b); }, py::arg("dF_a"), py::arg("dF_b"))
        .def("PK2Stress",      &Energy::PK2Stress);
    if constexpr (Energy::EDType == EDensityType::FBased) {
        ebind.def("stressToBiotStrain", [](const Energy &e, const Mat &PK1Stress, double gradTol, bool verbose) {
                        return stressToBiotStrain(e, PK1Stress, gradTol, verbose);
                    }, py::arg("PK1Stress"), py::arg("gradTol") = 1e-9, py::arg("verbose") = false, "Solve for a deformation gradient under which the material responds with a given PK1 stress")
            ;
    }

    // Bind the `projectionEnabled` boolean member if it exists.
    if constexpr (has_member_projectionEnabled_v<Energy>)
        ebind.def_readwrite("projectionEnabled", &Energy::projectionEnabled);

    return ebind;
}

template<class Energy>
py::class_<AutoHessianProjection<Energy>>
bindEnergyFBasedAutoProjected(py::module &detail_module)
{
    using HPE = AutoHessianProjection<Energy>;
    auto ebind = bindEnergyFBased<HPE>(detail_module);
    ebind.def("eigenvalues",      &HPE::eigenvalues)
         .def("eigenmatrices",    &HPE::eigenmatrices)
         .def("projectedHessian", &HPE::projectedHessian)
         .def_readwrite("projectionEnabled", &HPE::projectionEnabled)
         ;
    return ebind;
}

template<class Energy>
py::class_<Energy>
bindEnergyCBased(py::module &detail_module)
{
    py::class_<Energy> ebind(detail_module, (getEnergyName<Energy>() + "_C").c_str());
    using Mat = typename Energy::Matrix;
    ebind
        .def("setC",      &Energy::setC, py::arg("C"))
        .def("energy",    &Energy::energy)
        .def("PK2Stress", &Energy::PK2Stress)
        .def("delta_PK2Stress",  [](const Energy &e, const Mat &dC_a                ) { return e. delta_PK2Stress(dC_a      ); }, py::arg("dC_a"))
        .def("delta2_PK2Stress", [](const Energy &e, const Mat &dC_a, const Mat dC_b) { return e.delta2_PK2Stress(dC_a, dC_b); }, py::arg("dC_a"), py::arg("dC_b"))
        ;
    return ebind;
}

// Bind an isotropic energy that is parametrized by material properties.
// The underlying energy density `Psi` should have a constructor accepting
// Lamé parameters (lambda, mu).
//
// This function generates a "constructor" named `name` within module `m` that
// accepts Young's modulus and Poisson's ratio as parameters.
template<template<class Real_, size_t Dim> class Psi>
void generateEnergyBindingsYoungPoisson(const std::string &name, py::module &m, py::module &detail_module, bool planeStressFor2D = true) {
    bindEnergyFBased<Psi<double, 2>>(detail_module);
    bindEnergyFBased<Psi<double, 3>>(detail_module);

    // Isotropic Hooke's law parameter conversions.
    // When `planeStressFor2D == true` the plane stress formulas are used for 2D energies.
    // Certain energy densities may expect the 3D parameters even in 2D
    // if they implement their own plane stress/srain reductions internally
    // (e.g., minimizing over thickness strains).
    auto lambdaFromENu = [](double E, double nu, bool is3D = true) { return is3D ? (E * nu / ((1 + nu) * (1 - 2 * nu))) : ((nu * E) / (1.0 - nu * nu)); };
    auto     muFromENu = [](double E, double nu)                   { return E / (2 * (1 + nu)); };

    m.def(name.c_str(), [&](size_t dimension, double E, double nu) {                                                                     return constructDimensionSpecific<Psi>(dimension, lambdaFromENu(E, nu, !planeStressFor2D || (dimension == 3)), muFromENu(E, nu)); }, py::arg("dimension"), py::arg("E"), py::arg("nu"));
    m.def(name.c_str(), [&](py::object mesh,  double E, double nu) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructDimensionSpecific<Psi>(dimension, lambdaFromENu(E, nu, !planeStressFor2D || (dimension == 3)), muFromENu(E, nu)); }, py::arg("mesh"),      py::arg("E"), py::arg("nu"));
}

// Bind an energy that has no material parameters.
template<template<class Real_, size_t Dim> class Psi>
void generateEnergyBindingsParameterless(const std::string &name, py::module &m, py::module &detail_module) {
    bindEnergyFBased<Psi<double, 2>>(detail_module);
    bindEnergyFBased<Psi<double, 3>>(detail_module);

    m.def(name.c_str(), [](size_t dimension) {                                                                     return constructDimensionSpecific<Psi>(dimension); }, py::arg("dimension"));
    m.def(name.c_str(), [](py::object mesh ) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructDimensionSpecific<Psi>(dimension); }, py::arg("mesh")     );
}

} // namespace MeshFEM

#endif /* end of include guard: ENERGYBINDING_HH */
