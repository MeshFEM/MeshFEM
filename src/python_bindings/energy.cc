#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#include <MeshFEM/EnergyDensities/AutodiffEDensity.hh>
#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/IsoCRLEFixed.hh>
#include <MeshFEM/EnergyDensities/IsoCRLEWithHessianProjection.hh>
#include <MeshFEM/EnergyDensities/IsoCRLETensionFieldMembrane.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/EnergyDensities/TensionFieldTheory.hh>
#include <MeshFEM/EnergyDensities/TensionFieldNeoHookean.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>
#include <MeshFEM/EnergyDensities/StressToBiotStrain.hh>
#include <MeshFEM/EnergyDensities/MetricFitting.hh>
#include <MeshFEM/EnergyDensities/CollapsePreventionEnergy.hh>
#include <MeshFEM/EnergyDensities/SymmetricDirichlet.hh>
#include <MeshFEM/EnergyDensities/IsotropicAutodiffEDensity.hh>

#include "EnergyBinding.hh"

using namespace MeshFEM;

template<typename Real_, size_t Dim_>
using CommonNeoHookeanAD = AutodiffEDensity<CommonNeoHookeanPsi, Real_, Dim_>;

template<class WSP>
py::class_<WSP>
bindWrinkleStrainProblem(py::module &m, const std::string &name) {
    auto wsp = py::class_<WSP>(m, name.c_str())
        .def(py::init<typename WSP::Psi &, const typename WSP::M2d &, const Vec2_T<typename WSP::Real> &>(), py::arg("psi"), py::arg("C"), py::arg("n"))
        .def("getC",     &WSP::getC)
        .def("setC",     &WSP::setC, py::arg("C"))
        .def("numVars",  &WSP::numVars)
        .def("getVars",  &WSP::getVars)
        .def("setVars",  &WSP::setVars, py::arg("vars"))
        .def("energy",   &WSP::energy)
        .def("gradient", &WSP::gradient)
        .def("hessian",  &WSP::hessian)
        .def("solve",    &WSP::solve)
        ;
    return wsp;
}

template<size_t _Dimension>
void bindLinearElasticEnergy(py::module &detail_module)
{
    using LEEnergy = LinearElasticEnergy<double, _Dimension>;
    auto ebind = bindEnergyFBased<LEEnergy>(detail_module);
    ebind.def(py::init<const typename LEEnergy::ETensor&>(), py::arg("elasticity_tensor"));
}

template<size_t _Dimension>
void bindCRLinearElasticEnergy(py::module &detail_module)
{
    using CRLE = CorotatedLinearElasticity<double, _Dimension>;
    using Mat  = typename CRLE::Matrix;
    auto ebind = bindEnergyFBased<CRLE>(detail_module);
    ebind.def(py::init<const typename CRLE::ETensor&>(), py::arg("elasticity_tensor"))
         .def("R",     &CRLE::R)
         .def("S",     &CRLE::S)
         .def("sigma", &CRLE::biotStress)
         .def("delta_R",     [](const CRLE &cr, const Mat &dF) { return cr.delta_R(dF);                                 }, py::arg("dF"))
         .def("delta_S",     [](const CRLE &cr, const Mat &dF) { return cr.delta_S(dF, cr.delta_R(dF));                 }, py::arg("dF"))
         .def("delta_sigma", [](const CRLE &cr, const Mat &dF) { return cr.delta_sigma(cr.delta_S(dF, cr.delta_R(dF))); }, py::arg("dF"))
         .def("isIsotropic", &CRLE::isIsotropic)
         ;
}

template<size_t _Dimension>
void bindIsoCRLEWithHP(py::module &detail_module)
{
    using CRLE = IsoCRLEWithHessianProjection<double, _Dimension>;
    auto ebind = bindEnergyFBased<CRLE>(detail_module);
    ebind.def(py::init<double, double>(), py::arg("lambda"), py::arg("mu"))
         .def("R",     &CRLE::R)
         .def("S",     &CRLE::S)
         .def("sigma", &CRLE::biotStress)
         .def_readwrite("projectionEnabled", &CRLE::projectionEnabled)
         ;
}

template<size_t _Dimension>
void bindIsoCRLEFixed(py::module &detail_module)
{
    using CRLE = IsoCRLEFixed<double, _Dimension>;
    auto ebind = bindEnergyFBased<CRLE>(detail_module);
    ebind.def(py::init<double, double>(), py::arg("lambda"), py::arg("mu"))
         .def("R",     &CRLE::R)
         .def("S",     &CRLE::S)
         .def("sigma", &CRLE::biotStress)
         .def_readwrite("projectionEnabled", &CRLE::projectionEnabled)
         ;
}

template<size_t _Dimension>
void bindNeoHookeanEnergy(py::module& detail_module)
{
    auto ebind = bindEnergyFBased<NeoHookeanEnergy<double, _Dimension>>(detail_module);
    ebind.def(py::init<double, double>(), py::arg("lambda"), py::arg("mu"));
}

template<size_t _Dimension>
void bindCommonNeoHookeanEnergy(py::module& detail_module)
{
    bindEnergyFBased<CommonNeoHookeanEnergy<double, _Dimension>>(detail_module)
        .def(py::init<double, double>(), py::arg("lambda"), py::arg("mu"));
    // bindEnergyFBasedAutoProjected<CommonNeoHookeanEnergy<double, _Dimension>>(detail_module)
    //     .def(py::init<double, double>(), py::arg("lambda"), py::arg("mu"));
}

template<size_t _Dimension>
void bindNeoHookeanEnergyHP(py::module& detail_module)
{
    auto ebind = bindEnergyFBasedAutoProjected<NeoHookeanEnergy<double, _Dimension>>(detail_module);
    ebind.def(py::init<double, double>(), py::arg("lambda"), py::arg("mu"));
}

template<size_t _Dimension>
void bindStVKEnergy(py::module &detail_module)
{
    using STVK = StVenantKirchhoffEnergy<double, _Dimension>;
    auto ebind = bindEnergyFBased<STVK>(detail_module);
    ebind.def(py::init<const typename STVK::ETensor&>(), py::arg("elasticity_tensor"));
}

template<size_t _Dimension>
void bindStVKEnergyHP(py::module &detail_module)
{
    using STVK = StVenantKirchhoffEnergy<double, _Dimension>;
    auto ebind = bindEnergyFBasedAutoProjected<STVK>(detail_module);
    ebind.def(py::init<const typename STVK::ETensor&>(), py::arg("elasticity_tensor"));
}

py::object constructIsotropicLinear(size_t dimension, double young, double poisson) {
    if (dimension == 2) return py::cast(new LinearElasticEnergy<double, 2>(ElasticityTensor<double, 2>(young, poisson)), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new LinearElasticEnergy<double, 3>(ElasticityTensor<double, 3>(young, poisson)), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

py::object constructIsotropicCorotated(size_t dimension, double young, double poisson) {
    if (dimension == 2) return py::cast(new CorotatedLinearElasticity<double, 2>(ElasticityTensor<double, 2>(young, poisson), true), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new CorotatedLinearElasticity<double, 3>(ElasticityTensor<double, 3>(young, poisson), true), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

py::object constructIsoCRLEHessProj(size_t dimension, double lambda, double mu) {
    if (dimension == 2) return py::cast(new IsoCRLEWithHessianProjection<double, 2>(lambda, mu), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new IsoCRLEWithHessianProjection<double, 3>(lambda, mu), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

py::object constructIsoCRLEfixed(size_t dimension, double lambda, double mu) {
    if (dimension == 2) return py::cast(new IsoCRLEFixed<double, 2>(lambda, mu), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new IsoCRLEFixed<double, 3>(lambda, mu), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

py::object constructIsotropicStVK(size_t dimension, double young, double poisson) {
    if (dimension == 2) return py::cast(new StVenantKirchhoffEnergy<double, 2>(ElasticityTensor<double, 2>(young, poisson)), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new StVenantKirchhoffEnergy<double, 3>(ElasticityTensor<double, 3>(young, poisson)), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

py::object constructSymmetricDirichlet(size_t dimension) {
    if (dimension == 2) return py::cast(new SymmetricDirichlet<double, 2>(), py::return_value_policy::take_ownership);
    if (dimension == 3) return py::cast(new SymmetricDirichlet<double, 3>(), py::return_value_policy::take_ownership);
    throw std::runtime_error("Argument 'dimension' must be 2 or 3");
}

PYBIND11_MODULE(energy, m)
{
    py::module detail_module = m.def_submodule("detail");
    py::module::import("tensors");

    bindLinearElasticEnergy<2>  (detail_module);
    bindLinearElasticEnergy<3>  (detail_module);
    bindNeoHookeanEnergy<2>     (detail_module);
    bindNeoHookeanEnergy<3>     (detail_module);
    bindCommonNeoHookeanEnergy<2>(detail_module);
    bindCommonNeoHookeanEnergy<3>(detail_module);
    bindNeoHookeanEnergyHP<2>   (detail_module);
    bindNeoHookeanEnergyHP<3>   (detail_module);
    bindCRLinearElasticEnergy<2>(detail_module);
    bindCRLinearElasticEnergy<3>(detail_module);
    bindStVKEnergy<2>           (detail_module);
    bindStVKEnergy<3>           (detail_module);
    bindIsoCRLEWithHP<2>        (detail_module);
    bindIsoCRLEWithHP<3>        (detail_module);
    bindIsoCRLEFixed<2>         (detail_module);
    bindIsoCRLEFixed<3>         (detail_module);
    bindStVKEnergyHP<2>         (detail_module);
    bindStVKEnergyHP<3>         (detail_module);

    using ETensor2D = ElasticityTensor<double, 2>;
    using ETensor3D = ElasticityTensor<double, 3>;

    bindEnergyFBased<StVenantKirchhoffMembraneEnergy<double>>(detail_module)
        .def(py::init<const ETensor2D &>(), py::arg("elasticity_tensor"))
        ;
    using NeoHookeanMembrane = EnergyDensityFBasedMembraneFromFBased<NeoHookeanEnergy<double, 2>>;
    bindEnergyFBased<NeoHookeanMembrane>(detail_module)
        .def(py::init<double, double>(), py::arg("lambda"), py::arg("mu"));
        ;

    using IsoCRLETensionFieldMembrane = IsoCRLETensionFieldMembrane<double>;
    bindEnergyFBased<IsoCRLETensionFieldMembrane>(m)
        .def(py::init<double, double>(), py::arg("E"), py::arg("nu"))
        .def("c",        &IsoCRLETensionFieldMembrane::c,        py::arg("x"))
        .def("dc_dx",    &IsoCRLETensionFieldMembrane::dc_dx,    py::arg("x"))
        .def("d2c_dx2",  &IsoCRLETensionFieldMembrane::d2c_dx2,  py::arg("x"))
        .def("dc_de",    &IsoCRLETensionFieldMembrane::dc_de,    py::arg("x"))
        .def("d2c_dxde", &IsoCRLETensionFieldMembrane::d2c_dxde, py::arg("x"))
        .def("d2c_de2",  &IsoCRLETensionFieldMembrane::d2c_de2,  py::arg("x"))
        .def("unrelaxed_delta_denergy_undeformed", [](const IsoCRLETensionFieldMembrane &psi, const typename IsoCRLETensionFieldMembrane::M32d &dF) { return psi.unrelaxed_delta_denergy_undeformed(dF); }, py::arg("dF"))
        .def_readwrite("relaxationEnabled",        &IsoCRLETensionFieldMembrane::relaxationEnabled)
        .def_readwrite("smoothingEnabled",         &IsoCRLETensionFieldMembrane::smoothingEnabled)
        .def_readwrite("smoothingEps",             &IsoCRLETensionFieldMembrane::smoothingEps)
        .def_readwrite("relaxedStiffnessEpsilon",  &IsoCRLETensionFieldMembrane::relaxedStiffnessEps)
        .def_readwrite("hessianProjectionEnabled", &IsoCRLETensionFieldMembrane::hessianProjectionEnabled)
        .def_property_readonly("U",                &IsoCRLETensionFieldMembrane::U)
        .def_property_readonly("V",                &IsoCRLETensionFieldMembrane::V)
        .def_property_readonly("principalStrains", &IsoCRLETensionFieldMembrane::principalStrains)
        .def("principalBiotStrains", &IsoCRLETensionFieldMembrane::principalStrains)
        .def("tensionState", &IsoCRLETensionFieldMembrane::tensionState)
        ;

    using StVK_TFT = EnergyDensityFBasedFromCBased<RelaxedEnergyDensity<StVenantKirchhoffEnergyCBased<double, 2>>, 3>;
    bindEnergyFBased<StVK_TFT>(detail_module)
        .def(py::init<const ETensor2D &>(), py::arg("elasticity_tensor"))
        .def("tensionState", &StVK_TFT::tensionState)
        .def("principalBiotStrains", &StVK_TFT::principalBiotStrains)
        .def("psi", py::overload_cast<>(&StVK_TFT::psi), py::return_value_policy::reference)
        .def_property("relaxationEnabled", &StVK_TFT::relaxationEnabled, &StVK_TFT::setRelaxationEnabled)
        ;

    using INeo_TFT = EnergyDensityFBasedFromCBased<RelaxedEnergyDensity<IncompressibleNeoHookeanEnergyCBased<double>>, 3>;
    bindEnergyFBased<INeo_TFT>(detail_module)
        .def(py::init<double>(), py::arg("young_modulus"))
        .def("tensionState", &INeo_TFT::tensionState)
        .def("principalBiotStrains", &INeo_TFT::principalBiotStrains)
        .def("psi", py::overload_cast<>(&INeo_TFT::psi), py::return_value_policy::reference)
        .def_property("relaxationEnabled", &INeo_TFT::relaxationEnabled, &INeo_TFT::setRelaxationEnabled)
        ;

    using OTFE = OptionalTensionFieldEnergy<double>;
    using OTFE_M2d = OTFE::M2d;
    bindEnergyCBased<OTFE>(detail_module)
        .def("tensionState", & OTFE::tensionState)
        .def_property("relaxationEnabled", &OTFE::getRelaxationEnabled, &OTFE::setRelaxationEnabled)

        .def( "denergy",         py::overload_cast<const OTFE_M2d &>(&OTFE::denergy_dC, py::const_), py::arg("dC"))
        .def( "denergy",         py::overload_cast<                >(&OTFE::denergy_dC, py::const_))
        .def("d2energy",         &OTFE::d2energy_dC)
        .def("eigSensitivities", &OTFE::eigSensitivities)
        .def("setMatrix", [](OTFE &tfe, OTFE_M2d &mat) { tfe.setMatrix(mat); })

        .def("setEigs",   &OTFE::setEigs,   py::arg("l1"), py::arg("l2"))
        .def("psi",       &OTFE::psi)
        .def("dpsi_dl",   &OTFE::dpsi_dl)
        .def("d2psi_dl2", &OTFE::d2psi_dl2)
        .def_readwrite("useTensionField", &OTFE::useTensionField)
        ;

    using StVK_C = StVenantKirchhoffEnergyCBased<double, 2>;
    bindEnergyCBased<StVK_C>(detail_module)
        .def(py::init<const ETensor2D &>(), py::arg("elasticity_tensor"))
        .def_property("elasticityTensor", &StVK_C::elasticityTensor, &StVK_C::setElasticityTensor)
        ;

    using INeo_C = IncompressibleNeoHookeanEnergyCBased<double>;
    bindEnergyCBased<INeo_C>(detail_module)
        .def(py::init<double>(), py::arg("young_moduls"))
        .def_property("youngModulus", &INeo_C::youngModulus, &INeo_C::setYoungModulus)
        .def_readwrite("stiffness",   &INeo_C::stiffness)
        ;

    using MFE = MetricFittingEnergy<double, 2>;
    bindEnergyCBased<MFE>(detail_module)
        .def_readwrite("targetMetric",          &MFE::targetMetric)
        .def_property_readonly("currentMetric", &MFE::currentMetric)
        ;

    using CPE = CollapsePreventionEnergyDet<double, 2>;
    bindEnergyCBased<CPE>(detail_module)
        .def("det", &CPE::det)
        .def_property("activationThreshold", &CPE::activationThreshold, &CPE::setActivationThreshold)
        ;

    bindWrinkleStrainProblem<  IsotropicWrinkleStrainProblem<StVenantKirchhoffEnergyCBased<double, 2>>>(m,   "IsotropicWrinkleStrainProblem");
    bindWrinkleStrainProblem<AnisotropicWrinkleStrainProblem<StVenantKirchhoffEnergyCBased<double, 2>>>(m, "AnisotropicWrinkleStrainProblem");

    bindWrinkleStrainProblem<  IsotropicWrinkleStrainProblem<INeo_C>>(m,   "IsotropicWrinkleStrainProblemINeo");
    bindWrinkleStrainProblem<AnisotropicWrinkleStrainProblem<INeo_C>>(m, "AnisotropicWrinkleStrainProblemINeo");

    m.def("NeoHookean",    [](size_t dimension, double lambda, double mu, bool autoproject) {                                                                     return constructDimensionSpecificConditionalAP<NeoHookeanEnergy>(dimension, autoproject, lambda, mu); }, py::arg("dimension"), py::arg("lambda"), py::arg("mu"), py::arg("autoproject") = false);
    m.def("NeoHookean",    [](py::object mesh,  double lambda, double mu, bool autoproject) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructDimensionSpecificConditionalAP<NeoHookeanEnergy>(dimension, autoproject, lambda, mu); }, py::arg("mesh"),      py::arg("lambda"), py::arg("mu"), py::arg("autoproject") = false);
    m.def("NeoHookeanMembrane", [](double lambda, double mu) { return std::make_unique<NeoHookeanMembrane>(lambda, mu); }, py::arg("lambda"), py::arg("mu"));
    m.def("LinearElastic",             [](const ETensor3D &etensor) { return std::make_unique<LinearElasticEnergy          <double, 3>>(etensor); }, py::arg("elasticity_tensor"));
    m.def("LinearElastic",             [](const ETensor2D &etensor) { return std::make_unique<LinearElasticEnergy          <double, 2>>(etensor); }, py::arg("elasticity_tensor"));
    m.def("CorotatedLinearElastic",    [](const ETensor3D &etensor) { return std::make_unique<CorotatedLinearElasticity    <double, 3>>(etensor); }, py::arg("elasticity_tensor"));
    m.def("CorotatedLinearElastic",    [](const ETensor2D &etensor) { return std::make_unique<CorotatedLinearElasticity    <double, 2>>(etensor); }, py::arg("elasticity_tensor"));
    m.def("StVenantKirchhoff",         [](const ETensor3D &etensor) { return std::make_unique<StVenantKirchhoffEnergy      <double, 3>>(etensor); }, py::arg("elasticity_tensor"));
    m.def("StVenantKirchhoff",         [](const ETensor2D &etensor) { return std::make_unique<StVenantKirchhoffEnergy      <double, 2>>(etensor); }, py::arg("elasticity_tensor"));
    m.def("StVenantKirchhoffMembrane", [](const ETensor2D &etensor) { return std::make_unique<StVenantKirchhoffMembraneEnergy <double>>(etensor); }, py::arg("elasticity_tensor"));
    m.def("StVenantKirchhoffCBased",   [](const ETensor2D &etensor) { return std::make_unique<StVenantKirchhoffEnergyCBased<double, 2>>(etensor); }, py::arg("elasticity_tensor"));

    m.def("IsotropicLinearElastic", [](size_t dimension, double young, double poisson) {                                                                     return constructDimensionSpecific<LinearElasticEnergy>(dimension, young, poisson); }, py::arg("dimension"), py::arg("young"), py::arg("poisson"));
    m.def("IsotropicLinearElastic", [](py::object mesh,  double young, double poisson) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructDimensionSpecific<LinearElasticEnergy>(dimension, young, poisson); }, py::arg("mesh"),      py::arg("young"), py::arg("poisson"));

    m.def("CorotatedIsotropicLinearElastic", [](size_t dimension, double young, double poisson) {                                                                     return constructIsotropicCorotated(dimension, young, poisson); }, py::arg("dimension"), py::arg("young"), py::arg("poisson"));
    m.def("CorotatedIsotropicLinearElastic", [](py::object mesh,  double young, double poisson) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructIsotropicCorotated(dimension, young, poisson); }, py::arg("mesh"),      py::arg("young"), py::arg("poisson"));

    m.def("IsotropicStVenantKirchhoff",         [](size_t dimension, double young, double poisson) {                                                                     return constructIsotropicStVK(dimension, young, poisson); }, py::arg("dimension"), py::arg("young"), py::arg("poisson"));
    m.def("IsotropicStVenantKirchhoff",         [](py::object mesh,  double young, double poisson) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructIsotropicStVK(dimension, young, poisson); }, py::arg("mesh"),      py::arg("young"), py::arg("poisson"));
    m.def("IsotropicStVenantKirchhoffMembrane", [](double young, double poisson) { return std::make_unique<StVenantKirchhoffMembraneEnergy<double>>(ETensor2D(young, poisson)); }, py::arg("young"), py::arg("poisson"));

    m.def("RelaxedStVenantKirchhoffMembrane",          [](const ETensor2D &etensor)     { return std::make_unique<StVK_TFT>(etensor); }, py::arg("elasticity_tensor"));
    m.def("RelaxedIsotropicStVenantKirchhoffMembrane", [](double young, double poisson) { return std::make_unique<StVK_TFT>(ETensor2D(young, poisson)); }, py::arg("young"), py::arg("poisson"));
    m.def("RelaxedIncompressibleNeoHookeanMembrane",   [](double young)                 { return std::make_unique<INeo_TFT>(young); }, py::arg("young"));

    // Note: these expressions are for volumetric elasticity.
    // Note that in the 2D case, plane stress conditions are applied inside the
    // NeoHookeanEnergyBase class, so it is correct to pass the volumetric Lame
    // parameters to it in both cases.
    // This is why "is3D" defaults to true...
    auto lambdaFromENu = [](double E, double nu, bool is3D = true) { return is3D ? (E * nu / ((1 + nu) * (1 - 2 * nu))) : ((nu * E) / (1.0 - nu * nu)); };
    auto     muFromENu = [](double E, double nu)                   { return E / (2 * (1 + nu)); };

    // Convenience methods for constructing a neo-Hookean material from a Young's modulus Poisson's ratio
    m.def("NeoHookeanYoungPoisson",         [&](size_t dimension, double E, double nu, bool autoproject) {                                                                     return constructDimensionSpecificConditionalAP<NeoHookeanEnergy>(dimension, autoproject, lambdaFromENu(E, nu), muFromENu(E, nu)); }, py::arg("dimension"), py::arg("E"), py::arg("nu"), py::arg("autoproject") = false);
    m.def("NeoHookeanYoungPoisson",         [&](py::object mesh,  double E, double nu, bool autoproject) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructDimensionSpecificConditionalAP<NeoHookeanEnergy>(dimension, autoproject, lambdaFromENu(E, nu), muFromENu(E, nu)); }, py::arg("mesh"),      py::arg("E"), py::arg("nu"), py::arg("autoproject") = false);
    m.def("NeoHookeanMembraneYoungPoisson", [&](                  double E, double nu) {                                                                     return std::make_unique<NeoHookeanMembrane>( lambdaFromENu(E, nu), muFromENu(E, nu)); },                       py::arg("E"), py::arg("nu"));

    m.def("CommonNeoHookeanYoungPoisson",   [&](size_t dimension, double E, double nu) {                                                                     return constructDimensionSpecific<CommonNeoHookeanEnergy>(dimension, lambdaFromENu(E, nu, (dimension == 3)), muFromENu(E, nu)); }, py::arg("dimension"), py::arg("E"), py::arg("nu"));
    m.def("CommonNeoHookeanYoungPoisson",   [&](py::object mesh,  double E, double nu) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructDimensionSpecific<CommonNeoHookeanEnergy>(dimension, lambdaFromENu(E, nu, (dimension == 3)), muFromENu(E, nu)); }, py::arg("mesh"),      py::arg("E"), py::arg("nu"));

    m.def("IsoCRLEWithHessianProjection",   [&](size_t dimension, double E, double nu) {                                                                     return constructIsoCRLEHessProj(dimension, lambdaFromENu(E, nu, dimension == 3), muFromENu(E, nu)); }, py::arg("dimension"), py::arg("young"), py::arg("poisson"));
    m.def("IsoCRLEWithHessianProjection",   [&](py::object mesh,  double E, double nu) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructIsoCRLEHessProj(dimension, lambdaFromENu(E, nu, dimension == 3), muFromENu(E, nu)); }, py::arg("mesh"),      py::arg("young"), py::arg("poisson"));
    m.def("IsoCRLEFixed",   [&](size_t dimension, double E, double nu) {                                                                     return constructIsoCRLEfixed(dimension, lambdaFromENu(E, nu, dimension == 3), muFromENu(E, nu)); }, py::arg("dimension"), py::arg("young"), py::arg("poisson"));
    m.def("IsoCRLEFixed",   [&](py::object mesh,  double E, double nu) { size_t dimension = py::cast<double>(mesh.attr("simplexDimension")); return constructIsoCRLEfixed(dimension, lambdaFromENu(E, nu, dimension == 3), muFromENu(E, nu)); }, py::arg("mesh"),      py::arg("young"), py::arg("poisson"));
    m.def("StVenantKirchhoffAutoProjected", [ ](const ETensor3D &etensor) { return std::make_unique<AutoHessianProjection<StVenantKirchhoffEnergy<double, 3>>>(etensor); }, py::arg("elasticity_tensor"));
    m.def("StVenantKirchhoffAutoProjected", [ ](const ETensor2D &etensor) { return std::make_unique<AutoHessianProjection<StVenantKirchhoffEnergy<double, 2>>>(etensor); }, py::arg("elasticity_tensor"));

    generateEnergyBindingsParameterless<SymmetricDirichlet>              ("SymmetricDirichlet" ,              m, detail_module);
    generateEnergyBindingsParameterless<SymmetricDirichletDerivativeFree>("SymmetricDirichletDerivativeFree", m, detail_module);

    m.def("OptionalTensionFieldEnergy", [](double Y) { auto psi = std::make_unique<OTFE>(); psi->setStiffness(Y / 6.0); return psi; }, py::arg("youngModulus"));
}
