#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <IsotropicAutodiff/IsotropicAutodiffEDensity.hh>
#include <MeshFEM/../../python_bindings/CustomEnergyInstantiationsBinder.hh>
#include "ParametrizationVariantBinding.hh"

using namespace MeshFEM;

template<typename Real_, size_t Dim_>
using CommonNeoHookeanAD = AutodiffEDensity<CommonNeoHookeanPsi, Real_, Dim_>;

template<class EnergyTypeWrapper>
void bindPrivateParametrizationVariants(py::module &m, py::module &detail) {
    using E = typename EnergyTypeWrapper::template type<double, 2>;
    bindParametrizationMeshEnergyProjectToRestHessian<E>(m, detail);
    bindParametrizationMeshEnergyAKVF<E>(m, detail);
}

PYBIND11_MODULE(custom_energy_demo, m)
{
    CustomEnergyInstantiationsBinder b(m);

    // Parametrization-only energies (`bindSolid = false`)
    b.bindParameterless<APADIsotropicEDensity_SBased<SymmetricDirichletFromInvariants>, false>("SymmetricDirichletFromInvariants");
    b.bindParameterless<APADIsotropicEDensity_FBased<SymmetricDirichletPsi>,            false>("SymmetricDirichletIsotropicFAD");

    // Simulation and parametrization energies
    b.bindYoungPoisson<APADIsotropicEDensity_SBased<CommonNeoHookeanFromInvariants>>("CommonNeoHookeanFromInvariants");
    b.bindYoungPoisson<APADIsotropicEDensity_FBased<CommonNeoHookeanPsi>           >("CommonNeoHookeanIsotropicFAD");
    b.bindYoungPoisson<           ADEDensity_FBased<CommonNeoHookeanPsi>           >("CommonNeoHookeanAD");

    // Extend the public helper's standard bindings with our private variants.
    auto parametrization = m.attr("parametrization").cast<py::module>();
    auto detail = parametrization.attr("detail").cast<py::module>();
    bindPrivateParametrizationVariants<APADIsotropicEDensity_SBased<SymmetricDirichletFromInvariants>>(parametrization, detail);
    bindPrivateParametrizationVariants<APADIsotropicEDensity_FBased<SymmetricDirichletPsi>>(parametrization, detail);
    bindPrivateParametrizationVariants<APADIsotropicEDensity_SBased<CommonNeoHookeanFromInvariants>>(parametrization, detail);
    bindPrivateParametrizationVariants<APADIsotropicEDensity_FBased<CommonNeoHookeanPsi>>(parametrization, detail);
    bindPrivateParametrizationVariants<ADEDensity_FBased<CommonNeoHookeanPsi>>(parametrization, detail);
}
