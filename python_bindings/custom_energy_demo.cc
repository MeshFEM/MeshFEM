#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/EnergyDensities/IsotropicAutodiffEDensity.hh>
#include <MeshFEM/../../python_bindings/CustomEnergyInstantiationsBinder.hh>

template<typename Real_, size_t Dim_>
using CommonNeoHookeanAD = AutodiffEDensity<CommonNeoHookeanPsi, Real_, Dim_>;

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

}
