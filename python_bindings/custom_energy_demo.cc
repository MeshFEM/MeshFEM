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

    b.bindParameterless<SymmetricDirichletFromInvariants, false, false, true>("SymmetricDirichletFromInvariants");
    b.bindParameterless<SymmetricDirichletIsotropicFAD,   false, false, true>("SymmetricDirichletIsotropicFAD");

    b.bindYoungPoisson<CommonNeoHookeanFromInvariants, true, false, true>("CommonNeoHookeanFromInvariants");
    b.bindYoungPoisson<CommonNeoHookeanIsotropicFAD,   true, false, true>("CommonNeoHookeanIsotropicFAD");
    b.bindYoungPoisson<CommonNeoHookeanAD,             true, false, true>("CommonNeoHookeanAD");
}
