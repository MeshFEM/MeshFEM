#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/EnergyDensities/IsotropicAutodiffEDensity.hh>
#include <MeshFEM/../../python_bindings/CustomEnergyInstantiationsBinder.hh>

template<typename Real_, size_t Dim_>
using CommonNeoHookeanAD = AutodiffEDensity<CommonNeoHookeanPsi, Real_, Dim_>;

struct ARAP {
    template<class Vec>
    auto psi(const Vec &sigma) { return 0.5 * (sigma.array() - 1).square().sum(); }
};

struct SymmetricARAP {
    template<class Vec>
    typename Vec::Scalar psi(const Vec &sigma) {
        if (sigma[sigma.size() - 1] < 0) return std::numeric_limits<double>::infinity();
        return 0.5 * ((      sigma.array() - 1).square().sum()
                   +  (1.0 / sigma.array() - 1).square().sum());
    }
};

PYBIND11_MODULE(custom_energy_demo, m)
{
    CustomEnergyInstantiationsBinder b(m);

    // Parametrization-only energies (`bindSolid = false`)
    b.bindParameterless<APADIsotropicEDensity_Sigma<ARAP>, false>();
    b.bindParameterless<APADIsotropicEDensity_Sigma<SymmetricARAP>, false>();
    b.bindParameterless<APADIsotropicEDensity_SBased<SymmetricDirichletFromInvariants>, false>("SymmetricDirichletFromInvariants");
    b.bindParameterless<APADIsotropicEDensity_FBased<SymmetricDirichletPsi>,            false>("SymmetricDirichletIsotropicFAD");

    // Simulation and parametrization energies
    b.bindYoungPoisson<APADIsotropicEDensity_SBased<CommonNeoHookeanFromInvariants>>("CommonNeoHookeanFromInvariants");
    b.bindYoungPoisson<APADIsotropicEDensity_FBased<CommonNeoHookeanPsi>           >("CommonNeoHookeanIsotropicFAD");
    b.bindYoungPoisson<           ADEDensity_FBased<CommonNeoHookeanPsi>           >("CommonNeoHookeanAD");

}