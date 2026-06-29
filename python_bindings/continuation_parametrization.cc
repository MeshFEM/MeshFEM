#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <limits>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/EnergyDensities/IsotropicAutodiffEDensity.hh>
#include <MeshFEM/../../python_bindings/EnergyBinding.hh>
#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include <MeshFEM/../../python_bindings/ParametrizationBinding.hh>
#include "../ContinuationParametrization.hh"

struct ARAP {
    static constexpr const char *name() { return "ARAP"; }

    template<class Vec>
    auto psi(const Vec &sigma) { return 0.5 * (sigma.array() - 1).square().sum(); }
};

struct SymmetricARAP {
    static constexpr const char *name() { return "SymmetricARAP"; }

    template<class Vec>
    typename Vec::Scalar psi(const Vec &sigma) {
        if (sigma[sigma.size() - 1] < 0) return std::numeric_limits<double>::infinity();
        return 0.5 * ((      sigma.array() - 1).square().sum()
                   +  (1.0 / sigma.array() - 1).square().sum());
    }
};

template<class EnergyTypeWrapper>
void bindParameterlessParametrizationEnergy(py::module &energy, py::module &energyDetail,
                                            py::module &parametrization, py::module &parametrizationDetail) {
    const std::string name = EnergyTypeWrapper::template type<double, 2>::unmangled_name();
    generateEnergyBindingsParameterless<EnergyTypeWrapper::template type>(name, energy, energyDetail);
    bindParametrizationMeshEnergyVariants<typename EnergyTypeWrapper::template type<double, 2>>(parametrization, parametrizationDetail);
}

PYBIND11_MODULE(continuation_parametrization, m)
{
    py::module::import("mesh_energy");
    py::module detail = m.def_submodule("detail");
    py::module energy = m.def_submodule("energy");
    py::module energyDetail = energy.def_submodule("detail");
    py::module parametrization = m.def_submodule("parametrization");
    py::module parametrizationDetail = parametrization.def_submodule("detail");

    using Mesh = FEMMesh<2, 1, Vector3D>;
    using Vars = NodalVars<2>;
    using Stencil = ElementStencil</* K = */ 2, /* Deg = */ 1, /* N = */ 2>;

    bindParameterlessParametrizationEnergy<APADIsotropicEDensity_Sigma<ARAP>>(energy, energyDetail, parametrization, parametrizationDetail);
    bindParameterlessParametrizationEnergy<APADIsotropicEDensity_Sigma<SymmetricARAP>>(energy, energyDetail, parametrization, parametrizationDetail);

    bindMeshEnergy<ContinuationParamMeshEnergy>("symmetric_dirichlet_param", m, detail)
        .def("setInterpolatedReference", &ContinuationParamMeshEnergy::setInterpolatedReference, py::arg("lambda"), py::arg("x"))
        .def("computeTaylorCoefficients", &ContinuationParamMeshEnergy::computeTaylorCoefficients, py::arg("hessianFactorization"), py::arg("degree") = 6)
        .def("computeTaylorCoefficientsArclen", &ContinuationParamMeshEnergy::computeTaylorCoefficientsArclen, py::arg("hessianFactorization"), py::arg("degree") = 6)
        ;
}
