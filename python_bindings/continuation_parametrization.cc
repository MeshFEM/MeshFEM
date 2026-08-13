#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <limits>
#include <stdexcept>
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

inline void validateAttenuationParameter(double p) {
    if (!std::isfinite(p) || (p < 0.0) || (p > 1.0))
        throw std::runtime_error("AttenuatedSymmetricDirichlet p must be in [0, 1].");
}

struct AttenuatedSymmetricDirichlet {
    static constexpr const char *name() { return "AttenuatedSymmetricDirichlet"; }

    AttenuatedSymmetricDirichlet(double p_ = 1.0) : p(p_) { validateAttenuationParameter(p); }
    AttenuatedSymmetricDirichlet(const AttenuatedSymmetricDirichlet &other) : p(other.p) { }

    template<class Vec>
    typename Vec::Scalar psi(const Vec &sigma) const {
        if (sigma[sigma.size() - 1] < 0) return std::numeric_limits<double>::infinity();

        using Scalar = typename Vec::Scalar;
        if (p == 0.0) return Scalar(1.0 * sigma.size());

        using std::pow;
        const double exponent = 2.0 * p;
        Scalar result = 0.0;
        for (int i = 0; i < sigma.size(); ++i)
            result += 0.5*(pow(sigma[i], exponent) + pow(sigma[i], -exponent));
        return result;
    }

    double p;
};

template<class EnergyTypeWrapper>
void bindParameterlessParametrizationEnergy(py::module &energy, py::module &energyDetail,
                                            py::module &parametrization, py::module &parametrizationDetail) {
    const std::string name = EnergyTypeWrapper::template type<double, 2>::unmangled_name();
    generateEnergyBindingsParameterless<EnergyTypeWrapper::template type>(name, energy, energyDetail);
    bindParametrizationMeshEnergyVariants<typename EnergyTypeWrapper::template type<double, 2>>(parametrization, parametrizationDetail);
}

template<class EnergyTypeWrapper>
void bindAttenuatedSymmetricDirichlet(py::module &energy, py::module &energyDetail,
                                      py::module &parametrization, py::module &parametrizationDetail) {
    using Energy2D = typename EnergyTypeWrapper::template type<double, 2>;
    using Energy3D = typename EnergyTypeWrapper::template type<double, 3>;
    const std::string name = Energy2D::unmangled_name();

    bindEnergyFBased<Energy2D>(energyDetail)
        .def_property("p",
            [](const Energy2D &e) { return e.p; },
            [](Energy2D &e, double p) {
                validateAttenuationParameter(p);
                e.p = p;
                e.setDeformationGradient(e.getDeformationGradient());
            });
    bindEnergyFBased<Energy3D>(energyDetail)
        .def_property("p",
            [](const Energy3D &e) { return e.p; },
            [](Energy3D &e, double p) {
                validateAttenuationParameter(p);
                e.p = p;
                e.setDeformationGradient(e.getDeformationGradient());
            });

    energy.def(name.c_str(), [](size_t dimension, double p) {
        return constructDimensionSpecific<EnergyTypeWrapper::template type>(dimension, p);
    }, py::arg("dimension"), py::arg("p"));
    energy.def(name.c_str(), [](py::object mesh, double p) {
        size_t dimension = py::cast<double>(mesh.attr("simplexDimension"));
        return constructDimensionSpecific<EnergyTypeWrapper::template type>(dimension, p);
    }, py::arg("mesh"), py::arg("p"));

    bindParametrizationMeshEnergyVariants<Energy2D>(parametrization, parametrizationDetail);
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
    bindAttenuatedSymmetricDirichlet<APADIsotropicEDensity_Sigma<AttenuatedSymmetricDirichlet>>(energy, energyDetail, parametrization, parametrizationDetail);

    bindMeshEnergy<ContinuationParamMeshEnergy>("symmetric_dirichlet_param", m, detail)
        .def("setInterpolatedReference", &ContinuationParamMeshEnergy::setInterpolatedReference, py::arg("lambda"), py::arg("x"))
        .def("rebaseInterpolatedReference", &ContinuationParamMeshEnergy::rebaseInterpolatedReference, py::arg("lambda"), py::arg("x"))
        .def("computeTaylorCoefficients", &ContinuationParamMeshEnergy::computeTaylorCoefficients, py::arg("hessianFactorization"), py::arg("degree") = 6)
        .def("computeTaylorCoefficientsArclen", &ContinuationParamMeshEnergy::computeTaylorCoefficientsArclen, py::arg("hessianFactorization"), py::arg("degree") = 6)
        ;
}
