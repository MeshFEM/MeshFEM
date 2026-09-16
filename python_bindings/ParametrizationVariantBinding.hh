#ifndef PARAMETRIZATIONVARIANTBINDING_HH
#define PARAMETRIZATIONVARIANTBINDING_HH
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <type_traits>
#include <utility>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)
                         //
#include "../ParametrizationVariants.hh"
#include <MeshFEM/../../python_bindings/ParametrizationBinding.hh>

namespace MeshFEM {

template<class PME, class RawMaterial>
auto bindParametrizationMeshEnergyImpl(const std::string &name, py::module &m, py::module &detail) {
    using Element     = std::decay_t<decltype(std::declval<PME>().elements.front())>;
    using ElementData = typename Element::EData;
    using M32d        = typename ElementData::M32d;

    auto pyPME = bindMeshEnergy<PME, RawMaterial>(name, m, detail);
    pyPME.def("getB", [](const PME &pme, size_t ei) { return pme.elements.at(ei).elementData.B(); });
    pyPME.def("setB", [](      PME &pme, size_t ei, const M32d &B) { pme.elements.at(ei).elementData.setB(B); });
    pyPME.def("elementJacobian", [](const PME &pme, size_t ei) { return pme.elements.at(ei).getFB(pme.extractLocalVars(ei)); });

    return pyPME;
}

template<class E>
auto bindParametrizationMeshEnergyProjectToRestHessian(py::module &m, py::module &detail) {
    using PME = ParametrizationMeshEnergyProjectToRestHessian<E>;
    return bindParametrizationMeshEnergyImpl<PME, E>("ParametrizationProjectToRestHessian", m, detail);
}

template<class E>
auto bindParametrizationMeshEnergyAKVF(py::module &m, py::module &detail) {
    using PME = ParametrizationMeshEnergyAKVF<E>;
    return bindParametrizationMeshEnergyImpl<PME, E>("ParametrizationAKVF", m, detail);
}

template<class E>
void bindParametrizationMeshEnergyVariants(py::module &m, py::module &detail) {
    bindParametrizationMeshEnergy<E>(m, detail);
    bindParametrizationMeshEnergyProjectToRestHessian<E>(m, detail);
    bindParametrizationMeshEnergyAKVF<E>(m, detail);
}

} // namespace MeshFEM

#endif /* end of include guard: PARAMETRIZATIONVARIANTBINDING_HH */
