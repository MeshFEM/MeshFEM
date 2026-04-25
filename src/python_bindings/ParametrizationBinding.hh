#ifndef PARAMETRIZATIONBINDING_HH
#define PARAMETRIZATIONBINDING_HH
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)
                         //
#include <MeshFEM/Elements/ParametrizationElement.hh>

template<class E>
auto bindParametrizationMeshEnergy(py::module &m, py::module &detail) {
    using PME = ParametrizationMeshEnergy<E>;
    using Element     = std::decay_t<decltype(std::declval<PME>().elements.front())>;
    using ElementData = typename Element::EData;
    using M32d        = typename ElementData::M32d;

    auto pyPME = bindMeshEnergy<PME, E>("Parametrization", m, detail);
    pyPME.def("getB", [](const PME &pme, size_t ei) { return pme.elements.at(ei).elementData.B(); });
    pyPME.def("setB", [](      PME &pme, size_t ei, const M32d &B) { pme.elements.at(ei).elementData.setB(B); });
    pyPME.def("elementJacobian", [](const PME &pme, size_t ei) { return pme.elements.at(ei).getFB(pme.extractLocalVars(ei)); });

    return pyPME;
}

#endif /* end of include guard: PARAMETRIZATIONBINDING_HH */
