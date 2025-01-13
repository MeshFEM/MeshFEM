#ifndef BINDPARAMETRIZATIONMATERIAL_HH
#define BINDPARAMETRIZATIONMATERIAL_HH

#include <pybind11/pybind11.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)
#include <MeshFEM/Elements/ParametrizationElement.hh>

template<class Psi>
auto bindParametrizationMaterial(py::module &m, py::module &detail) {
    using PMat = ParametrizationMaterial<Psi>;
    py::class_<PMat, MaterialBase> pyMM(detail, (std::string(Psi::name()) + "ParametrizationMaterial").c_str());
    pyMM.def("setPsi",  [](PMat &m, const Psi &psi) { m.psi = psi; })
        .def(py::init<>())
        ;

    m.def("ParametrizationMaterial", [](const Psi &psi) -> std::unique_ptr<MaterialBase> {
        auto mat = std::make_unique<PMat>();
        mat->psi = psi;
        return mat;
    }, py::arg("psi"));

    return pyMM;
}

#endif /* end of include guard: BINDPARAMETRIZATIONMATERIAL_HH */
