////////////////////////////////////////////////////////////////////////////////
// BindMembraneMaterial.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/25/2023 16:22:16
*///////////////////////////////////////////////////////////////////////////////
#ifndef BINDMEMBRANEMATERIAL_HH
#define BINDMEMBRANEMATERIAL_HH

#include <pybind11/pybind11.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)
#include <MeshFEM/Elements/MembraneElement.hh>

template<class Psi>
auto bindMembraneMaterial(py::module &m, py::module &detail) {
    using MMat = MembraneMaterial<Psi>;
    py::class_<MMat, MaterialBase> pyMM(detail, (std::string(Psi::name()) + "MembraneMaterial").c_str());
    pyMM.def("setPsi",  [](MMat &m, const Psi &psi) { m.psi = psi; })
        .def_readwrite("thickness", &MMat::thickness)
        .def(py::init<>())
        ;

    m.def("MembraneMaterial", [](const Psi &psi, double thickness) -> std::unique_ptr<MaterialBase> {
        auto mat = std::make_unique<MMat>();
        mat->psi = psi;
        mat->thickness = thickness;
        return mat;
    }, py::arg("psi"), py::arg("thickness") = 1);

    return pyMM;
}

#endif /* end of include guard: BINDMEMBRANEMATERIAL_HH */
