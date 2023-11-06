////////////////////////////////////////////////////////////////////////////////
// MeshEnergyBinder.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Support for binding MeshEnergy classes
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  11/04/2023 15:27:51
*///////////////////////////////////////////////////////////////////////////////
#ifndef MESHENERGYBINDER_HH
#define MESHENERGYBINDER_HH

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <MeshFEM/MeshEnergy.hh>
#include <MeshFEM/Elements/HingeElement.hh>

namespace py = pybind11; // NOLINT (work around clang-tidy bug)

// Note: MEVars_ has not necessarily been bound as a subclass of NewtonVarsBase
// on the Python side, so we cannot rely on pybind11 to do the downcast for us.
template<class MEVars_>
static auto downcastVars(std::shared_ptr<NewtonVarsBase> varsBase) {
    auto vars = std::dynamic_pointer_cast<MEVars_>(varsBase);
    if (!vars) throw std::runtime_error("Incompatible vars type");
    return vars;
}

template<class ME>
struct ElementSpecificMEBindings {
    template<class PyME>
    static void bind(PyME &pyME) { }
};

template<class HingeEnergy>
struct ElementSpecificMEBindings<HingeMeshEnergy<HingeEnergy>> {
    using ME = HingeMeshEnergy<HingeEnergy>;
    template<class PyME>
    static void bind(PyME &pyME) {
        pyME.def("theta", [](const ME &me, size_t ei) { return me.elements[ei].theta(); } );
    }
};

// Bind a single MeshEnergy instantiation.
template<class ME>
auto bindMeshEnergy(const std::string &name, py::module &m, py::module &detail) {
    using Mesh = typename ME::Mesh;
    using Vars = typename ME::Vars;
    using Material = typename ME::Material;
    py::class_<ME, MeshEnergyBase, std::shared_ptr<ME>> pyME(detail, (name + getMeshName<Mesh>()).c_str());
    pyME.def("setHomogeneousMaterial",      [](ME &me, Material material) { me.setHomogeneousMaterial(material); }, py::arg("material"))
        .def("setSpatiallyVaryingMaterial", [](ME &me, const std::vector<Material> &mats, const std::vector<size_t> &materialForElement) { me.setSpatiallyVaryingMaterial(mats, materialForElement); }, py::arg("materials"), py::arg("materialForElement"))
        .def("elementEnergy",               [](const ME &me, size_t ei) { return me.elementEnergy(ei); }, py::arg("ei"))
        ;

    m.def(name.c_str(), [](std::shared_ptr<Mesh> mesh, std::shared_ptr<NewtonVarsBase> varsBase, Material material) {
        auto me = std::make_shared<ME>(mesh, downcastVars<Vars>(varsBase));
        me->setHomogeneousMaterial(material);
        return me;
    }, py::arg("mesh"), py::arg("vars"), py::arg("material") = Material());

    ElementSpecificMEBindings<ME>::bind(pyME);

    return pyME;
}

// Support for binding a mesh energy template for all mesh types.
template<class MEVars_, class Stencil_, class Element_>
struct MeshEnergyBinder {
    MeshEnergyBinder(const std::string &name) : m_name(name) { }
    using Material = typename Element_::Material;

    template<class FEMMesh_>
    void bind(py::module &m, py::module &detail) {
        using ME = MeshEnergy<FEMMesh_, MEVars_, Stencil_, Element_>;
        bindMeshEnergy<ME>(m_name, m, detail);
    }

private:
    const std::string &m_name;
};

#endif /* end of include guard: MESHENERGYBINDER_HH */
