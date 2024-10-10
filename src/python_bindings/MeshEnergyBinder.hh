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

// The MeshEnergy material class creates its own wrappers around, e.g.,
// volumetric elastic energy density objects. We need to convert
// from this underlying `RawMaterial` type to the `Material` type
// actually used by the `MeshEnergy`.
// In some cases the two types are the same, and so no conversion
// is necessary.
template<class Material, class RawMaterial>
auto convertMaterial(const RawMaterial &m) {
    if constexpr (std::is_same_v<RawMaterial, Material>)
        return m;
    else
        return Material(m);
}

template<class Material, class RawMaterial>
auto convertMaterialList(const std::vector<RawMaterial> &materials) {
    if constexpr (std::is_same_v<RawMaterial, Material>)
        return materials;
    else {
        std::vector<Material> result;
        result.reserve(materials.size());
        for (const auto &m : materials)
            result.emplace_back(m);
        return result;
    }
}

// Bind a single MeshEnergy instantiation.
template<class ME, class RawMaterial = typename ME::Material>
auto bindMeshEnergy(const std::string &name, py::module &m, py::module &detail) {
    using Mesh = typename ME::Mesh;
    using Vars = typename ME::Vars;
    using Material = typename ME::Material;

    py::class_<ME, MeshEnergyBase, std::shared_ptr<ME>> pyME(detail, (name + getMeshName<Mesh>()).c_str());
    pyME.def("setHomogeneousMaterial",      [](ME &me, RawMaterial material) { me.setHomogeneousMaterial(convertMaterial<Material>(material)); }, py::arg("material"))
        .def("setSpatiallyVaryingMaterial", [](ME &me, const std::vector<RawMaterial> &mats, const std::vector<size_t> &materialForElement) { me.setSpatiallyVaryingMaterial(convertMaterialList<Material>(mats), materialForElement); }, py::arg("materials"), py::arg("materialForElement"))
        .def("elementEnergy",               [](const ME &me, size_t ei) { return me.elementEnergy(ei); }, py::arg("ei"))
        .def_readwrite("blockAccelerateHessian", &ME::blockAccelerateHessian)
        ;

    m.def(name.c_str(), [](std::shared_ptr<Mesh> mesh, std::shared_ptr<Vars> vars, RawMaterial material) {
        auto me = std::make_shared<ME>(mesh, vars);
        me->setHomogeneousMaterial(convertMaterial<Material>(material));
        return me;
    }, py::arg("mesh"), py::arg("vars"), py::arg("material") = RawMaterial());

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
