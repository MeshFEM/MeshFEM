////////////////////////////////////////////////////////////////////////////////
// MaterialAssignment.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements homogeneous or inhomogeneous per-element material properties.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  11/10/2023 12:10:34
*///////////////////////////////////////////////////////////////////////////////
#ifndef MATERIALASSIGNMENT_HH
#define MATERIALASSIGNMENT_HH

#include <MeshFEM/Types.hh>

// Mapping from each element to a material property.
template<class Material>
struct MaterialAssignment {
    // We support three types of assignment:
    //      HOMOGENEOUS: all elements share a single material property instance.
    //      PER_ELEMENT: each element has its own material property instance.
    //      INDEXED:     elements are tagged with a material index, allowing
    //                   them to share a smaller number of material property instances.
    enum class Type { HOMOGENEOUS, PER_ELEMENT, INDEXED };

    // Assign a homogeneous material by default.
    MaterialAssignment(size_t numElements) : m_type(Type::HOMOGENEOUS), m_materials(1), m_numElements(numElements) { }

    struct ElementMaterialGetter {
        ElementMaterialGetter(MaterialAssignment &materials, size_t elementIndex)
            : m_materials(materials), m_elementIndex(elementIndex) { }

        const Material &get() const { return m_materials[m_elementIndex]; }
              Material &get()       { return m_materials[m_elementIndex]; }
    private:
        MaterialAssignment &m_materials;
        size_t m_elementIndex;
    };

    const Material &operator[](size_t i) const {
        if (m_type == Type::HOMOGENEOUS) return m_materials[0];
        if (m_type == Type::PER_ELEMENT) return m_materials[i];
        return m_materials[m_materialForElement[i]]; // indexed case
    }

    Material &operator[](size_t i) {
        if (m_type == Type::HOMOGENEOUS) return m_materials[0];
        if (m_type == Type::PER_ELEMENT) return m_materials[i];
        return m_materials[m_materialForElement[i]]; // indexed case
    }

    void setHomogeneous(const Material &mat) {
        m_type = Type::HOMOGENEOUS;
        m_materials.assign(1, mat);
        m_materialForElement.clear();
    }

    // Interface supporting indexed material assignment.
    void setSpatiallyVarying(const std::vector<Material> &materials, const std::vector<size_t> &materialForElement) {
        // An empty `materialForElement` indicates a non-indexed material assignment
        if (materialForElement.empty()) return setSpatiallyVarying(materials);

        m_type = Type::INDEXED;
        if (materialForElement.size() != m_numElements) throw std::runtime_error("Invalid material assignment size");
        if (*(std::max_element(materialForElement.begin(), materialForElement.end())) >= materials.size())
            throw std::runtime_error("Invalid material index");
        m_materials = materials;
        m_materialForElement = materialForElement;
    }

    // Interface for non-indexed material assignment.
    void setSpatiallyVarying(const std::vector<Material> &materials) {
        m_type = Type::PER_ELEMENT;
        if (materials.size() != m_numElements) throw std::runtime_error("Material/element count mismatch");
        m_materials = materials;
        m_materialForElement.clear();
    }

    void allocatePerElement() {
        m_type = Type::PER_ELEMENT;
        m_materials.resize(m_numElements);
        m_materialForElement.clear();
    }

    // Loop over each allocated material
    void foreach(std::function<void(const Material &)> f) const {
        for (const auto &m : m_materials)
            f(m);
    }

    void foreach(std::function<void(Material &)> f) {
        for (auto &m : m_materials)
            f(m);
    }

    const std::vector<size_t> &materialForElement() const { return m_materialForElement; }

private:
    Type m_type;
    std::vector<size_t> m_materialForElement;
    std::vector<Material> m_materials;
    size_t m_numElements;
};

struct MaterialBase {
    virtual ~MaterialBase() { }
};

#endif /* end of include guard: MATERIALASSIGNMENT_HH */
