////////////////////////////////////////////////////////////////////////////////
// MeshEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//
//  Generic infrastructure for defining elements whose local variables are
//  attached to entities of a mesh.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/22/2023 11:59:02
*///////////////////////////////////////////////////////////////////////////////
#ifndef MESHENERGY_HH
#define MESHENERGY_HH
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/ParallelAssembly.hh>
#include "Stencils.hh"
#include <MeshFEM/Utilities/NameMangling.hh>

enum class MeshVarType { PER_NODE, PER_EDGE, PER_CELL };
template<MeshVarType MVT, size_t N> struct MeshVarSpecification {
    static constexpr MeshVarType type = MVT;
    static constexpr size_t BlockDimension = N;
    template<class Mesh>
    static size_t numVars(const Mesh &m) {
        if (type == MeshVarType::PER_NODE) return m.numNodes();
        if (type == MeshVarType::PER_EDGE) return m.numEdges();
        if (type == MeshVarType::PER_CELL) return m.numElements();
        throw std::runtime_error("Invalid MeshVarType");
    }

    template<class Mesh, class Derived>
    static void initialize(const Mesh &m, Eigen::MatrixBase<Derived> &&x) {
        if (x.size() != N * numVars(m)) throw std::runtime_error("Invalid block size: " + std::to_string(x.size()) + " != " + std::to_string(N * numVars(m)));

        // We assume per-node variables whose block sizes match the embedding
        // dimension should be initialized from the mesh's rest node positions.
        if constexpr ((type == MeshVarType::PER_NODE) && (N == Mesh::EmbeddingDimension)) {
            const size_t nn = m.numNodes();
            for (size_t ni = 0; ni < nn; ++ni)
                x.template segment<N>(ni * N) = m.node(ni)->p;
        }
        else {
            // We zero-initialize all other variable types.
            x.setZero();
        }
    }

    static std::string name() {
        if (type == MeshVarType::PER_NODE) return std::to_string(N) + "_PER_NODE";
        if (type == MeshVarType::PER_EDGE) return std::to_string(N) + "_PER_EDGE";
        if (type == MeshVarType::PER_CELL) return std::to_string(N) + "_PER_CELL";
        throw std::runtime_error("Invalid MeshVarType");
    }
};

template<typename... MVSpec>
struct MeshEnergyVars : public NewtonVars {
    using Assembler = SystemAssembler<MVSpec::BlockDimension...>;

    template<class Mesh>
    MeshEnergyVars(const Mesh &m)
        : NewtonVars(0), m_assembler(MVSpec::numVars(m)...)
    {
        m_x.resize(m_assembler.numVars());

        // Initialize each block of the variable vector.
        const auto &v = varStructure();
        size_t type = 0;
        ((MVSpec::initialize(m, m_x.segment(v.offsetForType(type), v.numVarsOfType(type))), ++type), ...);
    }

    const auto &varStructure() const { return m_assembler.varStructure(); }
    const VXd &globalVars() const { return m_x; }
    const Assembler &assembler() const { return m_assembler; }

private:
    Assembler m_assembler;
};

template<typename... MVSpec>
struct NameMangler<MeshEnergyVars<MVSpec...>> {
    static std::string name() {
        return "MeshEnergyVars" + (... + ("_" +  MVSpec::name()));
    }
};

template<size_t N>
using NodalVars = MeshEnergyVars<MeshVarSpecification<MeshVarType::PER_NODE, N>>;

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

    // Indexed case
    void setSpatiallyVarying(const std::vector<Material> &materials, const std::vector<size_t> &materialForElement) {
        m_type = Type::INDEXED;
        if (materialForElement.size() != m_numElements) throw std::runtime_error("Invalid material assignment size");
        if (*(std::max_element(materialForElement.begin(), materialForElement.end())) >= materials.size())
            throw std::runtime_error("Invalid material index");
        m_materials = materials;
        m_materialForElement = materialForElement;
    }

    // Non-indexed case
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

private:
    Type m_type;
    std::vector<size_t> m_materialForElement;
    std::vector<Material> m_materials;
    size_t m_numElements;
};

// Traits class must define `Material` type
template<class Derived>
struct ElementTraits;

template<class Derived>
struct ElementBase {
    using Material       = typename ElementTraits<Derived>::Material;
    using MA             = MaterialAssignment<Material>;
    using MaterialGetter = typename MA::ElementMaterialGetter;
    ElementBase(size_t ei, MA &materials) : m_materialGetter(materials, ei) { }

    const Material &material() const { return m_materialGetter.get(); }
          Material &material()       { return m_materialGetter.get(); }
private:
    MaterialGetter m_materialGetter;
};

struct MaterialBase {
    virtual ~MaterialBase() { }
};

struct MESHFEM_EXPORT MeshEnergyBase : public NewtonObjectiveTerm { 
    MaterialBase &materialForElement(size_t ei) {
        if (ei >= numElements()) throw std::runtime_error("Element index out of bounds");
        auto &mat = m_getMaterial(ei);
        return mat;
    }

    virtual size_t numElements() const = 0;

    virtual ~MeshEnergyBase() { }
private:
    virtual MaterialBase &m_getMaterial(size_t ei) = 0;
};

// The instantiation of a stencil-based energy term for a given mesh
// and variable definition.
template<class Mesh_, class MEVars_, class Stencil_, class Element_>
struct MeshEnergy : public MeshEnergyBase {
    using Mesh    = Mesh_;
    using Stencil = Stencil_;
    using Element = Element_;
    using Vars    = MEVars_;
    using Assembler = typename Vars::Assembler;
    using LocalVars = typename Element::LocalVars;
    using Real      = typename Element::Real;
    using Material  = typename Element::Material;
    using MA        = MaterialAssignment<Material>;
    using EMG       = typename MA::ElementMaterialGetter;

    MeshEnergy(std::shared_ptr<Mesh> m, std::shared_ptr<Vars> vars)
        : m_mesh(m), stencils(*m), m_vars_ptr(vars), m_vars(*vars), m_assembler(vars->assembler()), materials(stencils.size()) {
        const size_t ns = stencils.size();
        elements.reserve(ns);
        for (size_t i = 0; i < ns; ++i)
            elements.emplace_back(i, *m, extractLocalVars(i), materials);
    }

    auto extractLocalVars(size_t si) const {
        return stencils[si].template extract<LocalVars>(m_vars.globalVars(), m_vars.varStructure());
    }

    Real elementEnergy(size_t ei) const { return elements[ei].energy(); }

    Real objective() const override {
        return summation_parallel([&](size_t ei) {
                return elements[ei].energy();
            }, elements.size());
    }

    void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        m_assembler.assembleGradient(g, elements.size(), [&](size_t ei) {
            return elements[ei].gradient(weight);
        }, [this](size_t ei) { return stencils[ei].blockVars; });
    }

    void accumulateHessian(Real weight, SuiteSparseMatrix &H, bool projectionMask = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("MeshEnergy<" + Element_::name() + ">.hessian");
        m_assembler.assembleHessian(H, elements.size(), [&](size_t ei) {
            return elements[ei].hessian(weight, projectionMask);
        }, [this](size_t ei) { return stencils[ei].blockVars; });
    }

    SuiteSparseMatrix hessianSparsityPattern(double val = 0.0) const override {
        const size_t nv = m_vars.numVars();

        SuiteSparseMatrix Hsp_block = m_assembler.blockSparsityPattern(elements.size(), [&](size_t ei) {
            return stencils[ei].blockVars;
        });
        return m_assembler.blockHessianSparsityPatternToScalar(Hsp_block, val);
    }

    void varsUpdated() override {
        // Update cached state for each element.
        if constexpr (Element::CachesDeformedQuantities) {
            for (size_t i = 0; i < elements.size(); ++i)
                elements[i].setDeformedConfiguration(extractLocalVars(i));
        }
    }

    size_t numElements() const override { return elements.size(); }

    void setHomogeneousMaterial(const Material &mat) { materials.setHomogeneous(mat); }
    void setSpatiallyVaryingMaterial(const std::vector<Material> &mats, const std::vector<size_t> &materialForElement) {
        materials.setSpatiallyVarying(mats, materialForElement);
    }

    StencilCollection<Stencil> stencils;
    std::vector<Element> elements;
    MA materials; // must come after stencils so stencil count is initialized first.
private:
    Material &m_getMaterial(size_t ei) override { return materials[ei]; }

    const Vars &m_vars;
    const Assembler &m_assembler;
    std::shared_ptr<Vars> m_vars_ptr; // Keep the variables structure alive.
    std::shared_ptr<Mesh> m_mesh;     // Keep the mesh alive in case the element energy class references it.
};

#endif /* end of include guard: MESHENERGY_HH */
