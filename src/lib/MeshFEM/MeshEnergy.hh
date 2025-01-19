////////////////////////////////////////////////////////////////////////////////
// MeshEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//
//  Generic infrastructure for defining energies constructed as a sum over
//  elements whose local variables are attached to entities of a mesh.
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
#include "Elements/MaterialAssignment.hh"

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
struct MESHFEM_EXPORT MeshEnergyVars : public NewtonVars {
    using Assembler = SystemAssembler<MVSpec::BlockDimension...>;

    template<class Mesh>
    MeshEnergyVars(const Mesh &m) {
        this->m_assembler = std::make_unique<Assembler>(MVSpec::numVars(m)...);
        m_x.resize(this->m_assembler->numVars());

        // Initialize each block of the variable vector.
        const auto &v = varStructure();
        size_t type = 0;
        ((MVSpec::initialize(m, m_x.segment(v.offsetForType(type), v.numVarsOfType(type))), ++type), ...);
    }

    const Assembler &assembler() const override { return dynamic_cast<const Assembler &>(NewtonVars::assembler()); }
    const auto &varStructure() const { return assembler().varStructure(); }
    const VXd &globalVars() const { return m_x; }

    virtual ~MeshEnergyVars() { }
};

template<typename... MVSpec>
struct NameMangler<MeshEnergyVars<MVSpec...>> {
    static std::string name() {
        return "MeshEnergyVars" + (... + ("_" +  MVSpec::name()));
    }
};

template<size_t N>
using NodalVars = MeshEnergyVars<MeshVarSpecification<MeshVarType::PER_NODE, N>>;

struct MeshEnergyBase : public NewtonObjectiveTerm { 
    MeshEnergyBase(std::shared_ptr<NewtonVarsBase> vars)
        : NewtonObjectiveTerm(vars) { }

    MaterialBase &materialForElement(size_t ei) {
        if (ei >= numElements()) throw std::runtime_error("Element index out of bounds");
        auto &mat = m_getMaterial(ei);
        return mat;
    }

    virtual size_t numElements() const = 0;

    virtual ~MeshEnergyBase() { }

    bool useXBasedProjection = false;
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

    MeshEnergy(std::shared_ptr<Mesh> m, std::shared_ptr<Vars> vars)
        : MeshEnergyBase(vars), m_mesh(m), stencils(*m), m_vars_ptr(vars), m_vars(*vars), materials(stencils.size()) {
        const size_t ns = stencils.size();
        elements.reserve(ns);

        if constexpr (Element::CachesDeformedQuantities) {
            for (size_t i = 0; i < ns; ++i)
                elements.emplace_back(i, *m, extractLocalVars(i), materials);
        }
        else {
            for (size_t i = 0; i < ns; ++i)
                elements.emplace_back(i, *m, materials);
        }
    }

    auto extractLocalVars(size_t si) const {
        return stencils[si].template extract<LocalVars>(m_vars.globalVars(), m_vars.varStructure());
    }

    Real elementEnergy(size_t ei) const {
        if constexpr (Element::CachesDeformedQuantities) {
            return elements[ei].energy();
        }
        else {
            return elements[ei].energy(extractLocalVars(ei));
        }
    }

    Real objective() const override {
        return summation_parallel([&](size_t ei) {
                return elementEnergy(ei);
            }, elements.size());
    }

    const auto &assembler() const { return m_vars.assembler(); }

    void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        if constexpr (Element::CachesDeformedQuantities) {
            assembler().assembleGradient(g, elements.size(), [&](size_t ei) {
                return elements[ei].gradient(weight);
            }, [this](size_t ei) { return stencils[ei].blockVars; });
        }
        else {
            assembler().assembleGradient(g, elements.size(), [&](size_t ei) {
                return elements[ei].gradient(weight, extractLocalVars(ei));
            }, [this](size_t ei) { return stencils[ei].blockVars; });
        }
    }

    void accumulateHessian(Real weight, NewtonHessian &H, bool projectionMask = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("MeshEnergy<" + Element_::name() + ">.hessian");
        if (!useXBasedProjection || !projectionMask) {
            // Use projection implemented by the element itself (e.g., F-based projection)
            assembler().assembleHessian(H, elements.size(), [&](size_t ei) {
                if constexpr (Element::CachesDeformedQuantities)
                    return elements[ei].hessian(weight, projectionMask);
                else
                    return elements[ei].hessian(weight, projectionMask, extractLocalVars(ei));
            }, [this](size_t ei) { return stencils[ei].blockVars; });
        }
        else {
            // Use a brute-force x-based projection
            using ElementHessian = typename Element::Hessian;
            auto getProjectedHessian = [&](size_t ei) -> ElementHessian {
                ElementHessian H_e;
                if constexpr (Element::CachesDeformedQuantities)
                    H_e = elements[ei].hessian(weight, /* projectionMask = */ false);
                else
                    H_e = elements[ei].hessian(weight, /* projectionMask = */ false, extractLocalVars(ei));
                Eigen::SelfAdjointEigenSolver<ElementHessian> Hes(H_e.transpose()); // WARNING: uses *lower* triangle, while we compute upper triangle!
                if (Hes.eigenvalues()[0] >= 0.0) return H_e; // sorted increasing
                return Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(0.0).asDiagonal() * Hes.eigenvectors().transpose();
            };

            assembler().assembleHessian(H, elements.size(), getProjectedHessian, [this](size_t ei) { return stencils[ei].blockVars; });
        }
    }

    NewtonHessian hessianSparsityPattern() const override {
        return assembler().sparsityPattern(elements.size(), [&](size_t ei) {
            return stencils[ei].blockVars;
        });
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
    std::shared_ptr<Vars> m_vars_ptr; // Keep the variables structure alive.
    std::shared_ptr<Mesh> m_mesh;     // Keep the mesh alive in case the element energy class references it.
};

#endif /* end of include guard: MESHENERGY_HH */
