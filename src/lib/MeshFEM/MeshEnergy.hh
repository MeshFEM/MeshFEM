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

#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/ParallelAssembly.hh>
#include "MeshFEM/Parallelism.hh"
#include "Stencils.hh"
#include <MeshFEM/Utilities/NameMangling.hh>
#include "Elements/MaterialAssignment.hh"
#include "MeshEnergyBase.hh"

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

    const Assembler &assembler() const override { return static_cast<const Assembler &>(NewtonVars::assembler()); }
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

    static std::string name() {
        std::string ename;
        if constexpr (has_name_method<Element>::value) {
            ename = Element::name();
        } else {
            ename = get_name_of_type<Element>();
        }

        return ename + "MeshEnergy";
    }

    auto extractLocalVars(size_t si) const {
        return stencils[si].template extract<LocalVars>(m_vars.globalVars(), m_vars.varStructure());
    }

    auto extractLocalVars(size_t si, const Eigen::Ref<const VXd> &x) const {
        return stencils[si].template extract<LocalVars>(x, m_vars.varStructure());
    }

    auto extractLocalVars(size_t si, const Eigen::Ref<const VXd> &x, const typename Assembler::VarStructure &vs) const {
        return stencils[si].template extract<LocalVars>(x, vs);
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
        return summation_parallel([this](size_t ei) {
                return elementEnergy(ei);
            }, elements.size());
    }

    Real objectiveAtVars(const Eigen::Ref<const VXd> &x) const override {
        if (x.size() != m_vars.globalVars().size()) throw std::runtime_error("Invalid variable vector size");
        BENCHMARK_SCOPED_TIMER_SECTION timer(name() + ".objectiveAtVars");
        if constexpr (Element::CachesDeformedQuantities) {
            throw std::runtime_error("objectiveAtVars not supported for energies with cached deformed quantities");
        }
        else {
            auto &vs = m_vars.varStructure();
            return summation_parallel([&](size_t ei) {
                    return elements[ei].energy(extractLocalVars(ei, x));
                }, elements.size());
        }
    }

    const auto &assembler() const { return m_vars.assembler(); }

    void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer(name() + ".accumulateGradient");
        if constexpr (Element::CachesDeformedQuantities) {
            assembler().assembleGradient(g, elements.size(), [&](size_t ei) {
                return elements[ei].gradient(weight);
            }, [this](size_t ei) { return stencils[ei].blockVars; });
        }
        else {
            const auto &vs = m_vars.varStructure();
            assembler().assembleGradient(g, elements.size(), [&](size_t ei) {
                return elements[ei].gradient(weight, extractLocalVars(ei, m_vars.globalVars(), vs));
            }, [this](size_t ei) { return stencils[ei].blockVars; });
        }
    }

    using ElementHessian = typename Element::Hessian;
    ElementHessian elementHessian(size_t ei, Real weight, bool projectionMask) const {
        if constexpr (Element::CachesDeformedQuantities)
            return elements[ei].hessian(weight, projectionMask);
        else
            return elements[ei].hessian(weight, projectionMask, extractLocalVars(ei));
    }

    template<class ShouldProject>
    void accumulateHessianImpl(Real weight, NewtonHessian &H, const ShouldProject &shouldProjectHE) const {
        if (elementHessianShift != 0.0) {
            // This mode is intended only for apples-to-apples comparison
            // against the Composite Majorization codebase, which addresses
            // Hessian rank deficiency by adding a small, fixed multiple of the identity to
            // each element Hessian.
            if (useXBasedProjection) throw std::runtime_error("Combining x-based projection with elementHessianShift not supported");

            assembler().assembleHessian(H, elements.size(), [&](size_t ei) {
                ElementHessian H_e = elementHessian(ei, weight, shouldProjectHE(ei));
                H_e.diagonal().array() += weight * elementHessianShift;
                return H_e;
            }, [this](size_t ei) { return stencils[ei].blockVars; });
            return;
        }

        if (!useXBasedProjection) {
            // Use projection implemented by the element itself (e.g., F-based projection)
            assembler().assembleHessian(H, elements.size(),
                    [&](size_t ei) { return elementHessian(ei, weight, shouldProjectHE(ei)); },
                    [this](size_t ei) { return stencils[ei].blockVars; });
        }
        else {
            // Use a brute-force x-based projection
            auto getProjectedHessian = [&](size_t ei) -> ElementHessian {
                ElementHessian H_e = elementHessian(ei, weight, false);
                if (!shouldProjectHE(ei)) return H_e;
                Eigen::SelfAdjointEigenSolver<ElementHessian> Hes(H_e.transpose()); // WARNING: uses *lower* triangle, while we compute upper triangle!
                if (Hes.eigenvalues()[0] >= xBasedProjectionClampEps) return H_e; // sorted increasing
                return (Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(xBasedProjectionClampEps).asDiagonal() * Hes.eigenvectors().transpose()).eval();
            };

            assembler().assembleHessian(H, elements.size(), getProjectedHessian, [this](size_t ei) { return stencils[ei].blockVars; });
        }
    }

    void accumulateHessian(Real weight, NewtonHessian &H, bool projectionMask = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer(name() + ".hessian" + (projectionMask ? " (projected)" : ""));
        if (!projectionMask || !hasPerElementHessianProjectionMasks()) {
            // No per-element projection mask customization.
            accumulateHessianImpl(weight, H, [projectionMask](size_t ei) { return projectionMask; });
        }
        else {
            accumulateHessianImpl(weight, H, [this](size_t ei) { return elementHessianProjectionMasks[ei]; });
        }
    }

    NewtonHessian hessianSparsityPattern() const override {
        return assembler().sparsityPattern(elements.size(), [&](size_t ei) {
            return stencils[ei].blockVars;
        });
    }

    using MeshEnergyBase::elementGradientNorms; // don't hide overloads.
    VXd elementGradientNorms(const VXd &g) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("MeshEnergy.elementGradientNorms");
        const size_t ne = numElements();
        VXd result(ne);
        parallel_for_range(ne, [this, &result, &g](size_t ei) {
            auto g_e = extractLocalVars(ei, g);
            result[ei] = g_e.norm();
        });
        return result;
    }

    void varsUpdated() override {
        // Update cached state for each element.
        if constexpr (Element::CachesDeformedQuantities) {
            BENCHMARK_SCOPED_TIMER_SECTION timer(name() + ".varsUpdated");
            parallel_for_range(elements.size(),
                [&](size_t i) { elements[i].setDeformedConfiguration(extractLocalVars(i)); },
                /* grain_size = */ 100, /* parallelism_threshold = */ 1000);
        }
    }

    const VXd &globalVars() const { return m_vars.globalVars(); }

    size_t numElements() const override { return elements.size(); }

    void setHomogeneousMaterial(const Material &mat) { materials.setHomogeneous(mat); }
    void setSpatiallyVaryingMaterial(const std::vector<Material> &mats, const std::vector<size_t> &materialForElement) {
        materials.setSpatiallyVarying(mats, materialForElement);
    }

    void allocatePerElementMaterials() { materials.allocatePerElement(); }

    // Non-indexed material assignment
    void setSpatiallyVaryingMaterial(const std::vector<Material> &mats) { materials.setSpatiallyVarying(mats); }

    StencilCollection<Stencil> stencils;
    std::vector<Element> elements;
    MA materials; // must come after stencils so stencil count is initialized first.

    const auto &mesh() const { return *m_mesh; }

    const auto &mesh_ptr() const { return m_mesh; }
    const auto &vars_ptr() const { return m_vars_ptr; }

private:
    Material &m_getMaterial(size_t ei) override { return materials[ei]; }

    const Vars &m_vars;
    std::shared_ptr<Vars> m_vars_ptr; // Keep the variables structure alive.
    std::shared_ptr<Mesh> m_mesh;     // Keep the mesh alive in case the element energy class references it.
};

// A Lagrange FEM discretization of an energy functional that is a function of
// the mesh's embedding (node positions).
// This embedding (e.g., the deformed configuration of an elastic solid) maps to R^N.
template<class Mesh_, size_t N, class Element_>
struct MeshEmbeddingEnergy : public MeshEnergy<Mesh_, NodalVars<N>, ElementStencil</* K = */ Mesh_::K, Mesh_::Deg, N>, Element_> {
    using Base = MeshEnergy<Mesh_, NodalVars<N>, ElementStencil</* K = */ Mesh_::K, Mesh_::Deg, N>, Element_>;
    using Base::Base;
};

#endif /* end of include guard: MESHENERGY_HH */
