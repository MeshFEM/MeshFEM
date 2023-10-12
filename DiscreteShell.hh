////////////////////////////////////////////////////////////////////////////////
// DiscreteShell.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Demonstrates how the discrete shell model (Grinspun 2003) can be implemented
//  using MeshFEM.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  06/29/2023 10:48:18
*///////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/ParallelAssembly.hh>
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <MeshFEM/Elements/HyperelasticLagrange.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include "HingeElement.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include <memory>
#include <vector>

// Recursion base case.
template<class GlobalVarStructure, size_t Idx, size_t Offset, size_t... VarSizes>
struct ExtractImpl {
    template<class BlockVars, class LocalVars, class Derived>
    static void run(const BlockVars &/* bvars */, LocalVars &/* result */, const Eigen::MatrixBase<Derived> &/* x */, const GlobalVarStructure &/* varStructure */) { }
};

template<class GlobalVarStructure, size_t Idx, size_t Offset, size_t FirstVarSize, size_t... VarSizes>
struct ExtractImpl<GlobalVarStructure, Idx, Offset, FirstVarSize, VarSizes...> {
    template<class BlockVars, class LocalVars, class Derived>
    static void run(const BlockVars &bvars, LocalVars &result, const Eigen::MatrixBase<Derived> &x, const GlobalVarStructure &varStructure) {
        Eigen::Map<VecX_T<typename LocalVars::Scalar>> result_ravel(result.data(), result.size());
        result_ravel.template segment<FirstVarSize>(Offset) = x.template segment<FirstVarSize>(varStructure.offsetForBlock(bvars[Idx]));
        ExtractImpl<GlobalVarStructure, Idx + 1, Offset + FirstVarSize, VarSizes...>::run(bvars, result, x, varStructure);
    }
};

template<size_t... BlockDimensions>
struct StaticStencil {
    static constexpr size_t NumLocalBlockVars = sizeof...(BlockDimensions);
    using BlockVars = std::array<int, NumLocalBlockVars>;

    StaticStencil(const BlockVars &bv) : blockVars(bv) { }

    template<class LocalVars, class Derived, class GlobalVarStructure>
    void extract(LocalVars &result, const Eigen::MatrixBase<Derived> &x, const GlobalVarStructure &varStructure) const {
        ExtractImpl<GlobalVarStructure, 0, 0, BlockDimensions...>::run(blockVars, result, x, varStructure);
    }

    template<class LocalVars, class Derived, class GlobalVarStructure>
    LocalVars extract(const Eigen::MatrixBase<Derived> &x, const GlobalVarStructure &varStructure) const {
        LocalVars result;
        extract(result, x, varStructure);
        return result;
    }

    BlockVars blockVars;
};

template<template<typename> class HingeEnergy_T>
struct DiscreteShell : public ElasticObject<double> {
    static constexpr size_t Deg = 1;
    static constexpr size_t K = 2;
    static constexpr size_t N = 3;
    using HingeEnergy = HingeEnergy_T<double>;

    using Assembler = SystemAssembler<3>;
    using HingeStencil = StaticStencil<3, 3, 3, 3>;
    using MembraneStencil = StaticStencil<3, 3, 3>;

    using V3d  = Eigen::Vector3d;
    using VXd  = Eigen::VectorXd;
    using MX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>; // Row major so that flattened order agrees with VField

    using Mesh = FEMMesh<K, Deg, V3d>; // Linear triangle mesh embedded in 3d.
    using Psi_2x2 = NeoHookeanEnergy<double, K>; // 2d energy density used to define the membrane energy
    using Psi = AutoHessianProjection<MembraneEnergyDensityFrom2x2Density<Psi_2x2>>;

    using HE = HingeElement<HingeEnergy>;
    using ME = elements::Membrane<Psi, K, Deg>;
    using NodePositions = typename ME::NodePositions;

    DiscreteShell(const std::shared_ptr<Mesh> &m, double Y = 200, double nu = 0.3)
        : m_mesh(m), m_assembler(m->numVertices()),
          m_psi(Y * nu / ((1 + nu) * (1 - 2 * nu)), // 3D Lame parameter lambda (plane stress conditions are applied inside `NeoHookeanEnergy`)
                Y / (2 * (1 + nu)))                 // Shear modulus mu
    {
        // Initialize deformed positions using the mesh's rest vertex positions.
        m_x.resize(m->numVertices() * 3);
        Eigen::Map<MX3d>(m_x.data(), m->numVertices(), 3) = getV(*m);

        // Construct hinges.
        for (const auto &he : mesh().halfEdges()) {
            if (!he.isPrimary() || he.isBoundary()) continue;
            m_hingeStencils.push_back(HingeStencil{{ he.tail().index(),
                                       he.tip ().index(),
                                       he.opposite().next().tip().index(),
                                       he           .next().tip().index() }});
            m_hingeElements.emplace_back(extractHingeVars(m_hingeStencils.size() - 1, m_x));
        }

        // Construct EmbeddedMembraneElementData
        const size_t ne = m->numElements();
        m_elementData.reserve(ne);
        for (size_t ei = 0; ei < ne; ++ei)
            m_elementData.emplace_back(*(m->element(ei)));
    }

    auto extractHingeVars(size_t hi, const VXd &x) const {
        return m_hingeStencils[hi].template extract<typename HE::Vars>(x, m_assembler.vars());
    }
    auto hingeVars(size_t hi) const { return extractHingeVars(hi, m_x); }

          Mesh &mesh()       { return *m_mesh; }
    const Mesh &mesh() const { return *m_mesh; }

    size_t numVertices() const { return mesh().numVertices(); }
    size_t   numHinges() const { return m_hingeElements.size(); }

    virtual size_t numDefoVars() const { return 3 * numVertices(); }
    virtual size_t numRestVars() const { return 3 * numVertices(); }

    virtual VXd getDefoVars() const { return m_x; }
    virtual VXd getRestVars() const { return Eigen::Map<const VXd>(getV(mesh()).data(), numDefoVars()); }
    MX3d  deformedPositions() const { return Eigen::Map<const MX3d>(m_x.data(), mesh().numVertices(), 3); }

    virtual double energy() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("DiscreteShell.energy");
        double result = 0;
        const auto &m = mesh();

        // Discrete shells bending energy term
        for (const auto &hinge : m_hingeElements)
            result += bendingStiffness * hinge.energy();

        result += summation_parallel([this](size_t ei) {
                return h * ME::energy(m_psi, getCornerPositions(ei), m_elementData[ei]);
            }, mesh().numElements());

        return result;
    }

    virtual VXd gradient(bool updatedParametrization = false, VariableMask vmask = VariableMask::Defo) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("DiscreteShell.gradient");
        VXd g = VXd::Zero(numVars());
        const auto &m = mesh();

        // Membrane energy contribution
        m_assembler.assembleGradient(g, mesh(), [this](size_t ei) {
            return (h * ME::gradient(m_psi, getCornerPositions(ei), m_elementData[ei])).eval();
        });

        // Bending energy contribution
        m_assembler.assembleGradient(g, numHinges(), [this](size_t hi) {
            return (bendingStiffness * m_hingeElements[hi].gradient()).eval();
        }, [this](size_t hi) { return m_hingeStencils[hi].blockVars; });

        return g;
    }

    using Hessian = Eigen::Matrix<double, 12, 12>;
    using ESolver  = Eigen::SelfAdjointEigenSolver<Hessian>;
    virtual void hessian(CSCMat &H, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const {
        // Assemble membrane term.
        BENCHMARK_SCOPED_TIMER_SECTION timer("DiscreteShell.hessian");
        m_assembler.assembleHessian(H, mesh(), [&](size_t ei) {
            return (h * ME::hessian(m_psi, getCornerPositions(ei), m_elementData[ei], !projectionMask)).eval();
        });

        // Assemble bending term.
        m_assembler.assembleHessian(H, numHinges(),
            [&](size_t hingeIndex) -> Hessian {
                auto H_e = (bendingStiffness * m_hingeElements[hingeIndex].hessian()).eval();
                if (!projectionMask) return H_e;

                ESolver Hes(H_e);
                return Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(0.0).asDiagonal() * Hes.eigenvectors().transpose();
            },
            [&](size_t hingeIndex) { return m_hingeStencils[hingeIndex].blockVars; }
        );
    }

    virtual CSCMat hessianSparsityPattern(double val = 0.0, VariableMask vmask = VariableMask::Defo) const {
        CSCMat Hsp_block = m_assembler.blockSparsityPattern(numHinges(),
            [this](size_t hingeIdx) { return m_hingeStencils[hingeIdx].blockVars; });
        return m_assembler.blockHessianSparsityPatternToScalar(Hsp_block, val);
    }

    NodePositions getCornerPositions(size_t ei) const {
        return MembraneStencil(mesh().template elementNodeIndices<int>(ei)).extract<NodePositions>(m_x, m_assembler.vars());
    }

    void setDelta(double delta) {
        for (auto &he : m_hingeElements)
            he.material.delta = delta;
    }

    double getDelta() const {
        return m_hingeElements[0].material.delta;
    }

    double bendingStiffness = 1.0;
    double h = 1; // sheet thickness

private:
    // update the deformed/rest states.
    virtual void m_setDefoVars(const Eigen::Ref<const VXd> &vars) {
        m_x = vars;

        // Deform edge hinge element.
        for (size_t hi = 0; hi < numHinges(); ++hi)
            m_hingeElements[hi].setDeformedConfiguration(hingeVars(hi));
    }

    virtual void m_setRestVars(const Eigen::Ref<const VXd> &vars) {
        mesh().setNodePositions(Eigen::Map<const MX3d>(vars.data(), numVertices(), 3));
        for (auto &d : m_elementData)
            d.embeddingUpdated();
    }

    std::shared_ptr<Mesh> m_mesh;
    VXd m_x;
    SystemAssembler<3> m_assembler;

    std::vector<HingeStencil> m_hingeStencils;
    std::vector<HE> m_hingeElements;
    Psi m_psi; // Membrane energy density function.

    std::vector<elements::EmbeddedMembraneElementData<typename Mesh::ElementData>> m_elementData;
};
