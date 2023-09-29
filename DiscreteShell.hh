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
#include <MeshFEM/ElasticElement.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <memory>
#include <vector>

template<class HalfEdge>
inline std::array<int, 4> bendingHingeStencil(const HalfEdge &he) {
    assert(he.isPrimary() && !he.isBoundary());
    return {{ he.tail().index(),
              he.tip ().index(),
              he.opposite().next().tip().index(),
              he           .next().tip().index() }};
}

template<template<typename> class HingeEnergy>
struct DiscreteShell : public ElasticObject<double> {
    static constexpr size_t Deg = 1;
    static constexpr size_t K = 2;
static constexpr size_t N = 3;

    using V3d  = Eigen::Vector3d;
    using VXd  = Eigen::VectorXd;
    using MX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>; // Row major so that flattened order agrees with VField

    using HE = HingeEnergy<double>;
    using ME = MembraneElement<Real, K, Deg>;
    using NodePositions = typename ME::NodePositions;

    using Mesh = FEMMesh<K, Deg, V3d>; // Linear triangle mesh embedded in 3d.
    using Psi_2x2 = NeoHookeanEnergy<double, K>; // 2d energy density used to define the membrane energy
    using Psi = AutoHessianProjection<MembraneEnergyDensityFrom2x2Density<Psi_2x2>>;



    DiscreteShell(const std::shared_ptr<Mesh> &m, double Y = 200, double nu = 0.3)
        : m_mesh(m), m_assembler(m->numVertices()),
          m_psi(Y * nu / ((1 + nu) * (1 - 2 * nu)), // 3D Lame parameter lambda (plane stress conditions are applied inside `NeoHookeanEnergy`)
                Y / (2 * (1 + nu)))                 // Shear modulus mu
    {
        // Initialize deformed positions using the mesh's rest vertex positions.
        m_x = getV(*m);

        // Construct hinges.
        for (const auto &he : mesh().halfEdges()) {
            if (!he.isPrimary() || he.isBoundary()) continue;
            m_halfedgeForHinge.push_back(he.index());

            auto stencil = bendingHingeStencil(he);
            m_edgeHinges.emplace_back(m_x.row(stencil[0]).transpose(),
                                      m_x.row(stencil[1]).transpose(),
                                      m_x.row(stencil[2]).transpose(),
                                      m_x.row(stencil[3]).transpose());
        }

        // Construct EmbeddedMembraneElementData
        const size_t ne = m->numElements();
        m_elementData.reserve(ne);
        for (size_t ei = 0; ei < ne; ++ei)
            m_elementData.emplace_back(*(m->element(ei)));
    }

    auto hingeStencil(size_t hingeIndex) const {
        return bendingHingeStencil(mesh().halfEdge(m_halfedgeForHinge[hingeIndex]));
    }
          Mesh &mesh()       { return *m_mesh; }
    const Mesh &mesh() const { return *m_mesh; }

    size_t numVertices() const { return mesh().numVertices(); }
    size_t   numHinges() const { return m_edgeHinges.size(); }

    virtual size_t numDefoVars() const { return 3 * numVertices(); }
    virtual size_t numRestVars() const { return 3 * numVertices(); }

    virtual VXd getDefoVars() const { return Eigen::Map<const VXd>(m_x.data(),          numDefoVars()); }
    virtual VXd getRestVars() const { return Eigen::Map<const VXd>(getV(mesh()).data(), numDefoVars()); }
    const MX3d &deformedPositions() const { return m_x; }

    virtual double energy() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("DiscreteShell.energy");
        double result = 0;
        const auto &m = mesh();

        // Discrete shells bending energy term
        for (const auto &hinge : m_edgeHinges)
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
            return (bendingStiffness * m_edgeHinges[hi].gradient()).eval();
        }, [this](size_t hi) { return hingeStencil(hi); });

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
                auto H_e = (bendingStiffness * m_edgeHinges[hingeIndex].hessian()).eval();
                if (!projectionMask) return H_e;

                ESolver Hes(H_e);
                return Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(0.0).asDiagonal() * Hes.eigenvectors().transpose();
            },
            [&](size_t hingeIndex) { return hingeStencil(hingeIndex); }
        );
    }

    virtual CSCMat hessianSparsityPattern(double val = 0.0, VariableMask vmask = VariableMask::Defo) const {
        const auto &m = mesh();
        CSCMat Hsp_block = m_assembler.blockSparsityPattern(numHinges(),
            [this, &m](size_t hingeIdx) {
                return bendingHingeStencil(m.halfEdge(m_halfedgeForHinge[hingeIdx]));
            });
        return m_assembler.blockHessianSparsityPatternToScalar(Hsp_block, val);
    }

    NodePositions getCornerPositions(size_t ei) const {
        const auto &e = mesh().element(ei);
        NodePositions result;
        result << m_x.row(e.vertex(0).index()),
                  m_x.row(e.vertex(1).index()),
                  m_x.row(e.vertex(2).index());
        return result;
    }

    void setDelta(Real delta) {
        for (auto &he : m_edgeHinges)
            he.delta = delta;
    }

    Real getDelta() const {
        return m_edgeHinges[0].delta;
    }

    double bendingStiffness = 1.0;
    double h = 1; // sheet thickness

private:
    // update the deformed/rest states.
    virtual void m_setDefoVars(const Eigen::Ref<const VXd> &vars) {
        m_x = Eigen::Map<const MX3d>(vars.data(), numVertices(), 3);

        // Deform edge hinge element.
        for (size_t hi = 0; hi < numHinges(); ++hi) {
            auto stencil = hingeStencil(hi);
            m_edgeHinges[hi].setDeformedConfiguration(m_x.row(stencil[0]).transpose(),
                                                      m_x.row(stencil[1]).transpose(),
                                                      m_x.row(stencil[2]).transpose(),
                                                      m_x.row(stencil[3]).transpose());
        }
    }

    virtual void m_setRestVars(const Eigen::Ref<const VXd> &vars) {
        mesh().setNodePositions(Eigen::Map<const MX3d>(vars.data(), numVertices(), 3));
        for (auto &d : m_elementData)
            d.embeddingUpdated();
    }

    std::shared_ptr<Mesh> m_mesh;
    MX3d m_x;
    SystemAssembler<3> m_assembler;

    std::vector<HE> m_edgeHinges;
    std::vector<int> m_halfedgeForHinge;
    Psi m_psi; // Membrane energy density function.

    std::vector<EmbeddedMembraneElementData<typename Mesh::ElementData>> m_elementData;
};
