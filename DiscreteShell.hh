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
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include "HingeEnergy.hh"
#include <memory>
#include <vector>

struct DiscreteShell : public ElasticObject<double> {
    using V3d  = Eigen::Vector3d;
    using M32d = Eigen::Matrix<double, 3, 2>;
    using VXd  = Eigen::VectorXd;
    using MX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>; // Row major so that flattened order agrees with VField

    using Mesh = FEMMesh<2, 1, V3d>; // Linear triangle mesh embedded in 3d.
    using Psi_2x2 = NeoHookeanEnergy<double, 2>; // 2d energy density used to
                                                 // define the membrane energy
    using Psi = AutoHessianProjection<MembraneEnergyDensityFrom2x2Density<Psi_2x2>>;
    static constexpr size_t N = 3;

    DiscreteShell(const std::shared_ptr<Mesh> &m, double Y = 200, double nu = 0.3)
        : m_mesh(m), m_assembler(m->numVertices()),
          m_psi(Y * nu / ((1 + nu) * (1 - 2 * nu)), // 3D Lame parameter lambda (plane stress conditions are applied inside `NeoHookeanEnergy`)
                Y / (2 * (1 + nu)))                 // Shear modulus mu
    {
        // Construct basis for each triangle.
        m_updateB(); 

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
        double result = 0;
        const auto &m = mesh();

        // Discrete shells bending energy term
        for (const auto &hinge : m_edgeHinges)
            result += bendingStiffness * hinge.energy();

        // Elastic energy term
        Psi psi(m_psi, UninitializedDeformationTag());
        for (auto e : m.elements()) {
            psi.setDeformationGradient(getDeformationGradient(e.index()), EvalLevel::EnergyOnly);
            result += psi.energy() * (h * e->volume());
        }

        return result;
    }

    virtual VXd gradient(bool updatedParametrization = false, VariableMask vmask = VariableMask::Defo) const {
        VXd g = VXd::Zero(numVars());
        const auto &m = mesh();

        // Membrane energy contribution
        Psi psi(m_psi, UninitializedDeformationTag());
        for (size_t ei = 0; ei < m.numElements(); ++ei) {
            auto e = m.element(ei);
            psi.setDeformationGradient(getDeformationGradient(ei), EvalLevel::Gradient);
            M32d psi_prime = psi.denergy() * (e->volume() * h);
            for (auto v : e.vertices())
                g.segment<3>(3 * v.index()) += psi_prime * m_jacobianBarycentricB[ei].row(v.localIndex()).transpose();
        }

        // Bending energy contribution
        for (size_t hi = 0; hi < m_halfedgeForHinge.size(); ++hi) {
            auto stencil = hingeStencil(hi);
            auto gradHingeEnergy = m_edgeHinges[hi].gradient();
            for (size_t svi = 0; svi < stencil.size(); ++svi)
                g.segment<3>(3 * stencil[svi]) += bendingStiffness * gradHingeEnergy.col(svi);
        }
        return g;
    }

    virtual void hessian(CSCMat &H, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const {
        // Assemble membrane term.
        projectionMask = false;
        m_assembler.assembleHessian(H, mesh(), [&](size_t ei) {
            const auto &e = mesh().element(ei);
            Psi psi(m_psi, UninitializedDeformationTag());
            psi.setDeformationGradient(getDeformationGradient(ei), projectionMask ? EvalLevel::Hessian
                                                                                  : EvalLevel::HessianWithDisabledProjection);

            M32d deltaF = M32d::Zero();
            Eigen::Matrix<Real, 9, 9> result;
            for (size_t lni_b = 0; lni_b < e.numVertices(); ++lni_b) {
                for (size_t c_b = 0; c_b < N; ++c_b) {
                    size_t var_b = N * lni_b + c_b;
                    deltaF.row(c_b) = m_jacobianBarycentricB[ei].row(lni_b);

                    M32d delta_psi_prime = psi.delta_denergy(deltaF);
                    for (size_t lni_a = 0; lni_a < e.numVertices(); ++lni_a)
                        result.block<N, 1>(N * lni_a, var_b) = delta_psi_prime * m_jacobianBarycentricB[ei].row(lni_a).transpose();

                    deltaF.row(c_b).setZero();
                }
            }

            return result * (h * e->volume());
        });

        // Assemble bending term.
        m_assembler.assembleHessian(H, numHinges(),
            [&](size_t hingeIndex) { return (bendingStiffness * m_edgeHinges[hingeIndex].hessian()).eval(); },
            [&](size_t hingeIndex) { return hingeStencil(hingeIndex); }
        );
    }

    virtual CSCMat hessianSparsityPattern(double val = 0.0, VariableMask vmask = VariableMask::Defo) const {
        const auto &m = mesh();
        CSCMat Hsp_block = m_assembler.blockSparsityPattern(m.numVertices(), numHinges(),
            [this, &m](size_t hingeIdx) {
                return bendingHingeStencil(m.halfEdge(m_halfedgeForHinge[hingeIdx]));
            });
        return m_assembler.blockHessianSparsityPatternToScalar(Hsp_block, val);
    }

    // Compute `FB`, the deformation gradient expressed in the
    // basis of the element's tangent plane.
    M32d getDeformationGradient(size_t ei) const {
        return getCornerPositions(ei) * m_jacobianBarycentricB[ei];
    }

    Eigen::Matrix3d getCornerPositions(size_t ei) const {
        const auto &e = mesh().element(ei);
        Eigen::Matrix3d result;
        result << m_x.row(e.vertex(0).index()).transpose(),
                  m_x.row(e.vertex(1).index()).transpose(),
                  m_x.row(e.vertex(2).index()).transpose();
        return result;
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
        m_updateB();
    }

    void m_updateB() {
        // Generate an orthonormal basis for the tangent plane of each triangle.
        const auto &m = mesh();
        const size_t nt = m.numTris();

        m_B.resize(nt);
        for (auto tri : m.elements()) {
            V3d b0 = (tri.node(1)->p - tri.node(0)->p).normalized();
            V3d b1 = tri->normal().cross(b0);
            const size_t ti = tri.index();
            m_B[ti].col(0) = b0;
            m_B[ti].col(1) = b1;
        }

        m_jacobianBarycentricB.reserve(nt);
        m_jacobianBarycentricB.clear();
        for (const auto e : m.elements())
            m_jacobianBarycentricB.push_back(e->gradBarycentric().transpose() * m_B[e.index()]);
    }

    std::shared_ptr<Mesh> m_mesh;
    MX3d m_x;
    SystemAssembler<3> m_assembler;

    std::vector<HingeEnergy<double>> m_edgeHinges;
    std::vector<int> m_halfedgeForHinge;
    Psi m_psi; // Membrane energy density function.

    // Orthonormal basis for each reference triangle's tangent space
    std::vector<M32d> m_B;
    std::vector<M32d> m_jacobianBarycentricB;
};
