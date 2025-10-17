////////////////////////////////////////////////////////////////////////////////
// SimpleElasticSolid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Demonstrates the lowest-level SystemAssembler interface;
//  many objectives can instead be implemented using the `MeshEnergy` class.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/16/2025 21:30:06
*///////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/Stencils.hh>
#include <MeshFEM/Elements/SolidElement.hh>

template<size_t N, size_t Degree, class Psi>
struct SimpleElasticSolid : public NewtonObjectiveTermBase {
    static constexpr size_t NodesPerElement = Simplex::numNodes(N, Degree);

    using Assembler = SystemAssembler<N>;
    using Stencil   = ElementStencil</* Simplex Dimension */ N, Degree, /* Embedding Dimension */ N>;
    using SE        = SolidElement<Degree, Psi, LinearlyEmbeddedElement<N, Degree, VecN_T<double, N>>>;
    using VXd       = Eigen::VectorXd;

    SimpleElasticSolid(const Eigen::MatrixXd &V,
                       const Eigen::Matrix<int, Eigen::Dynamic, NodesPerElement> &T,
                       std::shared_ptr<NewtonVarsBase> vars)
        : m_assembler(V.rows()), m_vars(vars), materials(T.rows())
    {
        for (int ei = 0; ei < T.rows(); ++ei) {
            m_elements.emplace_back(ei, V, T, materials);
            m_stencils.emplace_back(T, ei);
        }
    }

    Real objective() const override {
        const VXd &x = m_vars->getVars();
        return summation_parallel([&](size_t ei) { return m_elements[ei].energy(extractLocalVars(ei, x)); }, m_elements.size());
    }

    void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        const VXd &x = m_vars->getVars();
        m_assembler.assembleGradient(g, m_elements.size(),
                [&](size_t ei) { return m_elements[ei].gradient(weight, extractLocalVars(ei, x)); },
                [&](size_t ei) { return getStencil(ei); });
    }

    void accumulateHessian(Real weight, NewtonHessian &H, bool projectionMask = false) const override {
        const VXd &x = m_vars->getVars();
        m_assembler.assembleHessian(H, m_elements.size(),
                [&](size_t ei) { return m_elements[ei].hessian(weight, projectionMask, extractLocalVars(ei, x)); },
                [&](size_t ei) { return getStencil(ei); });
    };

    NewtonHessian hessianSparsityPattern() const override {
        return m_assembler.sparsityPattern(m_elements.size(), [this](size_t ei) { return getStencil(ei); });
    }

    auto extractLocalVars(size_t si, const VXd &globalVars) const {
        return m_stencils[si].template extract<typename SE::LocalVars>(globalVars, m_assembler.varStructure());
    }

    auto getStencil(size_t ei) const { return m_stencils[ei].blockVars; }

private:
    std::vector<SE> m_elements;
    std::vector<Stencil> m_stencils;

    Assembler m_assembler;
    std::shared_ptr<NewtonVarsBase> m_vars;

public:
    MaterialAssignment<typename SE::Material> materials;
};
