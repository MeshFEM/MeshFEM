#ifndef MEMBRANEELEMENT_HH
#define MEMBRANEELEMENT_HH

#include <MeshFEM/EmbeddedElement.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include "ElementBase.hh"
#include "HyperelasticLagrange.hh"

namespace elements {

// Data for a triangular membrane element whose *rest configuration* is embedded
// in 3D. This is useful for simulating shells (deformed configuration also
// embedded in 3D) and computing parametrizations (deformed configuration
// embedded in 2D). This class enriches a triangular `LinearlyEmbeddedElement`
// with an orthonormal basis for its tangent plane and cached shape function
// gradients in this 2D coordinate system.
template<class LEElement, class StorageType = const LEElement &>
struct EmbeddedMembraneElementData {
    static constexpr size_t K = LEElement::K;
    static constexpr size_t N = LEElement::EmbeddingSpace::RowsAtCompileTime;
    static constexpr size_t numNodes    = LEElement::numNodes;
    static constexpr size_t numVertices = LEElement::numVertices;
    static constexpr size_t Deg      = LEElement::Deg;

    EmbeddedMembraneElementData(const LEElement &ee) : m_embeddedElement(ee) {
        embeddingUpdated();
    }

    static_assert((K == 2) && (N == 3), "Only intended for triangles embedded in 3D");

    using M32d = Eigen::Matrix<Real, 3, 2>;
    using M23d = Eigen::Matrix<Real, 2, 3>;

    // Evaluated shape function gradients
    using GradPhis = Eigen::Matrix<Real, 2, numNodes>;

    const M23d &BtGradBarycentric() const { return m_BtGradBarycentric; }
    const M32d &B() const { return m_B; }
    void setB(const M32d &B) { m_B = B; }

    const M23d &gradPhis() const {
        if constexpr (Deg == 1) return m_BtGradBarycentric;
        throw std::runtime_error("This method is only meant for linear elements!");
    }

    GradPhis gradPhis(const EvalPt<K> &x) const {
        if constexpr (Deg == 1) { return m_BtGradBarycentric; }
        if constexpr (Deg == 2) {
            GradPhis result;
            EigenEvalPt<K> x4 = 4 * Eigen::Map<const EigenEvalPt<K>>(x.data());
            result.leftCols(numVertices).noalias() = m_BtGradBarycentric * (x4.array() - 1.0).matrix().asDiagonal();
            for (size_t j = 0; j < Simplex::numEdges(K); ++j) {
                const size_t start = Simplex::edgeStartNode(j),
                             end   = Simplex::  edgeEndNode(j);
                result.col(numVertices + j) = x4[  end] * m_BtGradBarycentric.col(start)
                                            + x4[start] * m_BtGradBarycentric.col(  end);
            }
            return result;
        }
        static_assert(Deg == 1 || Deg == 2, "Higher degrees not implemented");
    }

    Real volume() const { return m_embeddedElement.volume(); }

    // Recompute the orthonormal basis and the projected shape function gradients.
    void embeddingUpdated() {
        const auto &gradLambda = m_embeddedElement.gradBarycentric();
        const auto &n = m_embeddedElement.normal();

        // First, check if the triangle is parallel to the z=0 plane; in this
        // case we use the global 2D coordinate system's axis vectors as our
        // orthonormal basis to ease specification of anisotropic materials.
        if (n.template head<2>().squaredNorm() < 1e-32)
            m_B.setIdentity();
        else {
            // We pick an orthonormal basis with b_0 parallel to e_0 and
            // b_1 parallel to e_0^perp (also parallel to "grad lambda_0")
            m_B.col(1) = gradLambda.col(0).normalized();
            m_B.col(0) = -n.cross(m_B.col(1));
        }
        m_BtGradBarycentric = m_B.transpose() * gradLambda;
    }

private:
    M32d m_B;
    M23d m_BtGradBarycentric;
    StorageType m_embeddedElement;
};

template<size_t K, size_t Deg, class VNd>
using EmbeddedMembraneEData = EmbeddedMembraneElementData<LinearlyEmbeddedElement<K, Deg, VNd>>;

}

template<class Psi_2x2>
struct MembraneMaterial : public MaterialBase {
    using Real = typename Psi_2x2::Real;
    using Psi = AutoHessianProjection<MembraneEnergyDensityFrom2x2Density<Psi_2x2>>;
    Psi psi;
    Real thickness = 1;
};

template<size_t Deg, class Psi_2x2, class CustomMat_>
struct MembraneElement;

template<size_t Deg, class Psi_2x2, class CustomMat_>
struct ElementTraits<MembraneElement<Deg, Psi_2x2, CustomMat_>> {
    using Material = CustomMat_;
};

template<size_t Deg, class Psi_2x2, class CustomMat_ = MembraneMaterial<Psi_2x2>>
struct MembraneElement : public ElementBase<MembraneElement<Deg, Psi_2x2, CustomMat_>> {
    static constexpr size_t K = 2;
    static constexpr size_t N = 3;
    using Real     = typename Psi_2x2::Real;
    using Base     = ElementBase<MembraneElement>;
    using Material = typename Base::Material;

    using HLE = elements::HyperelasticLagrange<typename Material::Psi, K, N, Deg>;
    using LocalVars = typename HLE::NodePositions;
    using Gradient  = typename HLE::Gradient;
    using Hessian   = typename HLE::Hessian;

    static std::string name() { return "Membrane"; }

    static constexpr bool CachesDeformedQuantities = false;

    template<class Mesh>
    MembraneElement(size_t ei, const Mesh &m, MaterialAssignment<Material> &materials)
        : Base(ei, materials), elementData(*(m.element(ei))) { }

    auto FBGetter(const LocalVars &x) const { return typename HLE::ElasticFGetter(x); }
    auto getFB(const LocalVars &x) const { return FBGetter(x)(elementData.gradPhis()); }

    // void setRestConfiguration(const LocalVars &X) {
    // }

    Real       energy(                                const LocalVars &x) const { const auto &m = Base::material(); return HLE::  energy(m.psi, FBGetter(x), elementData) * m.thickness; }
    Gradient gradient(Real weight,                    const LocalVars &x) const { const auto &m = Base::material(); return HLE::gradient(m.psi, FBGetter(x), elementData, (weight * m.thickness)); }
    template<bool SetLowerTri = false>
    Hessian hessian(Real weight, bool projectionMask, const LocalVars &x) const { const auto &m = Base::material(); return HLE::template hessian<SetLowerTri>(m.psi, FBGetter(x), elementData, /* projectionDisabled  = */ !projectionMask, (weight * m.thickness)); }

    elements::EmbeddedMembraneEData<K, Deg, VecN_T<Real, N>> elementData;
};

#include "../MeshEnergy.hh"

template<class Psi_2x2, size_t Deg = 1>
using MembraneMeshEnergy = MeshEnergy<FEMMesh<2, Deg, Vector3D>, NodalVars<3>, ElementStencil<2, Deg, 3>, MembraneElement<Deg, Psi_2x2>>;

#endif /* end of include guard: MEMBRANEELEMENT_HH */
