////////////////////////////////////////////////////////////////////////////////
// ElasticSheet.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Simulate an thin elastic sheet with a potentially curved rest configuration
//  (modeling either plates or shells). The simulation consists of a membrane
//  term (capturing the energy due to in-plane stretching) and a bending energy
//  term.
//
//  The sheet is made of a material described by a generic (possibly
//  anisotropic) "C-based" energy density `Psi_C`, which is a function of the
//  2x2 Cauchy deformation tensor. Typically this `Psi_c` is obtained by
//  applying plane stress assumptions to some volumetric hyperelastic model.
//
//  However, our bending energy implementation is only really justified for
//  sheets made of a St. Venant-Kirchhoff material, where membrane and bending
//  strains neatly decouple as a fortunate consequence of the linear
//  constitutive law. For other material models, the bending energy term should
//  technically plug the shape operator into a quadratic form defined by the
//  tangent elasticity tensor evaluated at the current membrane strain (for
//  St.VK, this is just the energy density quadratic form itself).
//  Unfortunately, that means the elastic energy gradient and Hessian would
//  involve third and fourth derivatives of psi.
//  Note that the derivation of the  shell energy expression assumes small
//  strain to drop certain terms, so our simplified implementation should not
//  be a significant additional source of error.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/16/2020 18:16:58
////////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICSHEET_HH
#define ELASTICSHEET_HH

#include "FEMMesh.hh"
#include "GaussQuadrature.hh"
#include "GlobalBenchmark.hh"
#include "MeshIO.hh"
#include "ParallelAssembly.hh"
#include "SparseMatrices.hh"
#include "Types.hh"
#include "EnergyDensities/Tensor.hh"
#include "EnergyDensities/EnergyTraits.hh"
#include "Geometry.hh"

#include "RigidMotionPins.hh"
#include "ElasticObject.hh"

// Note: anisotropic materials are supported and, for plates (sheets with
// perfectly flat rest states), the anisotropic energy density function can be
// intuitively expressed in the global 2D coordinate system; this is
// enabled by the special case in m_updateB that uses the global x and y axes.
// However for shells with non-flat rest states, the energy density function is
// expressed in terms of each triangle's distinct orthonormal coordinate system
// m_B, which is probably quite inconvenient...
//
// TODO: option to disable midedge normals/bending energy.
template <class _Psi_C>
class ElasticSheet : public ElasticObject<typename _Psi_C::Real> {
public:
    using QuadratureRule = Quadrature<3, 1>; // Due to the bending strain discretization we use only linear FEM
    using EvalPtN = EvalPt<3>;

    using Psi_C = _Psi_C;
    using Real  = typename Psi_C::Real;
    using V2d   = Eigen::Matrix<Real, 2, 1>;
    using V3d   = Eigen::Matrix<Real, 3, 1>;
    using M2d   = Eigen::Matrix<Real, 2, 2>;
    using M3d   = Eigen::Matrix<Real, 3, 3>;
    using M32d  = Eigen::Matrix<Real, 3, 2>;
    using VXd   = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using MX3d  = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>; // Row major so that flattened order agrees with VField
    using MX2d  = Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>;
    using Frame = M3d; // Columns are [tangent, d1, d2], a right-handed orthonormal frame adapted to a particular edge tangent.

    static constexpr size_t K   = 2;
    static constexpr size_t Deg = 1;
    static constexpr size_t N   = 3;
    static constexpr size_t numNodesPerElement  = Simplex::numNodes(K, Deg);
    static constexpr size_t numElementLocalVars = N * numNodesPerElement + 3;
    using  Mesh = FEMMesh<2, Deg, V3d>;
    using TMesh = typename Mesh::BaseMesh; // TriMesh data structure underlying FEMMesh
    using VSFJ = VectorizedShapeFunctionJacobian<3, V3d>;

    using  HEHandle = typename TMesh::template HEHandle<      TMesh>;
    using CHEHandle = typename TMesh::template HEHandle<const TMesh>;
    using  CTHandle = typename TMesh::template  THandle<const TMesh>;

    enum class EnergyType { Full, Membrane, Bending };

    ElasticSheet(const std::shared_ptr<Mesh> &m, const Psi_C &psi) : m_mesh(m), m_psi{{psi}},
                                                                     m_numVertices(m->numVertices()),
                                                                     m_numEdges   (m->numEdges())
    {
        m_updateB();

        // Build the halfedge -> edge map.
        m_edgeForHalfEdge.resize(m->numHalfEdges());
        m->visitEdges([this](CHEHandle he, size_t edgeIndex) {
            m_edgeForHalfEdge.at(he.index()) = edgeIndex;
            auto hopp = he.opposite();
            if (hopp) m_edgeForHalfEdge.at(hopp.index()) = edgeIndex;
        });

        setIdentityDeformation();

        // Apply this resulting shape operator as the rest shape operator
        // (To handle curved shells.)
        m_restII = m_II;
    }

    const Mesh &mesh() const { return *m_mesh; }
          Mesh &mesh()       { return *m_mesh; }

    // The variables consist of deformed vertex positions and midedge normal angles.
    size_t numVars() const {
        return 3 * m_numVertices
                 + m_numEdges;
    }
    size_t numThetas() const { return m_numEdges; }

    size_t thetaOffset() const { return 3 * m_numVertices; }

    void setThickness(Real thickness) {
        m_h = thickness;
    }

    Real getThickness() const { return m_h; }
    size_t edgeForHalfEdge(size_t hei) const { return m_edgeForHalfEdge.at(hei); }

    virtual void setVars(Eigen::Ref<const VXd> vars) override {
        if (size_t(vars.rows()) != numVars()) throw std::runtime_error("Invalid vars size");
        m_thetas = vars.tail(m_numEdges);
        setDeformedPositions(Eigen::Map<const MX3d>(vars.data(), m_numVertices, 3));
    }

    void setDeformedPositions(Eigen::Ref<const MX3d> x) {
        if (size_t(x.rows()) != m_numVertices) throw std::runtime_error("Invalid x size");
        m_deformedPositions = x;
        m_updateDeformedElements();
        m_adaptReferenceFrame(); // Side effect: update shape operators/midedge normals

        this->m_deformedConfigUpdated();
    }

    const VXd &getThetas() const { return m_thetas; }

    void setThetas(Eigen::Ref<const VXd> thetas) {
        if (size_t(thetas.rows()) != m_numEdges) throw std::runtime_error("Invalid thetas size");
        m_thetas = thetas;

        m_updateShapeOperators();
        m_updateMidedgeNormals();

        this->m_deformedConfigUpdated();
    }

    VXd getVars() const {
        VXd result(numVars());
        result.head(3 * m_numVertices) = Eigen::Map<const VXd>(m_deformedPositions.data(), 3 * m_numVertices);
        result.tail(m_numEdges) = m_thetas;
        return result;
    }

    MX3d deformedPositions() const { return m_deformedPositions; }
    VXd  thetas()            const { return m_thetas;            }

    MX3d restPositions() const {
        const auto &m = mesh();
        MX3d rpos(m.numNodes(), 3);
        for (const auto &n : m.nodes())
            rpos.row(n.index()) = n->p;
        return rpos;
    }

    MX3d nodeDisplacements() const { return deformedPositions() - restPositions(); }

    const Psi_C &getEnergyDensity(size_t ei) const {
        if (m_psi.size() == 1) return m_psi.front();
        return m_psi.at(ei);
    }

    Real elementEnergy(size_t ei, const EnergyType etype) const;
    Real energy(const EnergyType etype) const;

    using ElementGradient = Eigen::Matrix<Real, numElementLocalVars, 1>;
    ElementGradient elementGradient(size_t, bool updatedSource, const EnergyType etype) const;
    VXd  gradient(bool updatedSource, const EnergyType etype = EnergyType::Full) const;

    // using PerElementHessian = Eigen::Matrix<Real, flatLen(numElementLocalVars), 1>;
    // PerElementHessian elementHessian(size_t ei, const EnergyType etype, bool projectionMask = false) const;

    void hessian(SuiteSparseMatrix &Hout, const EnergyType etype, bool projectionMask = false) const;
    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const override;

    // Overloads implementing generic ElasticObject interface.
    virtual Real  energy() const override { return energy(EnergyType::Full); }
    virtual VXd gradient() const override { return gradient(false, EnergyType::Full); }
    virtual void hessian(SuiteSparseMatrix &Hout, bool projectionMask = false) const override { hessian(Hout, EnergyType::Full, projectionMask); }

    template <class SHEHandle>
    M3d d_A_gamma_div_len_d_x(const SHEHandle &he, bool updatedSource) const;
    template <class SHEHandle>
    M3d d2_A_gamma_div_len_d_x_dtheta(const SHEHandle &he) const;
    template <class SHEHandle, class SVHandle>
    M3d delta_d_A_gamma_div_len_d_x(const SHEHandle &he, const SVHandle &v_b, const size_t c_b) const;

    const MX3d &midedgeNormals()                       const { return m_midedgeNormals; }
    const std::vector<Frame> &midedgeReferenceFrames() const { return m_referenceFrame; }
    const std::vector<Frame> & sourceReferenceFrames() const { return m_sourceReferenceFrame; }

    // For debugging visualizations of the edge frames, we need their application points
    MX3d edgeMidpoints() const {
        MX3d result(m_numEdges, 3);
        mesh().visitEdges([this, &result](CHEHandle he, size_t edgeIndex) {
            result.row(edgeIndex) = 0.5 * (m_deformedPositions.row(he.tip().index())
                                         + m_deformedPositions.row(he.tail().index()));
        });
        return result;
    }
    // To assist boundary conditions specification
    MX3d restEdgeMidpoints() const {
        MX3d result(m_numEdges, 3);
        mesh().visitEdges([this, &result](CHEHandle he, size_t edgeIndex) {
            result.row(edgeIndex) = 0.5 * (mesh().node(he. tip().index())->p +
                                           mesh().node(he.tail().index())->p);
        });
        return result;
    }

    // Apply an identity deformation and reset the source frame representation.
    // Note, we set the undeformed midedge normals by minimizing the bending energy
    // (since only a mesh is provided as input, these midedge normals are not
    // specified).
    void setIdentityDeformation();

    // (Re-)initialize the midedge normals, inferring them from the midsurface.
    void initializeMidedgeNormals(bool minimizeBending = true);

    void updateSourceFrame() {
        m_sourceReferenceFrame = m_referenceFrame;
        m_sourceAlphas         = m_alphas;
    }

    // Update our parametrizaton of the system's DoFs
    // (currently this just means updating the source frames.)
    void updateParametrization() { updateSourceFrame(); }

    template<class HEType>
    auto deformedEdgeVector(const HEType &he) const {
        return (m_deformedPositions.row(he. tip().index())
             - m_deformedPositions.row(he.tail().index())).eval();
    }
    const auto &deformedElement(size_t ei) const { return m_deformedElements.at(ei); }

    // Get the deformed positions of triangle ti's corners as columns
    // of a 3x3 matrix.
    M3d getCornerPositions(size_t ti) const {
        const auto &t = mesh().tri(ti);
        M3d result;
        result << m_deformedPositions.row(t.vertex(0).index()).transpose(),
                  m_deformedPositions.row(t.vertex(1).index()).transpose(),
                  m_deformedPositions.row(t.vertex(2).index()).transpose();
        return result;
    }

    // Get the deformed/rest second fundamental forms
    const std::vector<M3d>   &getII()     const { return m_II;     }
    const std::vector<M3d>   &getRestII() const { return m_restII; }
    const std::vector<M32d>  &getB()      const { return m_B;      }

    // Get the per-element right Cauchy-Green deformation tensors/first
    // fundamentals form representing the deformation.
    std::vector<M2d>  getC() const {
        std::vector<M2d> C;
        const auto &m = mesh();
        C.reserve(m.numElements());
        for (const auto e : m.elements()) {
            const size_t ei = e.index();
            M32d FB = getCornerPositions(ei) * (e->gradBarycentric().transpose() * m_B[ei]);
            C.push_back(FB.transpose() * FB);
        }
        return C;
    }

    const VXd &getAlphas()       const { return m_alphas;       }
    const VXd &getSourceAlphas() const { return m_sourceAlphas; }
    VXd        getGammas() const {
        const auto &m = mesh();
        VXd gammas(m.numHalfEdges());
        for (const auto &he : m.halfEdges()) {
            // The current triangle's shape operator is defined in terms of the
            // angle gamma between the triangle normal and midedge normal
            // ***around the oriented edge vectors***. But thetas/alphas are
            // defined as angles around the primary halfedge vector (which may
            // point in the opposite direction). Therefore we must negate gamma
            // for non-primary half edges.
            double sign = he.isPrimary() ? 1.0 : -1.0;
            gammas[he.index()] = sign * (m_thetas[m_edgeForHalfEdge[he.index()]] - m_alphas[he.index()]);
        }
        return gammas;
    }

    // Get the principal curvatures of the deformed sheet geometry.
    MX2d getPrincipalCurvatures() const;

    // The volume associated with a shell element is area * thickness.
    VXd element3DVolumes() const {
        const auto &m = mesh();
        VXd result(m.numElements());
        for (const auto &e : m.elements())
            result[e.index()] = e->volume() * m_h;
        return result;
    }

    // Apply a rigid transformation `x --> R x + t` to the deformed configuration.
    // Rotating the deformed configuration is slightly complicated by needing
    // to maintain source and current reference frames...
    void applyRigidTransform(const M3d &R, const V3d &t) {
        if (((R.transpose() * R - M3d::Identity()).norm() > 1e-8) || (R.determinant() < 0))
            throw std::runtime_error("R is not a rotation");

        // Rotate the source reference frame so that setDeformedConfiguration()
        // produces the correct normals/shape operators/reference frame...
        for (size_t i = 0; i < m_numEdges; ++i)
            m_sourceReferenceFrame[i] = (R * m_sourceReferenceFrame[i]).eval();

        auto prerotationFrames = m_referenceFrame; // for validation
        setDeformedPositions((m_deformedPositions * R.transpose()).rowwise() + t.transpose());

        for (size_t i = 0; i < m_numEdges; ++i) {
            if ((m_referenceFrame[i] - R * prerotationFrames[i]).norm() > 1e-8)
                throw std::runtime_error("Frame update failure");
        }
    }

    // Reorient the current deformed configuration so that global rigid motions
    // can be pinned down with just 6 variable pin constraints.
    // Also return the indices of these 6 variables.
    using RMPins = RigidMotionPins<ElasticSheet>;
    typename RMPins::PinInfo
    prepareRigidMotionPins() {
        return RMPins::run(*this);
    }

    void filterRMPinArtifacts(const typename RMPins::PinVertices &pinVertices) {
        throw std::runtime_error("Unimplemented");
        // ::filterRMPinArtifacts(*this, pinVertices);
    }

    void setDisabledBending(bool yesno) { m_disableBending = yesno; }
    bool getDisabledBending() const { return m_disableBending; }

private:
    // Update the current midedge reference frame to adapt to the new deformed
    // edge tagents. This also calls m_updateMidedgeNormals and m_updateShapeOperators.
    void m_adaptReferenceFrame();

    // Update the midedge normals (Whenever the thetas or reference frames change...)
    void m_updateMidedgeNormals();

    // Update geometric data cached for the deformed elements.
    void m_updateDeformedElements();

    // Update the second fundamental form (TODO: third fundamental form)
    void m_updateShapeOperators();

    // Method to update the tangent space basis for each triangle
    // (call after rest positions change, after element embeddings have been updated)
    void m_updateB();

    ////////////////////////////////////////////////////////////////////////////
    // Member variables
    ////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<Mesh> m_mesh;

    MX3d m_deformedPositions;
    VXd  m_thetas; // per-edge thetas

    // Map from the half edge index to our edge indices.
    std::vector<size_t> m_edgeForHalfEdge;

    // The reference frame with respect to which the midedge normals are expressed.
    // This frame is updated by parallel transport from the source configuration,
    std::vector<Frame> m_sourceReferenceFrame,
                       m_referenceFrame;
    // Angles between the reference director d1 and the triangle normal for each half-edge.
    // Note: we care about the boundary half-edges as well since we may wish to apply
    // clamp boundary conditions to the plate/shell.
    // The "source alpha" quantities are used to resolve the 2 * pi ambiguity
    // when updating the reference frame by enforcing temporal coherence
    // (preventing jumps in the measured gamma).
    VXd m_alphas, m_sourceAlphas;

    // Cached derived state quantities
    MX3d m_midedgeNormals;

    // Geometric information/shape functions for the deformed elements.
    std::vector<LinearlyEmbeddedElement<2, 1, V3d>> m_deformedElements;

    // Second fundamental form (shape operator pulled back to the reference
    // configuration). The discrete second fundamental form is a piecewise
    // constant matrix field.
    // Note: we use the same sign convention as [Grinspun2006], where the shape
    // operator computes the directional derivative of the normal (not its
    // negation). This is the opposite sign convention from most differential
    // geometry references, but actually the sign convention is irrelevant
    // for bending energy since only the square of the shape operator
    // enters into the elastic energy expression.
    std::vector<M3d> m_II, m_restII;

    // Energy density for each element (with support for multi-material microstructures).
    // For single-material microstructures, this vector will contain only a single entry.
    std::vector<Psi_C> m_psi;

    // Sheet thickness
    Real m_h = 1.0;

    // Orthonormal basis for each reference triangle's tangent space
    std::vector<M32d> m_B;
    std::vector<M32d> m_jacobianLambdaB;

    const size_t m_numVertices,
                 m_numEdges;

    bool m_disableBending = false;
};

#include "ElasticSheet.inl"

#endif /* end of include guard: ELASTICSHEET_HH */
