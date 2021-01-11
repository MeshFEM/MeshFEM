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
//  strains neatly decouple as a nice consequence of the linear
//  constitutive law. For other material models, the bending energy term should
//  technically plug the shape operator into a quadratic form defined by the
//  tangent elasticity tensor evaluated at the current membrane strain (for
//  St.VK, this is just the energy density quadratic form itself).
//  Unfortunately, that means the elastic energy gradient and Hessian would
//  involve third and fourth derivatives of psi.
//  Note that the derivation of the shell energy expression already assumes
//  small strain to drop certain terms, so our simplified implementation should
//  not be a significant additional source of error.
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
#include "EnergyDensities/EDensityAdaptors.hh"
#include "EnergyDensities/TangentElasticityTensor.hh"
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
// The sheet's material model is specified by template parameter "Psi_2x2",
// which can be an arbitrary 2x2 F-based or C-based *plane stress* energy
// density. The membrane energy term is just the integral of Psi_2x2
// over the sheet. The bending energy term is obtained by linearizing
// "Psi_2x2" around the identity to obtain a St. Venant Kirchhoff model into
// which the bending strain is inserted.
// Note, when Psi_2x2 is not St. Venant Kirchhoff, this uses an additional
// approximation/simplification compared to the standard nonlinear thin plate
// energy (which would technically require a Taylor expansion in the thickness
// direction).
template <class _Psi_2x2>
class ElasticSheet : public ElasticObject<typename _Psi_2x2::Real> {
public:
    using QuadratureRule = Quadrature<3, 1>; // Due to the bending strain discretization we use only linear FEM
    using EvalPtK = EvalPt<2>;

    using Psi_2x2 = _Psi_2x2;
    using Psi     = AutoHessianProjection<MembraneEnergyDensityFrom2x2Density<Psi_2x2>>;
    using Real    = typename Psi::Real;

    using V2d   = Eigen::Matrix<Real, 2, 1>;
    using V3d   = Eigen::Matrix<Real, 3, 1>;
    using M2d   = Eigen::Matrix<Real, 2, 2>;
    using M3d   = Eigen::Matrix<Real, 3, 3>;
    using M32d  = Eigen::Matrix<Real, 3, 2>;
    using VXd   = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using MX3d  = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>; // Row major so that flattened order agrees with VField
    using MX2d  = Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>;
    using Frame = M3d; // Columns are [tangent, d1, d2], a right-handed orthonormal frame adapted to a particular edge tangent.
    using SM2d  = SymmetricMatrixValue<Real, 2>; // Symmetric matrix in the reference configuration
    using CreaseEdges = Eigen::Matrix<int, Eigen::Dynamic, 2>;

    static constexpr size_t K   = 2;
    static constexpr size_t Deg = 1;
    static constexpr size_t N   = 3;
    static constexpr size_t numNodesPerElement  = Simplex::numNodes(K, Deg);
    // The local vars for an element are the corner positions and 3 midedge
    // normal rotation angles; for non-crease edges, this is just the edge's
    // theta, while for crease-edges this is (theta - creaseAngle / 2).
    static constexpr size_t numElementLocalVars = N * numNodesPerElement + 3;

    using  Mesh = FEMMesh<2, Deg, V3d>;
    using TMesh = typename Mesh::BaseMesh; // TriMesh data structure underlying FEMMesh
    using VSFJ = VectorizedShapeFunctionJacobian<3, V3d>;

    using  HEHandle = typename TMesh::template HEHandle<      TMesh>;
    using CHEHandle = typename TMesh::template HEHandle<const TMesh>;
    using  CTHandle = typename TMesh::template  THandle<const TMesh>;

    enum class EnergyType { Full, Membrane, Bending };
    enum class HessianProjectionType { Off, MembraneFBased, FullXBased };

    ElasticSheet(const std::shared_ptr<Mesh> &m, const Psi_2x2 &psi, const CreaseEdges &creases = CreaseEdges(0, 2))
        : m_mesh(m), m_psi{{psi}},
          m_etensor(tangentElasticityTensor(psi)),
          m_numVertices(m->numVertices()),
          m_numEdges   (m->numEdges()),
          m_numCreases(creases.rows())
    {
        m_updateB();

        // Build the halfedge -> edge map.
        m_edgeForHalfEdge.resize(m->numHalfEdges());
        m->visitEdges([this](CHEHandle he, size_t edgeIndex) {
            m_edgeForHalfEdge.at(he.index()) = edgeIndex;
            auto hopp = he.opposite();
            if (hopp) m_edgeForHalfEdge.at(hopp.index()) = edgeIndex;
        });

        {
            m_creaseEdgeIndexForEdge.assign(m_numEdges, -1);
            m_halfEdgeForCreaseAngle.reserve(m_numCreases);
            for (size_t i = 0; i < m_numCreases; ++i) {
                size_t a = creases(i, 0),
                       b = creases(i, 1);
                int hidx = std::max<int>(m->halfEdgeIndex(a, b),
                                         m->halfEdgeIndex(b, a));
                if (hidx < 0) throw std::runtime_error("Crease edge " + std::to_string(a) + ", " + std::to_string(b) + " not in mesh");
                if (m->halfEdge(hidx).isBoundary()) throw std::runtime_error("Crease edge " + std::to_string(a) + ", " + std::to_string(b) + " is on the boundary.");
                int &creaseIdx = m_creaseEdgeIndexForEdge[m_edgeForHalfEdge[hidx]];
                if (creaseIdx >= 0) throw std::runtime_error("Duplicate crease edge " + std::to_string(a) + ", " + std::to_string(b));
                m_creaseEdgeIndexForEdge[m_edgeForHalfEdge[hidx]] = m_halfEdgeForCreaseAngle.size();
                m_halfEdgeForCreaseAngle.push_back(hidx);
            }
            m_creaseAngles.setZero(m_numCreases);
            assert(m_halfEdgeForCreaseAngle.size() == m_numCreases);
        }

        setIdentityDeformation();

        // Apply this resulting shape operator as the rest shape operator
        // (To handle curved shells.)
        m_restII = m_II;

        setHessianProjectionType(HessianProjectionType::Off);
    }

    const Mesh &mesh() const { return *m_mesh; }
          Mesh &mesh()       { return *m_mesh; }

    // The variables consist of deformed vertex positions and midedge normal rotation angles.
    size_t numVars() const {
        return 3 * m_numVertices
                 + m_numEdges
                 + m_numCreases;
    }
    size_t numThetas()  const { return m_numEdges;   }
    size_t numCreases() const { return m_numCreases; }

    static constexpr size_t xOffset(){ return 0; }
    size_t       thetaOffset() const { return xOffset() + 3 * m_numVertices; }
    size_t creaseAngleOffset() const { return thetaOffset() + m_numEdges; }

    void setThickness(Real thickness) {
        m_h = thickness;
    }

    Real getThickness() const { return m_h; }
    size_t edgeForHalfEdge(size_t hei) const { return m_edgeForHalfEdge.at(hei); }
    int  creaseForHalfEdge(size_t hei) const { return m_creaseEdgeIndexForEdge[edgeForHalfEdge(hei)]; }

    virtual void setVars(Eigen::Ref<const VXd> vars) override {
        if (size_t(vars.rows()) != numVars()) throw std::runtime_error("Invalid vars size");
        m_thetas = vars.segment(thetaOffset(), m_numEdges);
        m_creaseAngles = vars.segment(creaseAngleOffset(), m_numCreases);
        setDeformedPositions(Eigen::Map<const MX3d>(vars.data(), m_numVertices, 3));
    }

    void setDeformedPositions(Eigen::Ref<const MX3d> x) {
        if (size_t(x.rows()) != m_numVertices) throw std::runtime_error("Invalid x size");
        m_deformedPositions = x;
        m_updateDeformedElements();
        m_adaptReferenceFrame(); // Side effect: update shape operators/midedge normals

        this->m_deformedConfigUpdated();
    }

    const VXd &getThetas()       const { return m_thetas;       }
    const VXd &getCreaseAngles() const { return m_creaseAngles; }

    void setThetas(Eigen::Ref<const VXd> thetas) {
        if (size_t(thetas.rows()) != m_numEdges) throw std::runtime_error("Invalid thetas size");
        m_thetas = thetas;

        m_updateShapeOperators();
        m_updateMidedgeNormals();

        this->m_deformedConfigUpdated();
    }

    void setCreaseAngles(Eigen::Ref<const VXd> creaseAngles) {
        if (size_t(creaseAngles.rows()) != m_numCreases) throw std::runtime_error("Invalid creaseAngles size");
        m_creaseAngles = creaseAngles;
        setThetas(m_thetas);
    }

    VXd getVars() const {
        VXd result(numVars());
        result.segment(          xOffset(), 3 * m_numVertices) = Eigen::Map<const VXd>(m_deformedPositions.data(), 3 * m_numVertices);
        result.segment(      thetaOffset(),        m_numEdges) = m_thetas;
        result.segment(creaseAngleOffset(),      m_numCreases) = m_creaseAngles;
        return result;
    }

    const MX3d &deformedPositions() const { return m_deformedPositions; }
    const VXd  &thetas()            const { return m_thetas;            }
    const VXd  &creaseAngles()      const { return m_creaseAngles;      }

    MX3d restPositions() const {
        const auto &m = mesh();
        MX3d rpos(m.numNodes(), 3);
        for (const auto n : m.nodes())
            rpos.row(n.index()) = n->p;
        return rpos;
    }

    MX3d nodeDisplacements() const { return deformedPositions() - restPositions(); }

    const Psi &getEnergyDensity(size_t ei) const {
        if (m_psi.size() == 1) return m_psi.front();
        return m_psi.at(ei);
    }

    Real elementEnergy(size_t ei, const EnergyType etype) const;
    Real energy(const EnergyType etype) const;

    // Gradient with respect to an individual element's corner positions and midedge normal angles.
    // (Note, we don't separately differentiate with respect to local crease angle vars;
    //  this dependence accounted for by chain rule in `gradient`)
    using ElementGradient = Eigen::Matrix<Real, numElementLocalVars, 1>;
    ElementGradient elementGradient(size_t, bool updatedSource, const EnergyType etype) const;

    VXd  gradient(bool updatedSource, const EnergyType etype = EnergyType::Full) const;

    // Hessian with respect to an individual element's corner positions and midedge normal angles.
    // (Note, we don't separately differentiate with respect to local crease angle vars;
    //  this dependence accounted for by chain rule in `hessian`)
    using PerElementHessian = Eigen::Matrix<Real, 12, 12>;
    PerElementHessian elementHessian(size_t ei, const EnergyType etype, bool projectionMask = false) const;

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
    // To assist boundary condition specification
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

    // (Re-)initialize the midedge normals (thetas), inferring them from the midsurface.
    // TODO: possibly make this infer crease angle as well?
    void initializeMidedgeNormals(bool minimizeBending = true);

    void updateSourceFrame() {
        m_sourceReferenceFrame = m_referenceFrame;
        m_sourceAlphas         = m_alphas;
    }

    // Update our parametrization of the system's DoFs
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
    const std::vector<M3d>  &getII()     const { return m_II;     }
    const std::vector<M3d>  &getRestII() const { return m_restII; }
    const std::vector<M32d> &getB()      const { return m_B;      }

    // Get the per-element right Cauchy-Green deformation tensors/first
    // fundamentals form representing the deformation.
    std::vector<M2d> getC() const {
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

    std::vector<M2d> getMembraneGreenStrains() const {
        auto result = getC();
        for (auto &r : result) {
            r = 0.5 * (r - M2d::Identity());
        }
        return result;
    }

    // Membrane green strains averaged onto the vertices.
    std::vector<M2d> vertexGreenStrains() const {
        return vertexAveragedField(mesh(), [this](size_t ei, const EvalPtK &) {
                auto e = mesh().element(ei);
                M32d FB = getCornerPositions(ei) * (e->gradBarycentric().transpose() * m_B[ei]);
                return (0.5 * (FB.transpose() * FB - M2d::Identity())).eval();
            });
    }

    const VXd &getAlphas()       const { return m_alphas;       }
    const VXd &getSourceAlphas() const { return m_sourceAlphas; }

    Real getGamma(size_t hei) const {
        Real result;
        // The current triangle's shape operator is defined in terms of the
        // angle gamma between the triangle normal and midedge normal
        // ***around the oriented edge vectors***. But thetas/alphas are
        // defined as angles around the primary halfedge vector (which may
        // point in the opposite direction). Therefore we must negate gamma
        // for non-primary half edges.
        double sign = mesh().halfEdge(hei).isPrimary() ? 1.0 : -1.0;
        result = sign * (m_thetas[m_edgeForHalfEdge[hei]] - m_alphas[hei]);

        int ci = creaseForHalfEdge(hei);
        if (ci >= 0) {
            // Positive crease angles rotate the midedge normal towards the
            // triangle (decreasing gamma)
            result -= 0.5 * m_creaseAngles[ci];
        }
        return result;
    }

    VXd getGammas() const {
        const auto &m = mesh();
        const size_t nhe = m.numHalfEdges();
        VXd gammas(nhe);
        for (size_t hei = 0; hei < nhe; ++hei)
            gammas[hei] = getGamma(hei);
        return gammas;
    }

    // Get the principal curvatures of the deformed sheet geometry.
    MX2d getPrincipalCurvatures() const;

    // The volume associated with a shell element is area * thickness.
    VXd element3DVolumes() const {
        const auto &m = mesh();
        VXd result(m.numElements());
        for (const auto e : m.elements())
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

    void filterRMPinArtifacts(const typename RMPins::PinVertices &/* pinVertices */) {
        throw std::runtime_error("Unimplemented");
        // ::filterRMPinArtifacts(*this, pinVertices);
    }

    void setDisabledBending(bool yesno) { m_disableBending = yesno; }
    bool getDisabledBending() const { return m_disableBending; }

    void setHessianProjectionType(HessianProjectionType hp) {
        m_hessianProjectionType = hp;
        bool projectPsi = (m_hessianProjectionType == HessianProjectionType::MembraneFBased);
        for (auto &psi : m_psi)
            psi.projectionEnabled = projectPsi;
    }

    HessianProjectionType getHessianProjectionType() const {
        return m_hessianProjectionType;
    }

    virtual std::unique_ptr<FieldSampler> referenceConfigSampler() const override {
        return FieldSampler::construct(std::shared_ptr<const Mesh>(m_mesh)); // work around template parameter deduction issue
    }

    virtual SuiteSparseMatrix deformationSamplerMatrix(Eigen::Ref<const Eigen::MatrixXd> P) const override {
        return fieldSamplerMatrix(mesh(), N, P, 0, numVars() - 3 * m_numVertices /* nodal value vector is padded by midedge normal variables */);
    }

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
    VXd  m_creaseAngles; // per-crease-edge angles

    // Map from the half edge index to our edge indices.
    std::vector<size_t> m_edgeForHalfEdge;
    std::vector<int>    m_creaseEdgeIndexForEdge; // -1 for non-crease edges
    std::vector<size_t> m_halfEdgeForCreaseAngle; // Arbitrary half-edge of the edge associated with each crease angle var

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
    std::vector<Psi> m_psi;
    ElasticityTensor<Real, 2> m_etensor;

    // Sheet thickness
    Real m_h = 1.0;

    // Orthonormal basis for each reference triangle's tangent space
    std::vector<M32d> m_B;
    std::vector<M32d> m_jacobianLambdaB;

    const size_t m_numVertices,
                 m_numEdges,
                 m_numCreases;

    bool m_disableBending = false;

    HessianProjectionType m_hessianProjectionType = HessianProjectionType::Off;
};

#include "ElasticSheet.inl"

#endif /* end of include guard: ELASTICSHEET_HH */
