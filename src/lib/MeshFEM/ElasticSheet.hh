////////////////////////////////////////////////////////////////////////////////
// ElasticSheet.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Simulate a thin elastic sheet with a potentially curved rest configuration
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
#include "newton_optimizer/newton_optimizer.hh"
#include "Geometry.hh"

#include "RigidMotionPins.hh"
#include "ElasticObject.hh"
#include "FieldPostProcessing.hh"
#include "Elements/PlateBending.hh"
#include "Elements/MembraneElement.hh"

#include "SystemAssembler.hh"

// Note: anisotropic materials are supported and, for plates (sheets with
// perfectly flat rest states), the anisotropic energy density function can be
// intuitively expressed in the global 2D coordinate system; this is
// enabled by the special case in EmbeddedMembraneElementData::embeddingUpdated
// that uses the global x and y axes.
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
struct ElasticSheet : public ElasticObject<typename _Psi_2x2::Real> {
    using Assembler = SystemAssembler<3, 1, 1>; // Variables: (Vertex positions, midedge normals, crease angles)
    using QuadratureRule = Quadrature<3, 1>; // Due to the bending strain discretization we use only linear FEM
    using EvalPtK = EvalPt<2>;

    using Psi_2x2 = _Psi_2x2;
    using Real    = typename Psi_2x2::Real;
    using ETensor = ElasticityTensor<Real, 2>;

    using ME = MembraneElement</* Deg = */ 1, Psi_2x2>;
    using CornerPositions = typename ME::LocalVars; // One position per row; row-major
    using PBE    = PlateBending<Real>;
    using PBEMat = typename PBE::Material;
    using MEMat  = typename  ME::Material;

    using Base = ElasticObject<Real>;
    using CSCMat  = typename Base::CSCMat;
    using Base::numVars;

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

    using  HEHandle = typename TMesh::template HEHandle<      TMesh>;
    using CHEHandle = typename TMesh::template HEHandle<const TMesh>;
    using  CTHandle = typename TMesh::template  THandle<const TMesh>;

    enum class EnergyType { Full, Membrane, Bending };
    enum class HessianProjectionType { Off, MembraneFBased, FullXBased };

    // Enumeration of edges and crease edges (used to allocate variables).
    struct EdgeVariableStructure {
        EdgeVariableStructure(const std::shared_ptr<Mesh> &m, const CreaseEdges &creases)
            : numCreases(creases.rows())
        {
            numEdges = 0;
            // Build the halfedge -> edge map.
            edgeForHalfEdge.resize(m->numHalfEdges());
            m->visitEdges([this](CHEHandle he, size_t edgeIndex) {
                ++numEdges;
                edgeForHalfEdge.at(he.index()) = edgeIndex;
                halfedgeForEdge.push_back(he.index());
                auto hopp = he.opposite();
                if (hopp) edgeForHalfEdge.at(hopp.index()) = edgeIndex;
            });

            // Allocate crease variables.
            creaseEdgeIndexForEdge.assign(numEdges, -1);
            halfEdgeForCreaseAngle.reserve(numCreases);
            for (size_t i = 0; i < numCreases; ++i) {
                size_t a = creases(i, 0),
                       b = creases(i, 1);
                int hidx = std::max<int>(m->halfEdgeIndex(a, b),
                                         m->halfEdgeIndex(b, a));
                if (hidx < 0) throw std::runtime_error("Crease edge " + std::to_string(a) + ", " + std::to_string(b) + " not in mesh");
                auto he = m->halfEdge(hidx).primary();
                hidx = he.index();

                if (he.isBoundary()) throw std::runtime_error("Crease edge " + std::to_string(a) + ", " + std::to_string(b) + " is on the boundary.");
                int &creaseIdx = creaseEdgeIndexForEdge[edgeForHalfEdge[hidx]];
                if (creaseIdx >= 0) throw std::runtime_error("Duplicate crease edge " + std::to_string(a) + ", " + std::to_string(b));
                creaseIdx = i;
                halfEdgeForCreaseAngle.push_back(hidx);
            }
        }

        size_t numEdges, numCreases;
        // Map from the half edge index to our edge indices.
        std::vector<size_t> edgeForHalfEdge, halfedgeForEdge;
        std::vector<int>    creaseEdgeIndexForEdge; // -1 for non-crease edges
        std::vector<size_t> halfEdgeForCreaseAngle; // Arbitrary half-edge of the edge associated with each crease angle var
    };

    ElasticSheet(const std::shared_ptr<Mesh> &mptr, const Psi_2x2 &psi, const CreaseEdges &creases = CreaseEdges(0, 2))
        : m_mesh(mptr),
          m_edgeVarStructure(mptr, creases),
          m_numVertices(mptr->numVertices()),
          m_membraneMaterials(mptr->numElements()),
          m_plateMaterials(mptr->numElements()),
          m_assembler(m_numVertices, m_edgeVarStructure.numEdges, m_edgeVarStructure.numCreases)
    {
        const auto &m = mesh();

        m_membraneMaterials[0].psi = psi;
        m_membraneMaterials[0].thickness = 1.0;
        m_plateMaterials   [0].setPsi(psi);
        m_plateMaterials   [0].setThickness(1.0);

        const size_t ne = m.numElements();
        m_membraneElements.reserve(ne);
        for (size_t ei = 0; ei < ne; ++ei)
            m_membraneElements.emplace_back(ei, m, m_membraneMaterials);

        // Construct and initialize plate elements (assuming gamma = 0)
        m_plateElements.reserve(ne);
        for (size_t ei = 0; ei < ne; ++ei)
            m_plateElements.emplace_back(ei, m_membraneElements[ei].elementData, m_plateMaterials);

        setIdentityDeformation();

        // Apply this resulting shape operator as the rest shape operator
        // (To handle curved shells.)
        programRestCurvature();

        setHessianProjectionType(HessianProjectionType::Off);
    }

    typename PBE::LocalVars extractPlateVars(size_t ei) const {
        const auto &e = m_mesh->element(ei);
        typename PBE::LocalVars x;
        for (auto v : e.vertices())
            x.template segment<3>(3 * v.localIndex()) = m_deformedPositions.row(v.index());
        x.template segment<3>(3 * numNodesPerElement) = getTriGammas(ei);
        return x;
    }

    const Mesh &mesh() const { return *m_mesh; }
          Mesh &mesh()       { return *m_mesh; }

    size_t numDefoVars() const override { return varStructure().numVars(); }
    size_t numRestVars() const override { return 3 * numVertices(); }

    size_t numVertices()  const { return m_numVertices;   }
    size_t numEdges()     const { return m_edgeVarStructure.numEdges;   }
    size_t numThetas()    const { return numEdges();   }
    size_t numCreases()   const { return m_edgeVarStructure.numCreases; }

    const auto &varStructure() const { return m_assembler.varStructure(); }

    size_t           xOffset() const { return varStructure().offsetForType(0); }
    size_t       thetaOffset() const { return varStructure().offsetForType(1); }
    size_t creaseAngleOffset() const { return varStructure().offsetForType(2); }

    template<class VarVector> auto sliceDeformedPositions(      VarVector &vars) const { return Eigen::Map<      MX3d>(varStructure().variablesOfType(vars, 0).data(), numVertices(), 3); }
    template<class VarVector> auto sliceDeformedPositions(const VarVector &vars) const { return Eigen::Map<const MX3d>(varStructure().variablesOfType(vars, 0).data(), numVertices(), 3); }
    template<class VarVector> auto sliceThetas      (VarVector &vars) const { return varStructure().variablesOfType(vars, 1); }
    template<class VarVector> auto sliceCreaseAngles(VarVector &vars) const { return varStructure().variablesOfType(vars, 2); }

    size_t   edgeForHalfEdge(size_t hei)    const { return m_edgeVarStructure.edgeForHalfEdge[hei]; }
    int    creaseEdgeIndexForEdge(size_t e) const { return m_edgeVarStructure.creaseEdgeIndexForEdge[e]; }
    size_t        halfEdgeForEdge(size_t e) const { return m_edgeVarStructure.halfedgeForEdge[e]; }
    size_t halfEdgeForCreaseAngle(size_t c) const { return m_edgeVarStructure.halfEdgeForCreaseAngle[c]; }
    int  creaseForHalfEdge(size_t hei) const { return creaseEdgeIndexForEdge(edgeForHalfEdge(hei)); }

    void setDeformedPositions(Eigen::Ref<const MX3d> x) {
        if (size_t(x.rows()) != numVertices()) throw std::runtime_error("Invalid vertex position size");
        m_deformedPositions = x;
        m_defoConfigUpdated(/* positionsUpdated = */ true);
    }

    const VXd &getThetas()       const { return m_thetas;       }
    const VXd &getCreaseAngles() const { return m_creaseAngles; }

    void setThetas(Eigen::Ref<const VXd> thetas) {
        if (size_t(thetas.rows()) != numThetas()) throw std::runtime_error("Invalid thetas size");
        m_thetas = thetas;
        m_defoConfigUpdated(/* positionsUpdated = */ false);
    }

    void setCreaseAngles(Eigen::Ref<const VXd> creaseAngles) {
        if (size_t(creaseAngles.rows()) != numCreases()) throw std::runtime_error("Invalid creaseAngles size");
        m_creaseAngles = creaseAngles;
        setThetas(m_thetas);
    }

    VXd getDefoVars() const override {
        VXd result(numDefoVars());
        sliceDeformedPositions(result) = m_deformedPositions;
        sliceThetas(result) = m_thetas;
        sliceCreaseAngles(result) = m_creaseAngles;
        return result;
    }

    VXd getRestVars() const override { return Eigen::Map<const VXd>(m_deformedPositions.data(), numRestVars()); }

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

    Real elementEnergy(size_t ei, const EnergyType etype) const;
    Real energy(const EnergyType etype) const;

    // Gradient with respect to an individual element's corner positions and midedge normal angles.
    // (Note, we don't separately differentiate with respect to local crease angle vars;
    //  this dependence accounted for by chain rule in `gradient`)
    using ElementGradient = Eigen::Matrix<Real, numElementLocalVars, 1>;
    ElementGradient elementGradient(size_t, bool updatedSource, const EnergyType etype) const;

    void accumulateGradient(Real weight, VXd &g, bool updatedSource, VariableMask vmask, const EnergyType etype = EnergyType::Full) const;

    // Hessian with respect to an individual element's corner positions and midedge normal angles.
    // (Note, we don't separately differentiate with respect to local crease angle vars;
    //  this dependence accounted for by chain rule in `hessian`)
    using PerElementHessian = Eigen::Matrix<Real, 12, 12>;
    PerElementHessian elementHessian(size_t ei, const EnergyType etype, bool projectionMask = false) const;

    using EBlockVars = VecMaxN_T<SuiteSparse_long, 9>; // Up to 9 (block) vars influence each element
    auto elementGetter() const {
        const auto &m = mesh();
        return [this, &m](size_t ei) {
            EBlockVars blockVars(9);
            auto e = m.element(ei);
            for (auto v : e.vertices())
                blockVars[v.localIndex()] = v.index();
            size_t crease_back = 6;
            for (auto he : e.halfEdges()) {
                blockVars[3 + he.localIndex()] = m.numVertices() + edgeForHalfEdge(he.index());
                int ci = creaseForHalfEdge(he.index());
                if (ci < 0) continue;
                blockVars[crease_back++] = numVertices() + numEdges() + ci;
            }
            blockVars.conservativeResize(crease_back);
            return blockVars;
        };
    }

    void accumulateHessian(Real weight, CSCMat &Hout, const EnergyType etype, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const;
    virtual CSCMat hessianSparsityPattern(Real val = 0.0, VariableMask vmask = VariableMask::Defo) const override;

    // Convenience methods
    VXd gradient(bool updatedSource, VariableMask vmask = VariableMask::Defo, const EnergyType etype = EnergyType::Full) const {
        VXd g = VXd::Zero(numVars());
        accumulateGradient(1.0, g, updatedSource, vmask, etype);
        return g;
    }

    SuiteSparseMatrix hessian(bool projectionMask = false, VariableMask vmask = VariableMask::Defo, const EnergyType etype = EnergyType::Full) const {
        SuiteSparseMatrix H(hessianSparsityPattern());
        accumulateHessian(1.0, H, etype, projectionMask, vmask);
        return H;
    }

    void accumulateHessianNew(Real weight, CSCMat &Hout, const EnergyType etype, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const;
    SuiteSparseMatrix hessianNew(bool projectionMask = false, VariableMask vmask = VariableMask::Defo, const EnergyType etype = EnergyType::Full) const {
        SuiteSparseMatrix H(hessianSparsityPattern());
        accumulateHessianNew(1.0, H, etype, projectionMask, vmask);
        return H;
    }

    // Overloads implementing generic ElasticObject interface.
    virtual Real  energy() const override { return energy(EnergyType::Full); }
    virtual void accumulateGradient(Real weight, VXd &g, bool updatedParametrization = false, VariableMask vmask = VariableMask::Defo) const override {
        return accumulateGradient(weight, g, updatedParametrization, vmask, EnergyType::Full);
    }
    virtual void accumulateHessian(Real weight, CSCMat &Hout, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const override {
        if (weight != 1.0) throw std::runtime_error("ElasticSheet::accumulateGradient does not support weight != 1.0");
        accumulateHessian(weight, Hout, EnergyType::Full, projectionMask, vmask);
    }

    const std::vector<Frame> &midedgeReferenceFrames() const { return m_referenceFrame; }
    const std::vector<Frame> & sourceReferenceFrames() const { return m_sourceReferenceFrame; }

    // For debugging visualizations of the edge frames, we need their application points
    MX3d midedgeNormals() const {
        MX3d result(numEdges(), 3);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numEdges()),
                          [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                result.row(i) = std::cos(m_thetas[i]) * m_referenceFrame[i].col(1) +
                                std::sin(m_thetas[i]) * m_referenceFrame[i].col(2);
            }
        });
        return result;
    }

    MX3d edgeMidpoints() const {
        MX3d result(numEdges(), 3);
        mesh().visitEdges([this, &result](CHEHandle he, size_t edgeIndex) {
            result.row(edgeIndex) = 0.5 * (m_deformedPositions.row(he.tip().index())
                                         + m_deformedPositions.row(he.tail().index()));
        });
        return result;
    }
    // To assist boundary condition specification
    MX3d restEdgeMidpoints() const {
        MX3d result(numEdges(), 3);
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
    void setIdentityDeformation() override;

    // (Re-)initialize the midedge normals (thetas), inferring them from the midsurface.
    void initializeMidedgeNormals(bool inferCreaseAngles = true, bool minimizeBending = true);

    void updateSourceFrame() {
        m_sourceReferenceFrame = m_referenceFrame;
        m_sourceAlphas         = m_alphas;
    }

    // Update our parametrization of the system's DoFs
    // (currently this just means updating the source frames.)
    void updateParametrization() override { updateSourceFrame(); }

    template<class HEType>
    auto deformedEdgeVector(const HEType &he) const {
        return (m_deformedPositions.row(he. tip().index())
              - m_deformedPositions.row(he.tail().index())).eval();
    }
    const auto &deformedTriGeometry(size_t ei) const { return m_plateElements[ei].de; }
    Real     deformedArea(size_t ei) const { return deformedTriGeometry(ei).A; }
    V3d deformedTriNormal(size_t ei) const { return deformedTriGeometry(ei).normal; }

    // Get the deformed positions of triangle ti's corners as rows
    // of a 3x3 matrix.
    CornerPositions getCornerPositions(size_t ti) const {
        auto t = mesh().tri(ti);
        CornerPositions result;
        result << m_deformedPositions.row(t.vertex(0).index()),
                  m_deformedPositions.row(t.vertex(1).index()),
                  m_deformedPositions.row(t.vertex(2).index());
        return result;
    }

    // Get the normal rotation angles at triangle ti's halfedge midpoints.
    V3d getTriGammas(size_t ti) const {
        auto t = mesh().tri(ti);
        return V3d(getGamma(t.halfEdge(0).index()),
                   getGamma(t.halfEdge(1).index()),
                   getGamma(t.halfEdge(2).index()));
    }

    const M32d &getB (size_t ei) const { return m_membraneElements[ei].elementData.B(); }
    M32d getFB(size_t ei)        const { return m_membraneElements[ei].getFB(getCornerPositions(ei)); }

    SM2d getBendingStrain(size_t ei) const { return m_plateElements[ei].bendingStrain(); }

    // Get the deformed/rest second fundamental forms (expressed in the
    // reference triangle's orthonormal frame).
    M2d     getII(size_t ei) const { return m_plateElements[ei].    II; }
    M2d getRestII(size_t ei) const { return m_plateElements[ei].restII; }

    // Deformed second fundamental forms expressed in the global frame.
    M3d getII_3D(size_t ei) const {
        M32d B = getB(ei);
        return B * getII(ei) * B.transpose();
    }

    // Set the rest state to be flat.
    void programFlatRestCurvature() {
        for (auto &pe : m_plateElements) pe.programFlatRestCurvature();
        this->m_restConfigUpdated();
    }

    // Bake the current deformed state's curvature into the rest curvature (plastically deforming)
    void programRestCurvature() {
        for (auto &pe : m_plateElements) pe.programRestCurvature();
        this->m_restConfigUpdated();
    }

    // Get the per-element right Cauchy-Green deformation tensors/first
    // fundamentals form representing the deformation.
    M2d getC(size_t ei) const {
        M32d FB = getFB(ei);
        return FB.transpose() * FB;
    }

    // Note: for nonzero Poisson's ratio, there will be strain along the
    // thickness direction that is omitted here.
    M2d getMembraneGreenStrain(size_t ei) const {
        M32d FB = getFB(ei);
        return 0.5 * (FB.transpose() * FB - M2d::Identity());
    }

    // Membrane green strains averaged onto the vertices.
    // Note: for nonzero Poisson's ratio, there will be strain along the
    // thickness direction that is omitted here.
    std::vector<M2d> vertexGreenStrains() const {
        return vertexAveragedField(mesh(), [this](size_t ei, const EvalPtK &) {
                return getMembraneGreenStrain(ei);
            });
    }

    // Evaluate approximate volumetric Green strain (combining stretching and
    // bending) for element `ei` at the thickness coordinate `z`.
    // Note: for nonzero Poisson's ratio, there will be strain along the
    // thickness direction that is omitted here.
    M2d getElementVolumetricStrain(size_t ei, Real z) const {
        M32d FB = getFB(ei);
        return getMembraneGreenStrain(ei) + z * (getII(ei) - getRestII(ei));
    }

    // Evaluate approximate volumetric stress for element `ei` at the thickness
    // coordinate `z`.
    M2d getElementVolumetricPlaneStress(size_t ei, Real z) const {
        M2d strain = getElementVolumetricStrain(ei, z);
        return m_plateMaterials[ei].C.doubleContract(SM2d(strain)).matrix();
    }

    // Sample the implied PK2 stress field for element `ei` at thickness coordinate `z`.
    // (This is really analogous to a PK2 stress, since to derive the bending energy term
    //  we use a St. Venant Kirchhoff model wherein the Green strain is plugged into
    //  elementETensor(ei)'s quadratic form.)
    M3d getElementVolumetricPK2Stress(size_t ei, Real z) const {
        M2d plane_stress = getElementVolumetricPlaneStress(ei, z);
        const auto &B = getB(ei);
        return B * plane_stress * B.transpose();
    }

    M3d getElementCauchyStress(size_t ei, Real z) const {
        M2d plane_PK2_stress = getElementVolumetricPlaneStress(ei, z); // PK2 stress in tri-local coordinate system
        M32d FB = getFB(ei);
        Real J = std::sqrt((FB.transpose() * FB).determinant());
        return (FB * plane_PK2_stress * FB.transpose()) / J;
    }

    // Evaluate approximate volumetric strain (combining stretching and
    // bending) at the thickness coordinate `z`, averaged onto the vertices.
    std::vector<M2d> getVertexVolumetricStrains(Real z) const {
        return vertexAveragedField(mesh(), [this, z](size_t ei, const EvalPtK &) {
                return getElementVolumetricStrain(ei, z);
            });
    }

    // Evaluate approximate volumetric strain (combining stretching and
    // bending) at the thickness coordinate `z`, averaged onto the vertices.
    std::vector<M3d> getVertexCauchyStresses(Real z) const {
        return vertexAveragedField(mesh(), [this, z](size_t ei, const EvalPtK &) {
                return getElementCauchyStress(ei, z);
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
        result = sign * (m_thetas[edgeForHalfEdge(hei)] - m_alphas[hei]);

        int ci = creaseForHalfEdge(hei);
        if (ci >= 0) {
            // Positive crease angles rotate the midedge normal towards the
            // triangle (decreasing gamma)
            result -= 0.5 * m_creaseAngles[ci];
        }
        return result;
    }

    VXd getGammas() const {
        const size_t nhe = mesh().numHalfEdges();
        VXd result(nhe);
        for (size_t hei = 0; hei < nhe; ++hei)
            result[hei] = getGamma(hei);
        return result;
    }

    template<class Result>
    void accumulateGradGamma(Real weight, size_t ei, size_t lhi, bool updatedSource, Result &&result) const;

    // Get the principal curvatures of the deformed sheet geometry.
    MX2d getPrincipalCurvatures() const;

    // The volume associated with a shell element is area * thickness.
    VXd element3DVolumes() const {
        const auto &m = mesh();
        VXd result(m.numElements());
        for (const auto e : m.elements())
            result[e.index()] = e->volume() * m_plateMaterials[e.index()].getThickness();
        return result;
    }

    void setThickness(Real t) {
        m_membraneMaterials.foreach([t](MEMat  &mat) { mat.thickness = t; });
        m_plateMaterials   .foreach([t](PBEMat &mat) { mat.setThickness(t); });
    }

    Real getThickness() const {
        Real result = m_plateMaterials[0].getThickness();
        m_plateMaterials.foreach([&result](const PBEMat &mat) {
            if (mat.getThickness() != result)
                throw std::runtime_error("Inconsistent thicknesses");
        });
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
        for (size_t i = 0; i < numEdges(); ++i)
            m_sourceReferenceFrame[i] = (R * m_sourceReferenceFrame[i]).eval();

        auto prerotationFrames = m_referenceFrame; // for validation
        setDeformedPositions((m_deformedPositions * R.transpose()).rowwise() + t.transpose());

        for (size_t i = 0; i < numEdges(); ++i) {
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
        m_membraneMaterials.foreach([projectPsi](MEMat &mat) { mat.psi.projectionEnabled = projectPsi; });
    }

    void setMaterials(const std::vector<Psi_2x2> &psis, const std::vector<size_t> &materialForElement = {}) {
        std::vector< MEMat>  membraneMaterials(psis.size());
        std::vector<PBEMat>     plateMaterials(psis.size());

        Real h = getThickness();

        for (size_t mi = 0; mi < psis.size(); ++mi) {
            membraneMaterials[mi].psi = psis[mi];
            membraneMaterials[mi].thickness = h;
            plateMaterials   [mi].setPsi(psis[mi]);
            plateMaterials   [mi].setThickness(h);
        }

        m_membraneMaterials.setSpatiallyVarying(membraneMaterials, materialForElement);
        m_plateMaterials   .setSpatiallyVarying(   plateMaterials, materialForElement);
    }

    const std::vector<size_t> &materialForElement() const { return m_membraneMaterials.materialForElement(); }

    HessianProjectionType getHessianProjectionType() const {
        return m_hessianProjectionType;
    }

    virtual std::unique_ptr<FieldSampler> referenceConfigSampler() const override {
        return FieldSampler::construct(std::shared_ptr<const Mesh>(m_mesh)); // work around template parameter deduction issue
    }

    virtual CSCMat deformationSamplerMatrix(Eigen::Ref<const Eigen::MatrixXd> P) const override {
        return fieldSamplerMatrix(mesh(), N, P, 0, numDefoVars() - 3 * m_numVertices /* nodal value vector is padded by midedge normal variables */);
    }

private:
    void m_setDefoVars(const Eigen::Ref<const VXd> &vars) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSheet.m_setDefoVars");
        if (size_t(vars.rows()) != numDefoVars()) throw std::runtime_error("Invalid vars size");

        m_thetas            = sliceThetas(vars);
        m_creaseAngles      = sliceCreaseAngles(vars);
        m_deformedPositions = sliceDeformedPositions(vars);

        m_defoConfigUpdated();
    }

    void m_setRestVars(const Eigen::Ref<const VXd> & /* vars */) override {
        throw std::runtime_error("Unimplemented");
        for (auto &me : m_membraneElements)
            me.elementData.embeddingUpdated();
    }

    // Update the current midedge reference frame to adapt to the new deformed
    // edge tagents.
    void m_adaptReferenceFrame();

    void m_updateElementEmbedding();

    // Update geometric data cached for the deformed elements.
    void m_updateDeformedElements(bool positionsUpdated = true);

    // Update all cached deformation-dependent quantities (when either positions
    // or midedge normal angles change).
    void m_defoConfigUpdated(bool positionsUpdated = true) {
        m_updateDeformedElements(positionsUpdated);
        Base::m_defoConfigUpdated(); // Call base implementation to dispatch notifications
    }

    ////////////////////////////////////////////////////////////////////////////
    // Member variables
    ////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<Mesh> m_mesh;
    EdgeVariableStructure m_edgeVarStructure; // must appear before m_assembler for proper initialization!

    MX3d m_deformedPositions;
    VXd  m_thetas; // per-edge thetas
    VXd  m_creaseAngles; // per-crease-edge angles

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

    MaterialAssignment<typename  ME::Material> m_membraneMaterials;
    MaterialAssignment<typename PBE::Material> m_plateMaterials;

    std::vector< ME> m_membraneElements;
    std::vector<PBE> m_plateElements;

    const size_t m_numVertices;

    bool m_disableBending = false;

    HessianProjectionType m_hessianProjectionType = HessianProjectionType::Off;

    std::unique_ptr<NewtonOptimizer> m_normalInferenceOptimizer;

    Assembler m_assembler;
};

#include "ElasticSheet.inl"

#endif /* end of include guard: ELASTICSHEET_HH */
