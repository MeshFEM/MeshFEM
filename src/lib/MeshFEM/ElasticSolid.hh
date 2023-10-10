////////////////////////////////////////////////////////////////////////////////
// ElasticSolid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Represents a hyperlastic elastic solid made of triangles/tets.
*///////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICSOLID_HH
#define ELASTICSOLID_HH

#include "FEMMesh.hh"
#include "GaussQuadrature.hh"
#include "GlobalBenchmark.hh"
#include "MeshIO.hh"
#include "ParallelAssembly.hh"
#include "SparseMatrices.hh"
#include "SystemAssembler.hh"
#include "Flattening.hh"
#include "Types.hh"
#include "Functions.hh"
#include "EnergyDensities/Tensor.hh"
#include "EnergyDensities/EnergyTraits.hh"
#include "FieldSamplerMatrix.hh"
#include <Eigen/Sparse>

#include <atomic>
#include <optional>

#include "RigidMotionPins.hh"
#include "FieldPostProcessing.hh"
#include "InterpolantRestriction.hh"

#include "ElasticObject.hh"
#include "MassMatrix.hh"
#include "Laplacian.hh"
#include "VonMises.hh"

#include "Elements/HyperelasticLagrange.hh"

// _K: simplex dimension (2 ==> tri/3 ==> tet)
// _Deg: finite element degree (1 or 2)
// EmbeddingSpace: ND point type; Note N may differ from K (for a triangle mesh embedded in 3D, e.g.)
template<size_t _K, size_t _Deg, class _EmbeddingSpace, class _Energy>
struct ElasticSolid : public ElasticObject<typename _EmbeddingSpace::Scalar> {
    using EmbeddingSpace = _EmbeddingSpace;
    using Real   = typename EmbeddingSpace::Scalar;
    using Energy = _Energy;
    static_assert(std::is_convertible<typename Energy::Real, Real>::value, "Incompatible real number types");

    using Base = ElasticObject<Real>;
    using CSCMat = typename Base::CSCMat;
    using Base::numVars;
    using VariableMask = typename Base::VariableMask;

    static constexpr size_t K = _K;
    static constexpr size_t N = EmbeddingSpace::RowsAtCompileTime;
    static constexpr size_t Deg = _Deg;
    static constexpr size_t numNodesPerElement  = Simplex::numNodes(N, Deg);
    static constexpr size_t numElementLocalVars = N * numNodesPerElement;

    using SE = elements::Solid<Energy, K, Deg>;
    using NodePositions = typename SE::NodePositions;

    using EvalPtK  = EvalPt<K>;
    using VNd      = Eigen::Matrix<Real, N, 1>;
    using MNd      = Eigen::Matrix<Real, N, N>;
    using VXd      = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using MXNd     = Eigen::Matrix<Real, Eigen::Dynamic, N, Eigen::RowMajor>; // Row major so that flattened order agrees with VField
    using Mesh     = FEMMesh<K, Deg, VNd>;
    using GradPhis = typename Mesh::ElementData::GradPhis;

    ElasticSolid(const Energy &energy, const std::shared_ptr<Mesh> &mesh)
        : m_mesh(mesh), m_energyDensities{{energy}}, m_assembler(mesh->numNodes()) { setIdentityDeformation(); }

    // Copy and degree-changing constructor
    template<size_t Deg2>
    ElasticSolid(const ElasticSolid<K, Deg2, EmbeddingSpace, Energy> &es)
        : m_assembler(es.mesh().numNodes()) { m_copy(es); }
    // Note: the degree-changing constructor template is excluded from overload resolution,
    // and the implicitly copy constructor is deleted due to the `m_assembler` member...
    ElasticSolid(const ElasticSolid &es)
        : m_assembler(es.mesh().numNodes()) { m_copy(es); }

    size_t numElements() const { return mesh().numElements(); }
    size_t numVertices() const { return mesh().numVertices(); }
    size_t numNodes   () const { return mesh().numNodes(); }

    size_t numDefoVars() const override { return m_x.size(); }
    size_t numRestVars() const override { return numVertices() * N; }

    void setIdentityDeformation() override {
        m_x.resize(numNodes(), size_t(N));
        for (const auto n : mesh().nodes())
            m_x.row(n.index()) = n->p;
        this->m_defoConfigUpdated();
    }

    VXd getDefoVars() const override { return Eigen::Map<const VXd>(m_x.data(), m_x.size()); }

    VXd getRestVars() const override {
        VXd rest_state(numRestVars());
        for (const auto v : mesh().vertices())
            rest_state.template segment<N>(N * v.index()) = v.node()->p;
        return rest_state;
    }

    void setDeformedPositions(Eigen::Ref<const MXNd> vertexPositions) {
        Base::setDefoVars(Eigen::Map<const VXd>(vertexPositions.data(), vertexPositions.size()));
    }

    // Energy stored in the full object.
    virtual Real energy() const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolid.energy");
        return summation_parallel([this](size_t ei) { return elementEnergy(ei); }, mesh().numElements());
    }

    // Energy stored in a single element.
    Real elementEnergy(size_t ei) const {
        return SE::energy(getEnergyDensity(ei), extractNodePositions(ei, m_x), mesh().elementData(ei));
    }

    // Gradient of a single element's energy with respect to its nodes' deformed positions..
    using ElementGradient = Eigen::Matrix<Real, numElementLocalVars, 1>;
    ElementGradient elementGradient(size_t ei) const {
        return SE::gradient(getEnergyDensity(ei), extractNodePositions(ei, m_x), mesh().elementData(ei));
    }

    // Gradient of the full object's energy with respect to all deformation variables.
    virtual VXd gradient(bool /* updatedParametrization */ = false, VariableMask vmask = VariableMask::Defo) const override {
        if (vmask != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");

        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolid.gradient");
#if 0
        VXd g(numVars());
        m_assembler.assembleGradientScatterGather(g, mesh(), [this](size_t ei) { return elementGradient(ei); } );
#else
        VXd g = VXd::Zero(numVars());
        m_assembler.assembleGradient(g, mesh(), [this](size_t ei) { return elementGradient(ei); } );
#endif
        return g;
    }

    CSCMat hessian(bool projectionMask = false) const {
        CSCMat H(hessianSparsityPattern());
        hessian(H, projectionMask);
        return H;
    }

    using PerElementBlockHessian = Eigen::Matrix<Real, numElementLocalVars, numElementLocalVars>;
    using PerElementHessian = Eigen::Matrix<Real, numElementLocalVars, numElementLocalVars>;
    PerElementHessian elementHessian(size_t ei, bool disableProjection = false) const {
        return SE::hessian(getEnergyDensity(ei), extractNodePositions(ei, m_x), mesh().elementData(ei), disableProjection);
    }

    CSCMat &getBlockHsp() const {
        if (!m_blockHsp) m_blockHsp = m_assembler.blockSparsityPatternForMesh(mesh());
        return m_blockHsp.value();
    }

    // Construct a scalar-valued Hessian.
    virtual void hessian(CSCMat &H, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const override {
        if (vmask != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");

        // BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolid.hessian");
        // m_assembler.assembleHessian(H, mesh(), [this, projectionMask](size_t ei) {
        //     return SE::hessian(getEnergyDensity(ei), extractNodePositions(ei, m_x), mesh().elementData(ei), !projectionMask);;
        // });
        m_assembler.assembleHessianBlockAccelerated(H, getBlockHsp(), mesh(), [this, projectionMask](size_t ei) {
            return SE::hessian(getEnergyDensity(ei), extractNodePositions(ei, m_x), mesh().elementData(ei), !projectionMask);;
        });
    }

    // Construct a block-valued Hessian.
    void blockHessian(CSCMat &H, bool projectionMask = false) const {
        CSCMatrix<SuiteSparse_long, MNd> blockH;
        blockH.copySparsityPattern(getBlockHsp());

        BENCHMARK_SCOPED_TIMER_SECTION timer("Assemble Block Hessian");
        m_assembler.assembleBlockHessian(blockH, mesh(), [this, projectionMask](size_t ei) {
            return SE::hessian(getEnergyDensity(ei), extractNodePositions(ei, m_x), mesh().elementData(ei), !projectionMask);;
        });
    }

    virtual CSCMat hessianSparsityPattern(Real val = 0.0, VariableMask vmask = VariableMask::Defo) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolid.hessianSparsityPattern");
        if (vmask != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");
        return m_assembler.blockHessianSparsityPatternToScalar(getBlockHsp(), val);
    }

    const CSCMat &hessianBlockSparsityPattern(Real val = 0.0, VariableMask vmask = VariableMask::Defo) const {
        if (vmask != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");
        CSCMat &result = getBlockHsp();
        result.fill(val);
        return result;
    }

    virtual void massMatrix(CSCMat &M, bool /* updatedParametrization */, bool lumped) const override {
        M.setZero();
        MassMatrix::accumulate_vector_valued<>(mesh(), M, lumped);
    }

    virtual CSCMat sobolevInnerProductMatrix(Real Mscale = 1.0) const override {
        CSCMat result = Laplacian::construct_vector_valued<>(mesh());
        if (Mscale != 0.0)
            MassMatrix::accumulate_vector_valued<>(mesh(), result);
        return result;
    }

    VNd getNodePosition(size_t node_index) const { return m_x.row(node_index); }

    auto deformedVertices() const { return m_x.topRows(numVertices()); } // return slice of m_x
    const MXNd &deformedPositions() const { return m_x; } // deformed positions for all nodes
    MXNd restNodePositions() const {
        MXNd rpos(numNodes(), size_t(N));
        for (const auto n : mesh().nodes())
            rpos.row(n.index()) = n->p;
        return rpos;
    }
    MXNd nodeDisplacements() const { return deformedPositions() - restNodePositions(); }

    const Mesh &mesh() const { return *m_mesh; }

    const Energy &getEnergyDensity(size_t ei) const {
        if (m_energyDensities.size() == 1) return m_energyDensities.front();
        return m_energyDensities.at(ei);
    }

    // Extract the values of `f` at the nodes of element `ei`.
    auto extractNodePositions(size_t ei, const MXNd &f) const {
        NodePositions nodalValues;
        auto enodes = mesh().elementNodeIndices(ei);
        for (size_t lni = 0; lni < numNodesPerElement; ++lni)
            nodalValues.row(lni) = f.row(enodes[lni]);
        return nodalValues;
    }

    // Evaluate F = \nabla f within element `ei` using a linear combination of
    // the element's shape function gradients `gradPhis`.
    MNd jacobian(size_t ei, Eigen::Ref<const GradPhis> gradPhis, const MXNd &f) const {
        return (gradPhis * extractNodePositions(ei, f)).transpose();
    }

    // Evaluate F = \nabla f at barycentric coordinates `bc` in element `ei`.
    MNd jacobian(size_t ei, const EvalPtK &bc, const MXNd &f) const {
        return jacobian(ei, mesh().element(ei)->gradPhis(bc), f);
    }

    MNd getDeformationGradient(size_t ei, Eigen::Ref<const GradPhis> gradPhis) const {
        return jacobian(ei, gradPhis, m_x);
    }

    MNd getDeformationGradient(size_t ei, const EvalPtK &x) const {
        return getDeformationGradient(ei, mesh().element(ei)->gradPhis(x));
    }

    MNd getDeformationGradient(size_t ei, Eigen::Ref<const GradPhis> gradPhis, const NodePositions &nodalValues) const {
        return (gradPhis * nodalValues).transpose();
    }

    MNd getDeformationGradient(size_t ei, const EvalPtK &x, const NodePositions &nodalValues) const {
        return getDeformationGradient(ei, mesh().element(ei)->gradPhis(x), nodalValues);
    }

    VNd deformedPosition(size_t ei, const EvalPtK &x) const {
        VNd result(VNd::Zero());
        const auto &e = mesh().element(ei);
        for (const auto n : e.nodes())
            result += shapeFunction<Deg, K>(n.localIndex(), x) * m_x.row(n.index()).transpose();
        return result;
    }

    VNd deformedBoundaryPosition(size_t bei, const EvalPt<K - 1> &x) const {
        VNd result(VNd::Zero());
        const auto &be = mesh().boundaryElement(bei);
        for (const auto bn : be.nodes())
            result += shapeFunction<Deg, K - 1>(bn.localIndex(), x) * m_x.row(bn.volumeNode().index()).transpose();
        return result;
    }

    // Get the Green strain tensor at a particular point in element `ei`
    MNd greenStrain(size_t ei, const EvalPtK &x) const {
        MNd F = getDeformationGradient(ei, x);
        return 0.5 * (F.transpose() * F - MNd::Identity());
    }

    MNd cauchyStress(size_t ei, const EvalPtK &x) const {
        Energy psi(getEnergyDensity(ei), UninitializedDeformationTag());
        MNd F = getDeformationGradient(ei, x);
        psi.setDeformationGradient(F);
        // For all energies *except* `LinearElaticEnergy`, `denergy`
        // returns the PK1 stress (dpsi/dF) which must be transformed
        // to obtained the Cauchy stress.
        // For `LinearElaticEnergy`, dpsi/dF is actually the Cauchy stress
        // directly, and transforming it is wrong!
        if (isLinearElastic<Energy>::value) return psi.denergy();
        return (psi.denergy() * F.transpose()) / F.determinant();
    }

    Real vonMisesStress(size_t ei, const EvalPtK &x) const {
        // Note: this is very inefficient!
        return std::sqrt(vonMises(SymmetricMatrixValue<Real, N>(cauchyStress(ei, x))).frobeniusNormSq());
    }

    // Get the average Green strain tensor over element `ei`
    MNd greenStrain(size_t ei) const {
        return Quadrature<N, 2 * (Deg - 1)>::integrate( // This quadrature rule is always exact
            [ei, this](const EvalPtK &x) { return greenStrain(ei, x); }, 1.0);
    }

    // Get the average cauchy stress tensor over element `ei`
    MNd cauchyStress(size_t ei) const {
        return Quadrature<N, 2 * (Deg - 1)>::integrate( // Exact for linear elasticity
            [ei, this](const EvalPtK &x) { return cauchyStress(ei, x); }, 1.0);
    }

    std::vector<MNd> vertexGreenStrains() const {
        return vertexAveragedField(mesh(), [this](size_t ei, const EvalPtK &x) { return greenStrain(ei, x); });
    }

    std::vector<MNd> vertexCauchyStresses() const {
        return vertexAveragedField(mesh(), [this](size_t ei, const EvalPtK &x) { return cauchyStress(ei, x); });
    }

    // Compute an integral of integrand(be, x) over a single boundary element `bei`
    template<size_t QDeg, class Integrand>
    auto surfaceElementIntegral(const Integrand &integrand, size_t bei) const {
        const auto &m = mesh();
        auto be = m.boundaryElement(bei);
        return Quadrature<K - 1, QDeg>::integrate([&](const EvalPt<K - 1> &x) { return integrand(be, x); },
                                                  be->volume());
    }

    // Compute an integral of integrand(be, x) over the surface.
    template<size_t QDeg, class Integrand>
    auto surfaceIntegral(const Integrand &integrand) const {
        return summation_parallel([&](size_t bei) {
                return surfaceElementIntegral<QDeg>(integrand, bei);
        }, mesh().numBoundaryElements());
    }

    // The Lp norm of the von Mises Cauchy stress
    Real surfaceStressLpNorm(double p) const {
        Real integral = surfaceIntegral<2 * (Deg - 1)>([&](auto be, const EvalPt<K - 1> &x) {
            auto e = mesh().element(be.opposite().element().index());
            return restrictIntegrand([&](const EvalPt<K> &x_vol) {
                        return std::pow(vonMisesStress(e.index(), x_vol), p); }, be, e)(x);
        });
        return std::pow(integral, 1.0 / p);
    }

    VXd restElementVolumes() const {
        const auto &m = mesh();
        VXd result(m.numElements());
        for (const auto e : m.elements())
            result[e.index()] = e->volume();
        return result;
    }

    // Numerical approximation of each element's volume in the deformed config.
    VXd deformedElementVolumes() const {
        const auto &m = mesh();
        VXd result(m.numElements());
        for (const auto e : m.elements()) {
            result[e.index()] =
                SE::QuadratureRule::integrate([&](const EvalPt<K> &x) {
                        return getDeformationGradient(e.index(), x).determinant();
                    }, e->volume());
        }
        return result;
    }

    // Apply a rigid transformation `x --> R x + t` to the deformed configuration.
    void applyRigidTransform(const MNd &R, const VNd &t) {
        if (((R.transpose() * R - MNd::Identity()).norm() > 1e-8) || (R.determinant() < 0))
            throw std::runtime_error("R is not a rotation");
        setDeformedPositions(((m_x * R.transpose()).rowwise() + t.transpose()).eval());
    }

    // Reorient the current deformed configuration so that global rigid motions
    // can be pinned down with just 6 variable pin constraints.
    // Also return the indices of these 6 variables.
    using RMPins = RigidMotionPins<ElasticSolid>;
    typename RMPins::PinInfo
    prepareRigidMotionPins() {
        return RMPins::run(*this);
    }

    void filterRMPinArtifacts(const typename RMPins::PinVertices &pinVertices) {
        ::filterRMPinArtifacts(*this, pinVertices);
    }

    virtual std::unique_ptr<FieldSampler> referenceConfigSampler() const override {
        return FieldSampler::construct(std::shared_ptr<const Mesh>(m_mesh)); // work around template parameter deduction issue
    }

    virtual CSCMat deformationSamplerMatrix(Eigen::Ref<const Eigen::MatrixXd> P) const override {
        return fieldSamplerMatrix(mesh(), N, P);
    }

private:
    void m_setDefoVars(const Eigen::Ref<const VXd> &vars) override {
        if (size_t(vars.size()) != numDefoVars())
            throw std::invalid_argument("Invalid variable size");
        m_x = Eigen::Map<const MXNd>(vars.data(), m_x.rows(), m_x.cols());
    }

    void m_setRestVars(const Eigen::Ref<const VXd> &vars) override {
        if (size_t(vars.size()) != N * numVertices())
            throw std::invalid_argument("Invalid vertexPositions size");
        m_mesh->setNodePositions(Eigen::Map<const MXNd>(vars.data(), numVertices(), size_t(N)));
    }

protected:
    std::shared_ptr<Mesh> m_mesh;
    // Energy density for each element (with support for multi-material solids).
    // For single-material solids, this vector will contain only a single entry.
    std::vector<Energy> m_energyDensities;

    // Deformed positions for each node
    MXNd m_x;

    SystemAssembler<N> m_assembler;

    // Block Hessian sparisty pattern.
    mutable std::optional<CSCMat> m_blockHsp;

    // All template instantiations must be friends for the degree-converting constructor.
    template<size_t _K2, size_t _Deg2, class _EmbeddingSpace2, class _Energy2>
    friend struct ElasticSolid;

    template<size_t Deg2>
    void m_copy(const ElasticSolid<K, Deg2, EmbeddingSpace, Energy> &es) {
        // WARNING: this currently can only be called from a copy constructor
        // which properly initializes `m_assembler`!!!
        m_mesh = std::make_shared<Mesh>(es.mesh());
        m_energyDensities = es.m_energyDensities;
        auto oldDeformation = es.deformedPositions();

        const auto &m = mesh();
        // Transfer/interpolate deformation field to our new mesh.
        m_x.resize(numNodes(), size_t(N));
        for (const auto n : m.nodes()) {
            const size_t ni = n.index();
            if (n.isVertexNode()) m_x.row(ni) = oldDeformation.row(ni);
            else if (n.isEdgeNode()) {
                static_assert((Deg2 == 1) || (Deg2 == 2), "Only Degree 1 and 2 implemented");
                if (Deg2 == 2) { m_x.row(ni) = oldDeformation.row(ni); }
                else           { m_x.row(ni) = 0.5 * (oldDeformation.row(n.halfEdge().tail().index())
                                                    + oldDeformation.row(n.halfEdge(). tip().index())); }
            }
            else throw std::runtime_error("Unimplemented");
        }

        setDeformedPositions(m_x);
    }
};

#endif /* end of include guard: ELASTICSOLID_HH */
