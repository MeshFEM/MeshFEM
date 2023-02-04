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
#include "Flattening.hh"
#include "Types.hh"
#include "Functions.hh"
#include "EnergyDensities/Tensor.hh"
#include "EnergyDensities/EnergyTraits.hh"
#include "FieldSamplerMatrix.hh"
#include <Eigen/Sparse>

#include <atomic>

#include "RigidMotionPins.hh"
#include "FieldPostProcessing.hh"
#include "InterpolantRestriction.hh"

#include "ElasticObject.hh"
#include "MassMatrix.hh"
#include "Laplacian.hh"
#include "VonMises.hh"

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
    using CSCMat  = typename Base::CSCMat;
    using Base::numVars;
    using VariableMask = typename Base::VariableMask;

    static constexpr size_t K = _K;
    static constexpr size_t N = EmbeddingSpace::RowsAtCompileTime;
    static constexpr size_t Deg = _Deg;
    static constexpr size_t numNodesPerElement  = Simplex::numNodes(N, Deg);
    static constexpr size_t numElementLocalVars = N * numNodesPerElement;

    using QuadratureRule = Quadrature<N, 2 * (Deg - 1)>; // Exact for linear elasticity or linear FEM...
    using EvalPtK = EvalPt<K>;
    using Vector = Eigen::Matrix<Real, N, 1>;
    using Matrix = Eigen::Matrix<Real, N, N>;
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using MXNd = Eigen::Matrix<Real, Eigen::Dynamic, N, Eigen::RowMajor>; // Row major so that flattened order agrees with VField
    using Mesh = FEMMesh<K, Deg, Vector>;
    using VSFJ = VectorizedShapeFunctionJacobian<N, Vector>;
    using GradPhis = typename Mesh::ElementData::GradPhis;

    ElasticSolid(const Energy &energy, const std::shared_ptr<Mesh> &mesh)
        : m_mesh(mesh), m_energyDensities{{energy}} { setIdentityDeformation(); }

    // Copy and degree-changing constructor
    template<size_t Deg2>
    ElasticSolid(const ElasticSolid<K, Deg2, EmbeddingSpace, Energy> &es) { m_copy(es); }
    // Note: the degree-changing constructor template is excluded from overload resolution,
    // and the implicitly copy constructor is deleted due to the `m_varLocks` member...
    ElasticSolid(const ElasticSolid &es) { m_copy(es); }

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
        BENCHMARK_SCOPED_TIMER_SECTION timer("energy");
        return summation_parallel([this](size_t ei) { return elementEnergy(ei); },
                                  mesh().numElements());
    }

    // Energy stored in a single element.
    Real elementEnergy(size_t ei) const {
        Energy psi(getEnergyDensity(ei), UninitializedDeformationTag());
        return QuadratureRule::integrate(
            [ei, &psi, this](const EvalPtK &x) {
                psi.setDeformationGradient(getDeformationGradient(ei, x), EvalLevel::EnergyOnly);
                return psi.energy();
            }, mesh().element(ei)->volume());
    }

    // Gradient of a single element's energy with respect to its nodes' deformed positions..
    using ElementGradient = Eigen::Matrix<Real, numElementLocalVars, 1>;
    ElementGradient elementGradient(size_t ei) const {
        Energy psi(getEnergyDensity(ei), UninitializedDeformationTag());
        const auto &e = mesh().element(ei);
        return QuadratureRule::integrate([&](const EvalPtK &x) {
                  ElementGradient integrand;
                  GradPhis gradPhis = e->gradPhis(x);
                  psi.setDeformationGradient(getDeformationGradient(ei, gradPhis), EvalLevel::Gradient);
                  Matrix denergy = psi.denergy();

                  for (const auto n : e.nodes()) {
                      VSFJ gradPhi(0, gradPhis.col(n.localIndex()));
                      for (size_t c = 0; c < N; ++c) {
                          gradPhi.c = c;
                          integrand[N * n.localIndex() + c] = doubleContract(gradPhi, denergy);
                      }
                  }
                return integrand;
            }, e->volume());
    }

    // Gradient of the full object's energy with respect to all deformation variables.
    virtual VXd gradient(bool /* updatedParametrization */ = false, VariableMask vmask = VariableMask::Defo) const override {
        if (vmask != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");
        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolid.gradient");
        VXd g(VXd::Zero(numVars()));

        auto accumulate_per_element_contrib = [&](size_t ei, VXd &g_out) {
            ElementGradient contrib = elementGradient(ei);
            for (const auto n : mesh().element(ei).nodes())
                g_out.template segment<N>(N * n.index()) += contrib.template segment<N>(N * n.localIndex());
        };

        assemble_parallel(accumulate_per_element_contrib, g, numElements());

        return g;
    }

    CSCMat hessian(bool projectionMask = false) const {
        CSCMat H(hessianSparsityPattern());
        hessian(H, projectionMask);
        return H;
    }

    // Simple columnwise flattening operation for (the upper triangle of) symmetric
    // matrices. Indices in the lower triangle are mapped to the upper triangle.
    static constexpr size_t perElementHessianFlattening(size_t i, size_t j) {
        return (i < j) ? i + (j * (j + 1)) / 2
                       : j + (i * (i + 1)) / 2;
    }

    using PerElementHessian = Eigen::Matrix<Real, flatLen(numElementLocalVars), 1>;
    PerElementHessian elementHessian(size_t ei, bool disableProjection = false) const {
        Energy psi(getEnergyDensity(ei), UninitializedDeformationTag());
        const auto &m = mesh();
        const auto &e = m.element(ei);
        return QuadratureRule::integrate([&](const EvalPtK &x) {
                GradPhis gradPhis = e->gradPhis(x);
                psi.setDeformationGradient(getDeformationGradient(ei, gradPhis), disableProjection ? EvalLevel::HessianWithDisabledProjection
                                                                                                   : EvalLevel::Hessian);
                Eigen::Matrix<Real, flatLen(numElementLocalVars), 1> contribution;

                for (const auto n_b : e.nodes()) {
                    VSFJ gradPhi_b(0, gradPhis.col(n_b.localIndex()));
                    for (size_t c_b = 0; c_b < N; ++c_b) {
                        size_t var_b = N * n_b.localIndex() + c_b;
                        gradPhi_b.c = c_b;
                        Matrix delta_denergy = psi.delta_denergy(gradPhi_b);
                        for (const auto n_a : e.nodes()) {
                            VSFJ gradPhi_a(0, gradPhis.col(n_a.localIndex()));
                            for (size_t c_a = 0; c_a < N; ++c_a) {
                                size_t var_a = N * n_a.localIndex() + c_a;
                                gradPhi_a.c = c_a;
                                contribution[perElementHessianFlattening(var_a, var_b)] = doubleContract(gradPhi_a, delta_denergy);
                            }
                        }
                    }
                }

                return contribution;
            },
            e->volume());
    }

    virtual void hessian(CSCMat &H, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const override {
        if (vmask != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");
        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolid.hessian");

        auto &varLocks = m_getVarLocks();
        get_hessian_assembly_arena().execute([&H, &varLocks, this, projectionMask]() {
            parallel_for_range(mesh().numElements(), [&H, &varLocks, this, projectionMask](size_t ei) {
                const auto &m = mesh();
                const auto &e = m.element(ei);
                PerElementHessian contrib = elementHessian(ei, /* disableProjection */ !projectionMask);

                for (const auto n_b : e.nodes()) {
                    for (size_t c_b = 0; c_b < N; ++c_b) {
                        size_t  var_b = N * n_b.localIndex() + c_b;
                        size_t gvar_b = N * n_b.index() + c_b;

                        while (varLocks[gvar_b].exchange(true, std::memory_order_acquire)); // lock column gvar_b

                        // Accumulate vertical strips to column gvar_b
                        for (const auto n_a : e.nodes()) {
                            size_t  var_a = N * n_a.localIndex();
                            size_t gvar_a = N * n_a.index();
                            if (gvar_a > gvar_b) continue;

                            Vector block;
                            size_t len = std::min(size_t(N), gvar_b - gvar_a + 1);
                            for (size_t c = 0; c < len; ++c)
                                block[c] = contrib(perElementHessianFlattening(var_a + c, var_b));
                            H.addNZStrip(gvar_a, gvar_b, block.topRows(len));
                        }
                        varLocks[gvar_b].store(false, std::memory_order_release); // unlock column gvar_b
                    }
                }
            });
        });
    }

    virtual CSCMat hessianSparsityPattern(Real val = 0.0, VariableMask vmask = VariableMask::Defo) const override {
        if (vmask != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");
        TripletMatrix<Triplet<Real>> triplet_result(numVars(), numVars());
        triplet_result.symmetry_mode = TripletMatrix<Triplet<Real>>::SymmetryMode::UPPER_TRIANGLE;

        for (const auto e : mesh().elements()) {
            for (const auto n_b : e.nodes()) {
                for (size_t c_b = 0; c_b < N; ++c_b) {
                    for (const auto n_a : e.nodes()) {
                        for (size_t c_a = 0; c_a < N; ++c_a) {
                            size_t var_b = N * n_b.index() + c_b,
                                   var_a = N * n_a.index() + c_a;
                            if (var_a > var_b) continue;
                            triplet_result.addNZ(var_a, var_b, 1.0);
                        }
                    }
                }
            }
        }

        CSCMat result(std::move(triplet_result));
        result.fill(val);
        return result;
    }

    CSCMat scalarHessianSparsityPattern(Real val = 0.0, VariableMask vmask = VariableMask::Defo) const {
        if (vmask != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");
        TripletMatrix<Triplet<Real>> triplet_result(numVars(), numVars());
        triplet_result.symmetry_mode = TripletMatrix<Triplet<Real>>::SymmetryMode::UPPER_TRIANGLE;

        for (const auto e : mesh().elements()) {
            for (const auto n_b : e.nodes()) {
                for (const auto n_a : e.nodes()) {
                    size_t var_b = n_b.index(),
                           var_a = n_a.index();
                    if (var_a > var_b) continue;
                    triplet_result.addNZ(var_a, var_b, 1.0);
                }
            }
        }

        CSCMat result(std::move(triplet_result));
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

    Vector getNodePosition(size_t node_index) const { return m_x.row(node_index); }

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

    // Evaluate F = \nabla f within element `ei` using a linear combination of
    // the element's shape function gradients `gradPhis`.
    Matrix jacobian(size_t ei, Eigen::Ref<const GradPhis> gradPhis, const MXNd &f) const {
        Matrix F(Matrix::Zero());
        const auto &e = mesh().element(ei);
        for (const auto n : e.nodes()) {
            F += (gradPhis.col(n.localIndex()) * f.row(n.index())).transpose();
        }
        return F;
    }

    // Evaluate F = \nabla f at barycentric coordinates `bc` in element `ei`.
    Matrix jacobian(size_t ei, const EvalPtK &bc, const MXNd &f) const {
        return jacobian(ei, mesh().element(ei)->gradPhis(bc), f);
    }

    Matrix getDeformationGradient(size_t ei, Eigen::Ref<const GradPhis> gradPhis) const {
        return jacobian(ei, gradPhis, m_x);
    }

    Matrix getDeformationGradient(size_t ei, const EvalPtK &x) const {
        return getDeformationGradient(ei, mesh().element(ei)->gradPhis(x));
    }

    Vector deformedPosition(size_t ei, const EvalPtK &x) const {
        Vector result(Vector::Zero());
        const auto &e = mesh().element(ei);
        for (const auto n : e.nodes())
            result += shapeFunction<Deg, K>(n.localIndex(), x) * m_x.row(n.index()).transpose();
        return result;
    }

    Vector deformedBoundaryPosition(size_t bei, const EvalPt<K - 1> &x) const {
        Vector result(Vector::Zero());
        const auto &be = mesh().boundaryElement(bei);
        for (const auto bn : be.nodes())
            result += shapeFunction<Deg, K - 1>(bn.localIndex(), x) * m_x.row(bn.volumeNode().index()).transpose();
        return result;
    }

    // Get the Green strain tensor at a particular point in element `ei`
    Matrix greenStrain(size_t ei, const EvalPtK &x) const {
        Matrix F = getDeformationGradient(ei, x);
        return 0.5 * (F.transpose() * F - Matrix::Identity());
    }

    Matrix cauchyStress(size_t ei, const EvalPtK &x) const {
        Energy psi(getEnergyDensity(ei), UninitializedDeformationTag());
        Matrix F = getDeformationGradient(ei, x);
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
    Matrix greenStrain(size_t ei) const {
        return Quadrature<N, 2 * (Deg - 1)>::integrate( // This quadrature rule is always exact
            [ei, this](const EvalPtK &x) { return greenStrain(ei, x); }, 1.0);
    }

    // Get the average cauchy stress tensor over element `ei`
    Matrix cauchyStress(size_t ei) const {
        return Quadrature<N, 2 * (Deg - 1)>::integrate( // Exact for linear elasticity
            [ei, this](const EvalPtK &x) { return cauchyStress(ei, x); }, 1.0);
    }

    std::vector<Matrix> vertexGreenStrains() const {
        return vertexAveragedField(mesh(), [this](size_t ei, const EvalPtK &x) { return greenStrain(ei, x); });
    }

    std::vector<Matrix> vertexCauchyStresses() const {
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
                QuadratureRule::integrate([&](const EvalPt<K> &x) {
                        return getDeformationGradient(e.index(), x).determinant();
                    }, e->volume());
        }
        return result;
    }

    // Apply a rigid transformation `x --> R x + t` to the deformed configuration.
    void applyRigidTransform(const Matrix &R, const Vector &t) {
        if (((R.transpose() * R - Matrix::Identity()).norm() > 1e-8) || (R.determinant() < 0))
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

    // All template instantiations must be friends for the degree-converting constructor.
    template<size_t _K2, size_t _Deg2, class _EmbeddingSpace2, class _Energy2>
    friend struct ElasticSolid;

    // Spin locks used for parallel Hessian assembly.
    mutable std::unique_ptr<std::vector<std::atomic<bool>>> m_varLocks;
    auto &m_getVarLocks() const {
        if (!m_varLocks) {
            const size_t nv = numVars(VariableMask::All);
            m_varLocks = std::make_unique<std::vector<std::atomic<bool>>>(nv);
            for (size_t i = 0; i < nv; ++i)
                atomic_init(&(*m_varLocks)[i], false);
        }
        return *m_varLocks;
    }

    template<size_t Deg2>
    void m_copy(const ElasticSolid<K, Deg2, EmbeddingSpace, Energy> &es) {
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
