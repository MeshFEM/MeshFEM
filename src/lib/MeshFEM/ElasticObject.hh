////////////////////////////////////////////////////////////////////////////////
// ElasticObject.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Represents a (potentially nonlinear) elastic object made of triangles/tets.
*///////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICOBJECT_HH
#define ELASTICOBJECT_HH

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
#include <Eigen/Sparse>

// _K: simplex dimension (2 ==> tri/3 ==> tet)
// _Deg: finite element degree (1 or 2)
// EmbeddingSpace: ND point type; Note N may differ from K (for a triangle mesh embedded in 3D, e.g.)
template<size_t _K, size_t _Deg, class EmbeddingSpace, class _Energy>
class ElasticObject {
public:
    using Real   = typename EmbeddingSpace::Scalar;
    using Energy = _Energy;
    static_assert(std::is_convertible<typename Energy::Real, Real>::value, "Incompatible real number types");

    static constexpr size_t K = _K;
    static constexpr size_t N = EmbeddingSpace::RowsAtCompileTime;
    static constexpr size_t Deg = _Deg;
    static constexpr size_t numNodesPerElement  = Simplex::numNodes(N, Deg);
    static constexpr size_t numElementLocalVars = N * numNodesPerElement;

    using QuadratureRule = Quadrature<N, 2 * (Deg - 1)>; // Exact for linear elasticity...
    using EvalPtN = EvalPt<N>;
    using Vector = Eigen::Matrix<Real, N, 1>;
    using Matrix = Eigen::Matrix<Real, N, N>;
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using MXNd = Eigen::Matrix<Real, Eigen::Dynamic, N, Eigen::RowMajor>; // Row major so that flattened order agrees with VField
    using Mesh = FEMMesh<K, Deg, Vector>;
    using VSFJ = VectorizedShapeFunctionJacobian<N, Vector>;

    ElasticObject(const Energy &energy, const Mesh &mesh)
        : m_mesh(mesh), m_energyDensities{{energy}} { setIdentityDeformation(); }

    size_t numVars() const { return m_x.size(); }
    size_t numElements() const { return m_mesh.numElements(); }
    size_t numVertices() const { return m_mesh.numVertices(); }
    size_t numRestStateVars() const { return numVertices() * N; }

    void setIdentityDeformation() {
        m_x.resize(m_mesh.numNodes(), N);
        for (const auto &n : m_mesh.nodes())
            m_x.row(n.index()) = n->p;
    }

    VXd getVars() const { return Eigen::Map<const VXd>(m_x.data(), m_x.size()); }
    void setVars(const VXd &vars) {
        if (size_t(vars.rows()) != numVars())
            throw std::invalid_argument("Invalid variable size");
        m_x = Eigen::Map<const MXNd>(vars.data(), m_x.rows(), m_x.cols());
    }

    void setRestState(const VXd &vertexPositions) {
        if (size_t(vertexPositions.size()) != N * numVertices())
            throw std::invalid_argument("Invalid vertexPositions size");
        m_mesh.setNodePositions(Eigen::Map<const MXNd>(vertexPositions.data(), numVertices(), N));
    }

    VXd getRestState() const {
        VXd rest_state(numRestStateVars());
        for (const auto& vertex : m_mesh.vertices())
            rest_state.template segment<N>(N * vertex.index()) = vertex.node()->p;
        return rest_state;
    }

    // TODO: boundary conditions (adapt from LinearElasticity::applyBoundaryConditions)
    // TODO: various stress measures

    // Evaluate the elastic energy stored in the object.
    Real energy() const {
        auto energy_summand = [&](size_t ei) {
            Energy psi(getEnergyDensity(ei), UninitializedDeformationTag());
            return QuadratureRule::integrate(
                [ei, &psi, this](const EvalPtN &x) {
                    psi.setDeformationGradient(getDeformationGradient(ei, x));
                    return psi.energy();
                }, m_mesh.element(ei)->volume());
        };

        return summation_parallel<Real>(energy_summand, m_mesh.numElements());
    }

    // Evaluate the gradient of the elastic energy with respect to the deformation variables.
    VXd gradient() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("gradient");
        VXd g(VXd::Zero(numVars()));

        auto accumulate_per_element_contrib = [&](size_t ei, VXd &g_out) {
            using LocalGradient = Eigen::Matrix<Real, numElementLocalVars, 1>;

            Energy psi(getEnergyDensity(ei), UninitializedDeformationTag());
            const auto &e = m_mesh.element(ei);
            auto contrib = QuadratureRule::integrate([&](const EvalPtN& x) {
                      LocalGradient integrand;
                      psi.setDeformationGradient(getDeformationGradient(ei, x));
                      Matrix denergy = psi.denergy();

                      for (const auto &n : e.nodes()) {
                          VSFJ gradPhi(0, e->gradPhi(n.localIndex())(x));
                          for (size_t c = 0; c < N; ++c) {
                              gradPhi.c = c;
                              integrand[N * n.localIndex() + c] = doubleContract(gradPhi, denergy);
                          }
                      }
                    return integrand;
                }, e->volume());
            for (const auto &n : e.nodes())
                g_out.template segment<N>(N * n.index()) += contrib.template segment<N>(N * n.localIndex());
        };

        assemble_parallel(accumulate_per_element_contrib, g, numElements());

        return g;
    }

    SuiteSparseMatrix hessian() const {
        SuiteSparseMatrix H(hessianSparsityPattern());
        hessian(H);
        return H;
    }

    // Simple columnwise flattening operation for (the upper triangle of) symmetric
    // matrices. Indices in the lower triangle are mapped to the upper triangle.
    static constexpr size_t perElementHessianFlattening(size_t i, size_t j) {
        return (i < j) ? i + (j * (j + 1)) / 2
                       : j + (i * (i + 1)) / 2;
    }

    void hessian(SuiteSparseMatrix& H) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Hessian");
        auto assembler_per_element_contrib = [&](size_t ei, SuiteSparseMatrix& Hout) {
            const auto &m = m_mesh;
            const auto &e = m.element(ei);
            Energy psi(getEnergyDensity(ei), UninitializedDeformationTag());
            VXd hessian_contribution = QuadratureRule::integrate([&](const EvalPtN &x) {
                    psi.setDeformationGradient(getDeformationGradient(ei, x));
                    Eigen::Matrix<Real, flatLen(numElementLocalVars), 1> contribution;

                    Eigen::Matrix<Real, N, numNodesPerElement> sfGrads;
                    for (const auto &n : e.nodes())
                        sfGrads.col(n.localIndex()) = e->gradPhi(n.localIndex())(x);

                    for (const auto &n_b : e.nodes()) {
                        VSFJ gradPhi_b(0, sfGrads.col(n_b.localIndex()));
                        for (size_t c_b = 0; c_b < N; ++c_b) {
                            size_t var_b = N * n_b.localIndex() + c_b;
                            gradPhi_b.c = c_b;
                            Matrix delta_denergy = psi.delta_denergy(gradPhi_b);
                            for (const auto &n_a : e.nodes()) {
                                VSFJ gradPhi_a(0, sfGrads.col(n_a.localIndex()));
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

            // Accumulate vertical strips into the global Sparse matrix.
            for (const auto &n_b : e.nodes()) {
                for (size_t c_b = 0; c_b < N; ++c_b) {
                    size_t  var_b = N * n_b.localIndex() + c_b;
                    size_t gvar_b = N * n_b.index() + c_b;
                    for (const auto &n_a : e.nodes()) {
                        size_t  var_a = N * n_a.localIndex();
                        size_t gvar_a = N * n_a.index();
                        if (gvar_a > gvar_b) continue;

                        Vector block;
                        size_t len = std::min(size_t(N), gvar_b - gvar_a + 1);
                        for (size_t c = 0; c < len; ++c)
                            block[c] = hessian_contribution(perElementHessianFlattening(var_a + c, var_b));
                        Hout.addNZ(gvar_a, gvar_b, block.topRows(len));
                    }
                }
            }
        };

        assemble_parallel(assembler_per_element_contrib, H, numElements());
    }

    SuiteSparseMatrix hessianSparsityPattern() const {
        TripletMatrix<Triplet<Real>> triplet_result(numVars(), numVars());
        triplet_result.symmetry_mode = TripletMatrix<Triplet<Real>>::SymmetryMode::UPPER_TRIANGLE;

        for (const auto &e : m_mesh.elements()) {
            for (const auto &n_b : e.nodes()) {
                for (size_t c_b = 0; c_b < N; ++c_b) {
                    for (const auto &n_a : e.nodes()) {
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

        SuiteSparseMatrix result(std::move(triplet_result));
        result.fill(0.);
        return result;
    }

    Vector getNodePosition(size_t node_index) const { return m_x.row(node_index); }

    MXNd deformedVertices() const  { return m_x.topRows(numVertices()); }
    MXNd deformedPositions() const { return m_x; } // deformed positions for all nodes

    const Mesh &mesh() const { return m_mesh; }

    const Energy &getEnergyDensity(size_t ei) const {
        if (m_energyDensities.size() == 1) return m_energyDensities.front();
        return m_energyDensities.at(ei);
    }

    Matrix getDeformationGradient(size_t ei, const EvalPtN &x) const {
        Matrix F(Matrix::Zero());
        const auto &e = m_mesh.element(ei);
        for (const auto &n : e.nodes())
            F += (e->gradPhi(n.localIndex())(x) * m_x.row(n.index())).transpose();
        return F;
    }

    VXd element3DVolumes() const {
        if (N != 3) { throw std::runtime_error("Only 3D meshes have element volumes"); }
        // For a tet mesh, the 3D volume associated with a tetrahedron is simply the tet's volume.
        const auto &m = mesh();
        VXd result(m.numElements());
        for (const auto &e : m.elements())
            result[e.index()] = e->volume();
        return result;
    }

protected:
    Mesh m_mesh;
    // Energy density for each element (with support for multi-material objects).
    // For single-material objects, this vector will contain only a single entry.
    std::vector<Energy> m_energyDensities;

    // Deformed positions for each node
    MXNd m_x;
};

#endif /* end of include guard: ELASTICOBJECT_HH */
