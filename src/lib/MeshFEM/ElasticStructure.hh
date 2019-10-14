#ifndef ELASTICSTRUCTURE_HH
#define ELASTICSTRUCTURE_HH

#include "FEMMesh.hh"
#include "GaussQuadrature.hh"
#include "GlobalBenchmark.hh"
#include "MeshIO.hh"
#include "ParallelAssembly.hh"
#include "SparseMatrices.hh"
#include "Flattening.hh"
#include "Types.hh"
#include "Functions.hh"
#include <Eigen/Sparse>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/iterator_range.hpp>

#ifndef PARALLEL_ASSEMBLY
    #define PARALLEL_ASSEMBLY true
#endif

#ifndef MESHFEM_WITH_TBB
#define PARALLEL_ASSEMBLY false
#endif


// template<typename Energy>
// concept bool EnergyType = 
//     require(Energy e) {
//         {energy()} -> Energy::Real;
//         {denergy()} -> Energy::Matrix;
//         {denergy(Energy::Matrix)} -> Energy::Real;
//         {d2energy(Energy::Matrix, Energy::Matrix)} -> Energy::Real;
//         {delta_denergy(Energy::Matrix)} -> Energy::Matrix;
//     };

// InfluencedVariable_T must be specialized for each derived
// structure classes inheriting from ElasticStructureBase. If the
// derived structure does not need to change the default behavior, the
// specialization can simply inherit from InfluencedVariable_T<ElasticStructure>.
// Otherwise, the specialization must be implemented to model the
// InfluencedVariableType concept defined below.
template<typename _EStructure>
class InfluencedVariable_T;

// template<typename InfluencedVariable>
// concept bool InfluencedVariableType = 
//     require(InfluencedVariable IV) {
//         // interfaces for InfluencedVariableIterator
//         {setNext()} -> void;
//         {getLocalIndex()} -> size_t;
//         // interfaces for structure impl (e.g. gradient computation)
//         {getIndex()} -> size_t;
//         {setDeltaGrad(Eigen::Matrix, const EvalPt<Dimension>&)} -> void;
//         {unsetDeltaGrad(Eigen::Matrix)} -> void;
//     };

// For more interface specification, please refer to document (elastic_structure.pdf)


template<typename IV>
class InfluencedVariableIterator_T
    : public boost::iterator_facade<InfluencedVariableIterator_T<IV>,
                                    const IV, boost::single_pass_traversal_tag,
                                    const IV&, size_t>
{
public:
    InfluencedVariableIterator_T(size_t local_index, size_t element_index,
                                const typename IV::Structure& m_elastic_structure)
        : m_variable(local_index, element_index, m_elastic_structure)
    {}

    InfluencedVariableIterator_T(const InfluencedVariableIterator_T&) = default;
    InfluencedVariableIterator_T& operator=(const InfluencedVariableIterator_T&) = default;

private:
    friend class boost::iterator_core_access;

    void increment() { m_variable.setNext(); }
    bool equal(const InfluencedVariableIterator_T& other) const { return other.m_variable.getLocalIndex() == m_variable.getLocalIndex(); }
    const IV& dereference() const { return m_variable; }

    IV m_variable;
};

template<typename Derived>
struct ElasticStructureTraits;

template<typename Derived>
class ElasticStructureBase {
public:
    using EStructure = Derived;
    using EST        = ElasticStructureTraits<Derived>;
    using Real       = typename EST::Real;
    using Energy     = typename EST::Energy;

    static constexpr int MATRIX_STORAGE_POLICY = Eigen::ColMajor;
    static constexpr Real MIN_MASS = 1e-9;

    static_assert(std::is_convertible<typename Energy::Real, Real>::value, "");

    static constexpr size_t Dimension                       = EST::Dimension;
    static constexpr size_t Degree                          = EST::Degree;
    static constexpr size_t NUM_INFLUENCED_VARS_PER_ELEMENT = EST::NUM_INFLUENCED_VARS_PER_ELEMENT;
    using InfluencedVariable                                = InfluencedVariable_T<Derived>;
    using InfluencedVariableIterator                        = InfluencedVariableIterator_T<InfluencedVariable>;

    using QuadratureRule = Quadrature<Dimension, 2 * (Degree - 1)>;
    using Vector  = Eigen::Matrix<Real, Dimension, 1>;
    using Matrix  = Eigen::Matrix<Real, Dimension, Dimension, MATRIX_STORAGE_POLICY>;
    using VectorX = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using EigenSparseMatrix = Eigen::SparseMatrix<Real, Eigen::ColMajor, SuiteSparse_long>;
    using Mesh = FEMMesh<Dimension, Degree, Vector>;

    ElasticStructureBase(const ElasticStructureBase& other) = default;
    ElasticStructureBase(const Energy& energy, const Mesh& mesh) 
        : ElasticStructureBase(energy, mesh, mesh.boundingBox().volume())
    {}
    ElasticStructureBase(const Energy& energy, const Mesh& mesh, Real volume)
        : m_mesh(mesh), m_energy(energy), m_volume(volume)
    {
        // NOTE: initialize() must be called by the derived class as the
        // derived class constructor might need to initialize some variables
        // before Base::initialize() is called.
        // (e.g. numOfFluctuationVariables)
    }

    /**
     * Initialize fluctuation variables and energy instance
     */
    void initialize() {
        getThis()->setIdentityDeformationGradient();
        m_elementEnergies.clear();
        for (size_t eIdx = 0; eIdx < m_mesh.numElements(); eIdx++)
            m_elementEnergies.push_back(Energy(m_energy));
    }


    Matrix getDeformationGradient(size_t element_index, const EvalPt<Dimension>& x) const {
        return Matrix::Identity() + getFluctuationDisplacementGradient(element_index, x);
    }
    void setIdentityDeformationGradient() { m_fluctuation_displacements = VectorX::Zero(getThis()->numNodeFluctuationDisplacementVars()); }
    
    
    Matrix getVariablesDeformationGradient(const VectorX& vars, size_t element_index, const EvalPt<Dimension>& x) const {
        return Matrix::Identity() + getVariablesFluctuationDisplacementGradient(vars, element_index, x);
    }
    Matrix getVariablesFluctuationDisplacementGradient(const VectorX& vars, size_t element_index, const EvalPt<Dimension>& x) const {
        Matrix fluctuation_displacement_gradient(Matrix::Zero());
        const auto& element = m_mesh.element(element_index);
        for (const auto& node : element.nodes()) {
            fluctuation_displacement_gradient += getVariablesNodeFluctuationDisplacement(vars, node.index()) 
                * element->gradPhi(node.localIndex())(x).transpose();
        }
        return fluctuation_displacement_gradient;
    }

    VectorX getVars() const { return fluctuationDisplacements(); }
    void setVars(const VectorX& vars) {
        if (size_t(vars.rows()) != getThis()->numVars()) { throw std::invalid_argument("Invalid variable size"); }
        m_fluctuation_displacements = vars;
    }
    size_t numVars() const { return numNodeFluctuationDisplacementVars(); }
    size_t numNodeFluctuationDisplacementVars() const { return Dimension * m_mesh.numNodes();}
    size_t numElements() const { return m_mesh.numElements(); }
    size_t numVertices() const { return m_mesh.numVertices(); }


    const VectorX& fluctuationDisplacements() const { return m_fluctuation_displacements; }
    void setFluctuationDisplacement(const VectorX& fluctuation_displacements) {
        if (fluctuation_displacements.rows() != getThis()->numNodeFluctuationDisplacementVars()) { throw std::invalid_argument("Invalid fluctuation displacement size"); }
        fluctuationDisplacements() = fluctuation_displacements;
    }
    void setNodeFluctuationDisplacement(size_t nodeIdx, size_t dim, Real value) {
        fluctuationDisplacements()(getThis()->fluctuationDisplacementVarIdx(nodeIdx, dim)) = value;
    }


    void setRestState(const VectorX& vertices_positions) {
        std::vector<Vector3D> positions(numVertices());
        for (const auto& vertex : m_mesh.vertices()) {
            positions[vertex.index()] = padTo3D(Vector(vertices_positions.template segment<Dimension>(Dimension * vertex.index())));
        }
        m_mesh.setNodePositions(positions);
    }
    VectorX getRestState() const {
        VectorX rest_state(Dimension * numVertices());
        for (const auto& vertex : m_mesh.vertices()) {
            rest_state.template segment<Dimension>(Dimension * vertex.index()) = vertex.node()->p;
        }
        return rest_state;
    }


    // TODO 
    Matrix getStressTensor() const {
        VectorX energy_gradient = gradient();
        
        return Matrix();
    }


    /**
     *  Return the elastic energy stored in one cell of the structure.
     */
    Real energy() const {
        Real energy = 0;
        auto f = [&](size_t element_index, Energy& energy) {
            return [&, element_index](const EvalPt<Dimension>& x) {
                energy.setDeformationGradient(getThis()->getDeformationGradient(element_index, x));
                auto v = energy.energy();
                return v;
            };
        };
#if !PARALLEL_ASSEMBLY  // Sequential
        for (const auto& element : m_mesh.elements()) {
            energy += QuadratureRule::integrate(f(element.index(), m_energy), element->volume());
        }
#else   // Parallel
        auto energy_summand = [&](size_t element_index, VectorX& summands) {
            summands[element_index] = QuadratureRule::integrate(
                f(element_index, m_elementEnergies[element_index] /*localEnergy*/),
                m_mesh.element(element_index)->volume());
        };
        energy = summation_parallel<Real>(energy_summand, m_mesh.numElements(), true);
#endif
        return energy / getVolume();
    }


    /**
     *  Return the gradient of the stored elastic energy in a cell with respect
     *  to the cell fluctuation displacement.
     */
    VectorX gradient() const {
        VectorX g(VectorX::Zero(getThis()->numVars()));

        auto accumulate_per_element_contrib = [&](size_t ei, VectorX& g_out) {
            Matrix delta_grad(Matrix::Zero());

            using LocalGradient = Eigen::Matrix<Real, numInfluencedVarsPerElements(), 1>;

            auto contrib = QuadratureRule::integrate(
                [ei, &delta_grad, this](const EvalPt<Dimension>& x) {
                      LocalGradient integrand;
                      auto &psi = m_elementEnergies[ei];
                      psi.setDeformationGradient(getThis()->getDeformationGradient(ei, x));

                      for (const auto &var : getInfluencedVariableRange(ei)) {
                          var.setDeltaGrad(delta_grad, x);
                          integrand[var.getLocalIndex()] = psi.denergy(delta_grad);
                          var.unsetDeltaGrad(delta_grad);
                      }
                      integrand /= getVolume();

                  return integrand;
              },
              m_mesh.element(ei)->volume());
            for (const auto &var : getInfluencedVariableRange(ei))
                g_out[var.getIndex()] += contrib[var.getLocalIndex()];
        };

#if PARALLEL_ASSEMBLY
            assemble_parallel(accumulate_per_element_contrib, g, numElements());
#else
            for (const auto &e : m_mesh.elements())
                accumulate_per_element_contrib(e.index(), g);
#endif

        return g;
    }

    /**
     *  Returns a matrix that maps a vector containing nodal displacements 
     *  entries, with variable indexing following that given by 
     *  fluctuationDisplacementVarIdx and maps it to a list of per-element 
     *  deformation gradient matrices, represented as flattened vectors, column by column.
     *
     *  One flattened matrix per element, PER QUADRATURE POINT is returned,
     *  the ordering is by elements (as in m_mesh.elements()) and then by
     *  quadrature point (as in Quadrature<>::points)
     */
    TripletMatrix<Triplet<Real>> deformationGradientMapTriplets() const {
        const auto& quadPoints = QuadratureRule::points;

        TripletMatrix<Triplet<Real>> triplets(getThis()->defGradVecSize(), getThis()->numVars());
        triplets.symmetry_mode = TripletMatrix<Triplet<Real>>::SymmetryMode::NONE;

        size_t eIdx = 0;
        for (const auto& element : m_mesh.elements()) {
            size_t qpIdx = 0;
            for (const auto& x : quadPoints) {
                for (const auto& node : element.nodes()) {
                    Vector gradPhi = element->gradPhi(node.localIndex())(x);
                    for (size_t defGradCol = 0; defGradCol < Dimension; defGradCol++) {
                        Real entry = gradPhi(defGradCol);
                        for (size_t defGradRow = 0; defGradRow < Dimension; defGradRow++) {
                            size_t defGradIdx = defGradsQPIdx(eIdx, qpIdx, defGradRow, defGradCol);
                            size_t nodeFlucIdx = getThis()->fluctuationDisplacementVarIdx(node.index(), defGradRow);
                            triplets.addNZ(defGradIdx, nodeFlucIdx, entry);
                        }
                    }
                }
                qpIdx++;
            }
            eIdx++;
        }
        return triplets;
    }

    EigenSparseMatrix deformationGradientMapEigen() const {
        TripletMatrix<Triplet<Real>> triplets = getThis()->deformationGradientMapTriplets();
        EigenSparseMatrix defGradMap(triplets.m, triplets.n);
        defGradMap.setFromTriplets(triplets.begin(), triplets.end());
        return defGradMap;
    }

    size_t defGradsQPIdx(size_t elementIdx, size_t qpIdx, size_t row, size_t col) const {
        return elementIdx * QuadratureRule::numPoints * Dimension * Dimension + qpIdx * Dimension * Dimension +
               col * Dimension + row;
    }

    size_t defGradVecSize() const { return numElements() * QuadratureRule::numPoints * Dimension * Dimension; }

    EigenSparseMatrix lumpedMassesDefGradsEigen(Real density) const {
        VectorX masses = getThis()->lumpedMassesDefGradsVec(density);
        EigenSparseMatrix massMat(masses.rows(), masses.rows());
        massMat.setIdentity();
        for (int i = 0; i < massMat.rows(); i++) {
            massMat.coeffRef(i, i) = masses(i);
        }
        return massMat;
    }

    VectorX lumpedMassesDefGradsVec(Real density) const {
        size_t numqps = QuadratureRule::numPoints;
        VectorX masses(getThis()->defGradVecSize());
        masses.setZero();
        // Fluctuation displacement masses
        SuiteSparse_long elIdx = 0;
        for (const auto& element : m_mesh.elements()) {
            Real curMassPerQP = std::max(MIN_MASS, (element->volume() * density) / (Real)numqps);
            for (size_t qpIdx = 0; qpIdx < numqps; qpIdx++) {
                for (size_t row = 0; row < Dimension; row++) {
                    for (size_t col = 0; col < Dimension; col++) {
                        SuiteSparse_long curIdx = defGradsQPIdx(elIdx, qpIdx, row, col);
                        masses(curIdx) += curMassPerQP;
                    }
                }
            }
            elIdx++;
        }

        return masses;
    }

    EigenSparseMatrix lumpedMassesEigen(Real density) const {
        size_t nVars = getThis()->numVars();
        EigenSparseMatrix massMat(nVars, nVars);
        massMat.setIdentity();
        VectorX masses = getThis()->lumpedMassesVec(density);
        for (SuiteSparse_long i = 0; i < massMat.rows(); i++) {
            massMat.coeffRef(i, i) = masses(i);
        }
        return massMat;
    }

    VectorX lumpedMassesVec(Real density) const {
        VectorX masses(getThis()->numVars());
        masses.setZero();
        // Fluctuation displacement masses
        for (const auto& element : m_mesh.elements()) {
            Real curMassPerNode = (element->volume() * density) / (Real)element.nodes().size();
            curMassPerNode = std::max(MIN_MASS, curMassPerNode);
            for (const auto& node : element.nodes()) {
                for (size_t d = 0; d < Dimension; d++) {
                    SuiteSparse_long curIdx = getThis()->fluctuationDisplacementVarIdx(node.index(), d);
                    masses(curIdx) += curMassPerNode;
                }
            }
        }

        return masses;
    }

    EigenSparseMatrix laplacianEigen(Real addM = 0) const {
        EigenSparseMatrix G = deformationGradientMapEigen();
        EigenSparseMatrix M = lumpedMassesDefGradsEigen(1.);
        EigenSparseMatrix laplacian = G.transpose() * M * G;
        if (addM > 0)
            laplacian += addM * lumpedMassesEigen(1.);
        return laplacian;
    }

    SuiteSparseMatrix laplacian(Real addM = 0) const {
        TripletMatrix<Triplet<Real>> triplets(getThis()->numVars(), getThis()->numVars());
        triplets.symmetry_mode = TripletMatrix<Triplet<Real>>::SymmetryMode::UPPER_TRIANGLE;
        EigenSparseMatrix lapEigen = laplacianEigen(addM);
        for (SuiteSparse_long k = 0; k < lapEigen.outerSize(); ++k) {
            for (typename EigenSparseMatrix::InnerIterator it(lapEigen, k); it; ++it) {
                if (it.row() <= it.col()) {
                    triplets.addNZ(it.row(), it.col(), it.value());
                }
            }
        }
        SuiteSparseMatrix laplacian(std::move(triplets));
        return laplacian;
    }

    SuiteSparseMatrix hessian() const {
        SuiteSparseMatrix H(hessianSparsityPattern());
        hessian(H);
        return H;
    }

    SuiteSparseMatrix variablesHessian(const VectorX& vars) const {
        SuiteSparseMatrix H(hessianSparsityPattern());
        variablesHessian(vars, H);
        return H;
    }

    // Note: One might want to factor out the two hessian computation function by passing getVars()
    // to the variablesHessian method. But this produces too many copies of the variables vector
    // and introduces a non negligible slowdown to hessian computation.
    // The factorization must be done in another way.
    /**
     *  Stores in the given matrix the hessian of the stored elastic
     *  energy in a cell with respect to the cell fluctuation displacement.
     *
     *  The sparse matrix must have the right sparsity pattern. See
     *  hessianSparsityPattern.
     */
    void hessian(SuiteSparseMatrix& H) const {
        BENCHMARK_START_TIMER("Hessian");
        static constexpr size_t contribution_dimension =numInfluencedVarsPerElements() * (numInfluencedVarsPerElements() + 1) / 2;

        auto assembler_per_element_contrib = [&](size_t element_index, SuiteSparseMatrix& Hout) {
            VectorX hessian_contribution = QuadratureRule::integrate(
                [&element_index, this](const EvalPt<Dimension>& x) {
                    std::vector<Matrix> delta_grads(numInfluencedVarsPerElements(), Matrix::Zero());
                    for (const auto& variable : getInfluencedVariableRange(element_index)) {
                        variable.setDeltaGrad(delta_grads[variable.getLocalIndex()], x);
                    }

                    m_elementEnergies[element_index].setDeformationGradient(getThis()->getDeformationGradient(element_index, x));

                    Eigen::Matrix<Real, contribution_dimension, 1> contribution;
                    for (const auto& variable_b : getInfluencedVariableRange(element_index)) {
                        Matrix delta_denergy = m_elementEnergies[element_index].delta_denergy(delta_grads[variable_b.getLocalIndex()]);
                        for (const auto& variable_a : getInfluencedVariableRange(element_index)) {
                            if (variable_a.getLocalIndex() > variable_b.getLocalIndex())
                                continue;

                            size_t variable_pair_index = getInfluencedVariablePairFlattenedIndex(variable_a, variable_b);
                            contribution[variable_pair_index] =
                                (delta_denergy.transpose() * delta_grads[variable_a.getLocalIndex()]).trace() / getVolume();
                        }
                    }

                    return contribution;
                },
                m_mesh.element(element_index)->volume());

            size_t hint = 0;
            for (const auto& variable_b : getInfluencedVariableRange(element_index)) {
                for (const auto& variable_a : getInfluencedVariableRange(element_index)) {
                    if (variable_a.getIndex() > variable_b.getIndex())
                        continue;

                    size_t variable_pair_index = getInfluencedVariablePairFlattenedIndex(variable_a, variable_b);
                    hint = Hout.addNZ(variable_a.getIndex(), variable_b.getIndex(),
                                      hessian_contribution[variable_pair_index], hint);
                }
            }
        };

#if PARALLEL_ASSEMBLY
        assemble_parallel(assembler_per_element_contrib, H, numElements());
#else
        for (size_t e = 0; e < numElements(); ++e) {
            assembler_per_element_contrib(e, H);
        }
#endif

        BENCHMARK_STOP_TIMER("Hessian");
    }

    /**
     *  Stores in the given matrix the hessian of the stored elastic
     *  energy in a cell with respect to a given set of variables for
     *  the fluctuation displacement.
     *
     *  The sparse matrix must have the right sparsity pattern. See
     *  hessianSparsityPattern.
     */
    void variablesHessian(const VectorX& vars, SuiteSparseMatrix& H) const {
        BENCHMARK_START_TIMER("Hessian");
        auto assembler_per_element_contrib = [&](size_t element_index, SuiteSparseMatrix& Hout) {
            VectorX hessian_contribution = QuadratureRule::integrate(
                [&element_index, &vars, this](const EvalPt<Dimension>& x) {
                    std::vector<Matrix> delta_grads(numInfluencedVarsPerElements(), Matrix::Zero());
                    for (const auto& variable : getInfluencedVariableRange(element_index)) {
                        variable.setDeltaGrad(delta_grads[variable.getLocalIndex()], x);
                    }

                    m_elementEnergies[element_index].setDeformationGradient(
                        getThis()->getVariablesDeformationGradient(vars, element_index, x));

                    static constexpr size_t contribution_dimension =
                        numInfluencedVarsPerElements() * (numInfluencedVarsPerElements() + 1) / 2;
                    Eigen::Matrix<Real, contribution_dimension, 1> contribution;
                    for (const auto& variable_b : getInfluencedVariableRange(element_index))
                    {
                        Matrix delta_denergy = m_elementEnergies[element_index].delta_denergy(
                            delta_grads[variable_b.getLocalIndex()]);
                        for (const auto& variable_a : getInfluencedVariableRange(element_index))
                        {
                            if (variable_a.getLocalIndex() > variable_b.getLocalIndex())
                                continue;

                            size_t variable_pair_index =
                                getInfluencedVariablePairFlattenedIndex(variable_a, variable_b);
                            contribution[variable_pair_index] =
                                (delta_denergy.transpose() * delta_grads[variable_a.getLocalIndex()])
                                .trace() /
                                getVolume();
                        }
                  }

                  return contribution;
              },
              m_mesh.element(element_index)->volume());

            size_t hint = 0;
            for (const auto& variable_b : getInfluencedVariableRange(element_index)) {
                for (const auto& variable_a : getInfluencedVariableRange(element_index)) {
                    if (variable_a.getIndex() > variable_b.getIndex())
                        continue;

                    size_t variable_pair_index =
                      getInfluencedVariablePairFlattenedIndex(variable_a, variable_b);

                    hint = Hout.addNZ(variable_a.getIndex(),
                                      variable_b.getIndex(),
                                      hessian_contribution[variable_pair_index],
                                      hint);
                }
            }
        };

#if!PARALLEL_ASSEMBLY
            assemble_parallel(assembler_per_element_contrib, H, numElements());
#else
            for (size_t e = 0; e < numElements(); ++e) {
                assembler_per_element_contrib(e, H);
            }
#endif

        BENCHMARK_STOP_TIMER("Hessian");
    }

    SuiteSparseMatrix hessianSparsityPattern() const {
        size_t nVars = getThis()->numVars();
        TripletMatrix<Triplet<Real>> triplet_result(nVars, nVars);
        triplet_result.symmetry_mode = TripletMatrix<Triplet<Real>>::SymmetryMode::UPPER_TRIANGLE;

        // Since the Hessian is symmetric only compute the upper triangle
        for (const auto& element : m_mesh.elements()) {
            for (const auto& variable_a : getInfluencedVariableRange(element.index())) {
                for (const auto& variable_b : getInfluencedVariableRange(element.index())) {
                    if (variable_a.getIndex() > variable_b.getIndex())
                        continue;
                    triplet_result.addNZ(variable_a.getIndex(), variable_b.getIndex(), 1.);
                }
            }
        }

        SuiteSparseMatrix result(std::move(triplet_result));
        result.fill(0.);
        return result;
    }

    Vector getNodePosition(size_t node_index) const { return m_mesh.node(node_index)->p + getNodeFluctuationDisplacement(node_index); }

    auto getNodeFluctuationDisplacement(size_t node_index) const {
        return m_fluctuation_displacements.template segment<Dimension>(fluctuationDisplacementLocalVarIdx(node_index, 0));
    }

    auto getVariablesNodeFluctuationDisplacement(const VectorX& vars, size_t node_index) const {
        return vars.template segment<Dimension>(getThis()->fluctuationDisplacementVarIdx(node_index, 0));
    }

    std::vector<size_t> getNodeFluctuationDisplacementVarIndices(size_t node_index) const {
        std::vector<size_t> result(Dimension);

        for (size_t i = 0; i < Dimension; ++i) {
            result[i] = getThis()->fluctuationDisplacementVarIdx(node_index, i);
        }

        return result;
    }

    std::vector<size_t> getNodeIndicesForVertices(const std::vector<size_t>& vertex_indices) const {
        std::vector<size_t> node_indices;
        for (size_t ind : vertex_indices)
            node_indices.push_back(ind);

        // Check all edge nodes to see if they are on an edge where both outer
        // vertices are contained in the vertex_indices set
        for (const auto& node : m_mesh.nodes()) {
            // Check if node is an edge node
            if (node.edgeNodeIndex() > 0) {
                // Check the endpoints of its edge:
                auto edge = m_mesh.edgeForEdgeNode(node.edgeNodeIndex());
                if (std::find(vertex_indices.begin(), vertex_indices.end(), edge[0]) !=
                      vertex_indices.end() &&
                    std::find(vertex_indices.begin(), vertex_indices.end(), edge[1]) !=
                      vertex_indices.end())
                {
                    node_indices.push_back(node.index());
                }
            }
        }

        return node_indices;
    }

    size_t fluctuationDisplacementVarIdx(size_t node_index, size_t component) const { return fluctuationDisplacementLocalVarIdx(node_index, component);}
    size_t fluctuationDisplacementLocalVarIdx(size_t node_index, size_t component) const {
        return Dimension * getThis()->getNodeDOFIndex(node_index) + component;
    }
    size_t getNodeDOFIndex(size_t node_index) const { return node_index; }

    Real getVolume() const { return m_volume; }
    const Mesh& mesh() const { return m_mesh; }
    Energy getEnergyDensity() const { return m_energy; }

    static constexpr size_t numNodesPerElements() { return Simplex::numNodes(Dimension, Degree); }
    static constexpr size_t numNodesPerFaces() { return Simplex::numNodes(Dimension - 1, Degree); }

protected:

    boost::iterator_range<InfluencedVariableIterator> getInfluencedVariableRange(size_t element_index) const
    {
        return boost::make_iterator_range(
          InfluencedVariableIterator(0, element_index, *getThis()),
          InfluencedVariableIterator(numInfluencedVarsPerElements(), element_index, *getThis()));
    }

    size_t getInfluencedVariablePairFlattenedIndex(const InfluencedVariable& variable_a, const InfluencedVariable& variable_b) const {
        return flattenIndices(numInfluencedVarsPerElements(), variable_a.getLocalIndex(), variable_b.getLocalIndex());
    }

    const Derived* getThis() const { return  static_cast<const Derived*>(this); }
    Derived* getThis() { return static_cast<Derived*>(this); }
    /**
     * Return the index of the fluctuation displacement variable within the
     * fluctuation displacement variable in the variable's element.
     */
    size_t fluctuationDisplacementVarElementNodesIdx(size_t node_index, size_t component) const {
        return Dimension * node_index + component;
    }

    Matrix getFluctuationDisplacementGradient(size_t element_index, const EvalPt<Dimension>& x) const {
        Matrix fluctuation_displacement_gradient(Matrix::Zero());
        const auto& element = m_mesh.element(element_index);
        for (const auto& node : element.nodes()) {
            fluctuation_displacement_gradient += getNodeFluctuationDisplacement(node.index()) *
                                                 element->gradPhi(node.localIndex())(x).transpose();
        }
        return fluctuation_displacement_gradient;
    }

    VectorX& fluctuationDisplacements() { return m_fluctuation_displacements; }

    static constexpr size_t numInfluencedVarsPerElements() { return NUM_INFLUENCED_VARS_PER_ELEMENT; }

    Mesh m_mesh;
    // This is mutable because the change of the deformation gradient stored
    // in the energy doesn't actually change the behaviour of the class.
    mutable Energy m_energy;
    mutable std::vector<Energy> m_elementEnergies;
    Real m_volume;

    VectorX m_fluctuation_displacements;
};

template<typename _Real, typename _Energy, size_t _Dimension, size_t _Degree>
class ElasticStructure: 
    public ElasticStructureBase<ElasticStructure<_Real, _Energy, _Dimension, _Degree>> {
public:
    using Base = ElasticStructureBase<ElasticStructure>;
    using Energy = typename Base::Energy;
    using Mesh = typename Base::Mesh;
    using Real = typename Base::Real;

    ElasticStructure(const Energy& energy, const Mesh& mesh) 
        : ElasticStructure(energy, mesh, mesh.boundingBox().volume())
    {}

    ElasticStructure(const Energy& energy, const Mesh& mesh, Real volume)
        : Base(energy, mesh, volume)
    {
        Base::initialize();
    }
};

template<typename _Real, typename _Energy, size_t _Dimension, size_t _Degree>
class InfluencedVariable_T<ElasticStructure<_Real, _Energy, _Dimension, _Degree>>{
public:
    using Structure = ElasticStructure<_Real, _Energy, _Dimension, _Degree>;
    using Mesh = typename Structure::Mesh;
    static constexpr size_t Dimension = Structure::Dimension;
    using Matrix = typename Structure::Matrix;
    using Vector = typename Structure::Vector;

    InfluencedVariable_T(size_t local_index, size_t element_index, const Structure& elastic_structure)
        : m_local_index(local_index), m_element_index(element_index), m_elastic_structure(elastic_structure)
    {}

    InfluencedVariable_T(const InfluencedVariable_T& other)
        : m_local_index(other.m_local_index), m_element_index(other.m_element_index), m_elastic_structure(other.m_elastic_structure)
    {}

    /**
     *  Return the index of the variable within the variables influenced by the given element.
     *  Watchout, the local index is not necessarily increasing with each call to setNext().
     */
    size_t getLocalIndex() const { return m_local_index; }
    size_t getIndex() const {
        return m_elastic_structure.fluctuationDisplacementVarIdx(
            element().node(getCurrentNodeElementLocalIndex()).index(), getCurrentNodeComponent());
    }

    /**
     *  Change the instance to represent the next variable in the set of influenced variable.
     */
    void setNext() { ++m_local_index; }

    /**
     * Set \a delta_grad such that m_energy.denergy(delta_grad) is the partial derivative of
     * the energy density with respect to the variable. The input matrix is supposed to be
     * the null matrix.
     */
    void setDeltaGrad(Matrix& delta_grad, const EvalPt<Dimension>& x) const {
        delta_grad.row(getCurrentNodeComponent()) = element()->gradPhi(getCurrentNodeElementLocalIndex())(x);
    }

    /**
     *  Set the entries of delta_grad changed by setDeltaGrad to 0..
     */
    void unsetDeltaGrad(Matrix& delta_grad) const { delta_grad.row(getCurrentNodeComponent()) = Vector::Zero(); }

private:
    auto element() const { return m_elastic_structure.mesh().element(m_element_index); }
    size_t getCurrentNodeElementLocalIndex() const { return (m_local_index) / Dimension; }
    size_t getCurrentNodeComponent() const { return (m_local_index) % Dimension; }

    size_t m_local_index;
    size_t m_node_element_local_index;
    size_t m_element_index;
    const Structure& m_elastic_structure;
};

template<typename _Real, typename _Energy, size_t _Dimension, size_t _Degree>
struct ElasticStructureTraits<ElasticStructure<_Real, _Energy, _Dimension, _Degree>>
{
    using Energy = _Energy;
    using Real   = _Real;
    static constexpr size_t Dimension = _Dimension;
    static constexpr size_t Degree = _Degree;
    static constexpr size_t NUM_INFLUENCED_VARS_PER_ELEMENT = Dimension * Simplex::numNodes(Dimension, Degree);
};

#endif
