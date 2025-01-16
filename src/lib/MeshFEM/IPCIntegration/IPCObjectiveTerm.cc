#include "IPCObjectiveTerm.hh"
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/SystemAssembler.hh>

#if MESHFEM_WITH_IPC_TOOLKIT

template<typename _Real>
IPCObjectiveTerm<_Real>::IPCObjectiveTerm(std::shared_ptr<EO> eo, const ObstaclesCollection &obsts)
    : NewtonObjectiveTerm(eo), m_obj(eo), m_k(1.0e6)
{
    m_combinedCollisionMesh = std::make_unique<CombinedCollisionMesh<Real>>(*eo, obsts);
    m_collisionVertexPositions = m_combinedCollisionMesh->vertexPositionsForVars(getVars());
    
    m_N = m_combinedCollisionMesh->N;
    m_ipcWrapper = make_ipc_wrapper(*m_combinedCollisionMesh, m_collisionVertexPositions);

    if (!m_ipcWrapper) throw std::runtime_error("Unsupported dimension " + std::to_string(m_N));
    m_ipcWrapper->dhat = m_combinedCollisionMesh->getBboxDiagonal() * 1e-3;//std::pow(m_combinedCollisionMesh.bboxDiagonal,2)*1e-4 / 2.0 ;
    
    m_buildCollisionConstraints();

    if (!obsts.size()) setAdaptiveTimestep(false);
}

template<typename _Real>
IPCObjectiveTerm<_Real>::~IPCObjectiveTerm() { }

template<typename _Real>
size_t IPCObjectiveTerm<_Real>::numCollisionConstraints() const { return m_ipcWrapper->numCollisionConstraints(); }

template<typename _Real>
void IPCObjectiveTerm<_Real>::m_buildCollisionConstraints() {
    m_ipcWrapper->build_collision_constraints(m_collisionVertexPositions);
}

template<typename _Real>
_Real IPCObjectiveTerm<_Real>::contactPotentialEnergy() const {
    return m_ipcWrapper->compute_potential(m_collisionVertexPositions);
}

template<typename _Real>
typename IPCObjectiveTerm<_Real>::VXd IPCObjectiveTerm<_Real>::contactPotentialGradient() const {
    return m_ipcWrapper->compute_potential_gradient(m_collisionVertexPositions);
}

template<typename _Real>
bool IPCObjectiveTerm<_Real>::detectSparsityPatternChange(const NewtonHessian &oldHsp) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.m_updateContactHessianSparsityPattern");
    if (!oldHsp.H_ss) throw std::logic_error("IPCObjectiveTerm::detectSparsityPatternChange called with uninitialized sparsity pattern");
    size_t changed = m_ipcWrapper->detect_contact_set_change(*(oldHsp.H_ss), m_combinedCollisionMesh->nodeForCollisionMeshVertex);
    return (changed > sparsityPatternUpdateThreshold);
}

template<typename _Real>
NewtonHessian IPCObjectiveTerm<_Real>::hessianSparsityPattern() const {
    return NewtonHessian(m_ipcWrapper->block_hessian_sparsity_pattern(m_combinedCollisionMesh->nodeForCollisionMeshVertex));
}

template<typename _Real>
void IPCObjectiveTerm<_Real>::accumulateHessian(Real weight, NewtonHessian &H, bool projectionMask) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.accumulateHessian");
    m_ipcWrapper->hessian(H, m_collisionVertexPositions, m_combinedCollisionMesh->nodeForCollisionMeshVertex, weight * m_k, projectionMask);
}

template<typename _Real>
_Real IPCObjectiveTerm<_Real>::customFeasibleStepLength(const VXd &vars, const VXd &step) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.feasibleStepLength");

    const MXd &cvp0 = getCollisionVertexPositions();
    const MXd &cvp1 = m_combinedCollisionMesh->vertexPositionsForVars(vars + step);

    if (CCD == CCDMethod::CFL)            return m_ipcWrapper->compute_collision_cfl_stepsize(cvp0, cvp1);
    if (CCD == CCDMethod::TightInclusion) return m_ipcWrapper->compute_collision_tightInclusion_stepsize(cvp0, cvp1);
    throw std::runtime_error("Unknown CCDMethod");
}

template<typename _Real>
void IPCObjectiveTerm<_Real>::initialBarrierStiffness(double weight, const Eigen::VectorXd &primaryPotentialGradient) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.initialBarrierStiffness");
    if (!useAdaptiveBarrier) return;

    VXd dB_dCV = contactPotentialGradient();
    const EO &o = object();
    double avgMass = o.getMassDensity() * o.volume() / (m_combinedCollisionMesh->numCombinedNodes());

    // Compute the gradient of the "primary potential" with respect to the combined collision mesh vertices.
    // Note that the obstacle vertices do not influence the primary potential,
    // so this vector will be padded with zeros at the end.
    size_t numObstacleVars = m_combinedCollisionMesh->numObstaclesVertices()   * m_N;
    size_t numEOVars       = m_combinedCollisionMesh->numEOCollisionVertices() * m_N;
    size_t numCMVars       = numEOVars + numObstacleVars;
    if (size_t(dB_dCV.size()) != numCMVars) throw std::runtime_error("Unexpected dB_dCV size");
    VXd dE_dCV(numCMVars);
    MXd dE_dEOCM = m_combinedCollisionMesh->getElasticObjectCollisionMesh().extractVectorField(primaryPotentialGradient); // Primary potential gradient wrt. each collision mesh vertex
    Eigen::Map<MXdRowMajor>(dE_dCV.data(), dE_dEOCM.rows(), m_N) = dE_dEOCM; // Row major to ensure proper component ordering!
    // The obstacle vertices do not influence the primary potential
    dE_dCV.tail(numObstacleVars).setZero();

    m_k = m_ipcWrapper->initial_barrier_stiffness(m_collisionVertexPositions, avgMass, weight * dE_dCV, dB_dCV, weight);
}

template<typename _Real>
void IPCObjectiveTerm<_Real>::updateBarrierStiffness() {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.updateBarrierStiffness");
    m_k = m_ipcWrapper->update_barrier_stiffness(m_collisionVertexPositions, m_k);
}

template<typename _Real>
_Real IPCObjectiveTerm<_Real>::CCDFeasibleStepLength(const MXd &x0, const MXd &x1) const {
    return m_ipcWrapper->compute_collision_tightInclusion_stepsize(x0, x1);
}

template <typename _Real>
_Real IPCObjectiveTerm<_Real>::CCDStepSize(const VXd &x0, const VXd &x1) const
{
    return m_ipcWrapper->compute_collision_tightInclusion_stepsize(
                                m_combinedCollisionMesh->vertexPositionsForPolyfemVars(x0),
                                m_combinedCollisionMesh->vertexPositionsForPolyfemVars(x1));
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
////////////////////////////////////////////////////////////////////////////////
template struct IPCObjectiveTerm<double>;
#if MESHFEM_BIND_LONG_DOUBLE
    template struct IPCObjectiveTerm<long double>;
#endif

#endif // MESHFEM_WITH_IPC_TOOLKIT
