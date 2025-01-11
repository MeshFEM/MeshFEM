#include "IPCObjectiveTerm.hh"
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/SystemAssembler.hh>

#if MESHFEM_WITH_IPC_TOOLKIT
#include <chrono>

template<typename _Real>
IPCObjectiveTerm<_Real>::IPCObjectiveTerm(std::shared_ptr<EO> eo, const ObstaclesCollection &obsts)
    : m_obj(eo), m_k(1.0e6)
{
    m_combinedCollisionMesh = std::make_unique<CombinedCollisionMesh<Real>>(*eo, obsts);
    m_collisionVertexPositions = m_combinedCollisionMesh->mergeCombinedCollisionFields(getVars(), m_combinedCollisionMesh->getObstaclesVertices());
    
    m_N = m_combinedCollisionMesh->N;
    m_ipcWrapper = make_ipc_wrapper(*m_combinedCollisionMesh, m_collisionVertexPositions);

    if (!m_ipcWrapper) throw std::runtime_error("Unsupported dimension " + std::to_string(m_N));
    m_ipcWrapper->dhat = m_combinedCollisionMesh->getBboxDiagonal() * 1e-3;//std::pow(m_combinedCollisionMesh.bboxDiagonal,2)*1e-4 / 2.0 ;
    
    m_primaryBlockSparsity = eo->hessianBlockSparsityPattern();
    // m_hessianSparsity = m_ipcWrapper->block_hessian_sparsity_pattern_to_scalar(m_primaryBlockSparsity);
    // m_hessianSparsity = m_ipcWrapper->block_hessian_sparsity_pattern_to_scalar(m_contactBlockSparsity.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE));
    m_buildCollisionConstraints();

    if (!obsts.size()) setAdaptiveTimestep(false);
    if (useAdaptiveBarrier) initialBarrierStiffness(1.0);
}

template<typename _Real>
IPCObjectiveTerm<_Real>::~IPCObjectiveTerm() { }

template<typename _Real>
size_t IPCObjectiveTerm<_Real>::numCollisionConstraints() const { return m_ipcWrapper->numCollisionConstraints(); }

template<typename _Real>
void IPCObjectiveTerm<_Real>::m_buildCollisionConstraints() {
    m_ipcWrapper->build_collision_constraints(m_collisionVertexPositions);
    m_updateContactHessianSparsityPattern();
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
void IPCObjectiveTerm<_Real>::m_updateContactHessianSparsityPattern() {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.m_updateContactHessianSparsityPattern");
    size_t changed = m_ipcWrapper->detect_contact_set_change(m_contactBlockSparsity, m_combinedCollisionMesh->nodeForCollisionMeshVertex);
    if (changed < m_sparsityPatternUpdateThreshold) { sparsityPatternChanged = false; return; }

    m_contactBlockSparsity = m_ipcWrapper->block_hessian_sparsity_pattern(m_combinedCollisionMesh->nodeForCollisionMeshVertex);
    // {
    //     static size_t counter = 0;
    //     m_contactBlockSparsity.Ax.resize(m_contactBlockSparsity.nz); // Block sparsity patterns are not filled in by default.
    //     m_contactBlockSparsity.dumpBinary("contact_block_sparsity_" + std::to_string(counter++) + ".bin");
    // }

    m_blockSparsity = m_primaryBlockSparsity;
    m_blockSparsity.addWithDistinctSparsityPattern</* IgnoreValues = */ true>(m_contactBlockSparsity);

    // The elastic object needs the new block sparsity pattern to accelerate Hessian assembly.
    m_obj->setBlockHsp(m_blockSparsity); // TODO: remove this hack once we've
                                         // implemented block Hessian assembly at
                                         // the NewtonMultiobjectiveProblem level.

    m_hessianSparsity = m_ipcWrapper->block_hessian_sparsity_pattern_to_scalar(m_contactBlockSparsity);
    // {
    //     static size_t counter = 0;
    //     m_hessianSparsity.dumpBinary("hessian_sparsity_pattern_IPCOT_" + std::to_string(counter++) + ".bin");
    // }

    sparsityPatternChanged = true;
}

template<typename _Real>
void IPCObjectiveTerm<_Real>::contactHessian(Real weight, SuiteSparseMatrix &H, bool projectionMask) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("contactHessian");
    m_ipcWrapper->hessian(H, m_collisionVertexPositions, m_combinedCollisionMesh->nodeForCollisionMeshVertex, weight * m_k, projectionMask);
}

template<typename _Real>
_Real IPCObjectiveTerm<_Real>::customFeasibleStepLength(const VXd &vars, const VXd &step) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.feasibleStepLength");
    if (CCD == CCDMethod::CFL)
        return m_ipcWrapper->compute_collision_cfl_stepsize(m_collisionVertexPositions, m_combinedCollisionMesh->vertexPositionsForVars(vars + step));
    if (CCD == CCDMethod::TightInclusion){
        return m_ipcWrapper->compute_collision_tightInclusion_stepsize(m_collisionVertexPositions, m_combinedCollisionMesh->vertexPositionsForVars(vars + step));
    }
    throw std::runtime_error("Unimplemented");
}

template<typename _Real>
void IPCObjectiveTerm<_Real>::initialBarrierStiffness(double weight) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.initialBarrierStiffness");
    if (useAdaptiveBarrier){
        VXd dB_dCV = contactPotentialGradient();
        VXd dE_dCV(dB_dCV.size());
        const EO &o = object();
        double avgMass = o.rho * o.volume() / (m_combinedCollisionMesh->numCombinedVertices());
        Eigen::Map<MXd>(dE_dCV.data(), m_combinedCollisionMesh->numCombinedCollisionVertices(), m_N) = m_combinedCollisionMesh->mergeCombinedCollisionFields(o.gradient(true), MXd::Zero(m_combinedCollisionMesh->numObstaclesVertices(), m_N));
        m_k = m_ipcWrapper->initial_barrier_stiffness(m_collisionVertexPositions, m_combinedCollisionMesh->getBboxDiagonal(), avgMass,
                                                    dE_dCV, dB_dCV, weight);
    }
}

template<typename _Real>
void IPCObjectiveTerm<_Real>::updateBarrierStiffness() {
    BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.updateBarrierStiffness");
    m_k = m_ipcWrapper->update_barrier_stiffness(m_collisionVertexPositions, m_k, m_combinedCollisionMesh->getBboxDiagonal());
}

template<typename _Real>
_Real IPCObjectiveTerm<_Real>::CCDFeasibleStepLength(const MXd &x0, const MXd &x1) const {
    return m_ipcWrapper->compute_collision_tightInclusion_stepsize(x0, x1);
}

template <typename _Real>
_Real IPCObjectiveTerm<_Real>::CCDStepSize(const VXd &x0, const VXd &x1) const
{
    return m_ipcWrapper->compute_collision_tightInclusion_stepsize(m_combinedCollisionMesh->getCombinedCollisionFields(x0), m_combinedCollisionMesh->getCombinedCollisionFields(x1));
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
////////////////////////////////////////////////////////////////////////////////
template struct IPCObjectiveTerm<double>;
#if MESHFEM_BIND_LONG_DOUBLE
    template struct IPCObjectiveTerm<long double>;
#endif
#endif
