#ifndef IPCObject_HH
#define IPCObject_HH

#include <MeshFEMCore/GlobalBenchmark.hh>
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>
#include <MeshFEM/ElasticObject.hh>
#include <MeshFEM/DynamicSimulator.hh>


#include <Eigen/Sparse>
#include <MeshFEM_export.h>
#include "IPCWrapper.hh"

namespace MeshFEM {

template<typename _Real>
struct MESHFEM_EXPORT IPCObjectiveTerm : public NewtonObjectiveTerm, public TimestepLimiter {
    enum class CCDMethod { TightInclusion, CFL };
    using Real = _Real;

    using MXd = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using MXdRowMajor = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MXi = Eigen::MatrixXi;

    IPCObjectiveTerm(std::shared_ptr<NewtonVarsBase> vars, CollisionMesh cm, const ObstaclesCollection &obsts = ObstaclesCollection(), _Real dhat = 0);

    virtual void varsUpdated() override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.varsUpdated");
        // Update IPC collision constraints and contact hessian sparsity pattern
        m_collisionVertexPositions = m_combinedCollisionMesh->vertexPositionsForVars(getVars());
        m_buildCollisionConstraints();
    }

    const VXd getVars() const { return this->getNVars().getVars().template cast<double>(); }

    virtual Real objective() const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.energy");
        return m_k * contactPotentialEnergy();
    }

    virtual void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.accumulateGradient");
        // Add contact potential gradient computed by IPC
        const VXd &dB_dCV = contactPotentialGradient();
        // Extract entries from the gradient with respect to each collision mesh vertex.
        for (size_t i = 0; i < m_combinedCollisionMesh->numEOCollisionVertices(); ++i) {
            int bvar = m_combinedCollisionMesh->nodeForCollisionMeshVertex[i];
            if (bvar < 0) continue;
            g.segment(m_N * bvar, m_N) += weight * m_k * dB_dCV.segment(m_N * i, m_N);
        }
        // Store the contact potential gradient related to obstacles points
        m_combinedCollisionMesh->setObstaclesForce(dB_dCV.tail(m_N * m_combinedCollisionMesh->numObstaclesVertices()));
    }

    virtual SparsityUpdateFrequency sparsityUpdateFrequency() const override { return SparsityUpdateFrequency::SOMETIMES; }

    void setBarrierStiffness(Real k) { m_k = k; }
    Real getBarrierStiffness() const { return m_k; }

    void set_dhat(Real dhat) { m_ipcWrapper->dhat = dhat; m_ipcWrapper->resetCandidateCache(); m_buildCollisionConstraints();  }
    Real get_dhat() const { return m_ipcWrapper->dhat; }

    void set_ccdTol(Real ccdTol) { m_ipcWrapper->ccdTol = ccdTol; }
    Real get_ccdTol() const { return m_ipcWrapper->ccdTol; }

    void set_ccdMaxIters(Real ccdMaxIters) { m_ipcWrapper->ccdMaxIters = ccdMaxIters; }
    Real get_ccdMaxIters() const { return m_ipcWrapper->ccdMaxIters; }

    // Get the last attempted Newton step (for debugging the line search)
    const MXd &getCollisionVertexPositions() const { return m_collisionVertexPositions;     }
    const MXi &getCollisionMeshFaces()       const { return m_combinedCollisionMesh->faces; }
    const MXi &getCollisionMeshEdges()       const { return m_combinedCollisionMesh->edges; }

    ////////////////////////////////////////////////////////////////////////////////
    // Definition of IPC methods
    ////////////////////////////////////////////////////////////////////////////////
    Real contactPotentialEnergy() const;
    VXd contactPotentialGradient() const; // Gradient with respect to just the collision mesh vertex positions

    VXd contactGradient(bool includeObstacleVertices = false) const {
        VXd g;
        
        size_t gSize = numVars();
        size_t numObstacleVars = m_combinedCollisionMesh->numObstaclesVertices() * m_N;
        if (includeObstacleVertices) gSize += numObstacleVars;

        g.setZero(gSize);
        const VXd &dB_dCV = contactPotentialGradient();
        // Extract entries from the gradient with respect to each collision mesh vertex.
        for (size_t i = 0; i < m_combinedCollisionMesh->numEOCollisionVertices(); ++i) {
            int bvar = m_combinedCollisionMesh->nodeForCollisionMeshVertex[i];
            if (bvar < 0) continue; // Ignore obstacle vertices.
            g.segment(m_N * bvar, m_N) += dB_dCV.segment(m_N * i, m_N);
        }
        if (includeObstacleVertices) g.tail(numObstacleVars) = dB_dCV.tail(numObstacleVars);
        return g;
    }

    virtual bool detectSparsityPatternChange(const NewtonHessian &oldHsp) const override;

    virtual void accumulateHessian(Real weight, NewtonHessian &result, bool projectionMask = false) const override;
    NewtonHessian hessianSparsityPattern() const override;

    // Determine the maximum collision-free step size.
    Real feasibleStepLength(const VXd &vars, const VXd &step, Real initialAlpha = 1.0, Real currentObjectiveValue = std::numeric_limits<Real>::max()) const override;

    // Adaptive barrier stiffness support
    // Note that `initialBarrierStiffness` needs access to the current primary
    // potential gradient; for a dynamic simulation this must incorporate the
    // inertia term gradient!
    void initialBarrierStiffness(double dtSq, const Eigen::VectorXd &primaryPotentialGradient, double primaryObjectMass) override;
    void updateBarrierStiffness();

    size_t numCollisionConstraints() const;

    // Determine a timestep attentuation factor `alpha` so that the moving
    // obstacle does not collide with the primary object over the time
    // interval [0, alpha * dt].
    virtual Real getTimestepLength(Real t, Real dt) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.AddaptiveTimeStepLength");
        Real alpha;
        VXd vars = getVars();
        // Move Obstacle with t time in linear trajectory
        m_combinedCollisionMesh->updateObstaclePosition(t + dt);
        // Fix primary object and run CCD to detect collision with the moving obstacle
        alpha = feasibleStepLength(vars, VXd::Zero(numVars()));
        if (alpha != 1.0) m_combinedCollisionMesh->updateObstaclePosition(t + alpha*dt);
        m_collisionVertexPositions = m_combinedCollisionMesh->vertexPositionsForVars(vars);
        m_buildCollisionConstraints();
        m_ipcWrapper->resetCandidateCache();
        return alpha;
    }

    void updateObstaclePosition(Real t) {
        m_combinedCollisionMesh->updateObstaclePosition(t);
    }

    // Called at the beginning of each Newton iteration
    virtual void newtonIterationBegan() override { if (useAdaptiveBarrier) updateBarrierStiffness(); }

     // Called at the end of a line search
    virtual void lineSearchTerminated() override {
        m_ipcWrapper->resetCandidateCache();
    }

    Real CCDFeasibleStepLength(const MXd &x0, const MXd &x1) const;

    Real CCDStepSize(const VXd &x0, const VXd &x1) const;

    ~IPCObjectiveTerm();
    
    CCDMethod CCD = CCDMethod::TightInclusion;
    bool useAdaptiveBarrier = true;

    // Sparsity pattern update acceleration:
    // number of blocks that must disappear from the sparsity pattern before it
    // is rebuilt. Note that even if a rebuild is triggered,
    // the symbolic factorization is not necessarily updated
    // because `SparsityLRU` also caches old entries.
    size_t sparsityPatternUpdateThreshold = 0;

protected:
    void m_buildCollisionConstraints();

    std::unique_ptr<CombinedCollisionMesh<Real>> m_combinedCollisionMesh;
    std::unique_ptr<IPCWrapperBase> m_ipcWrapper;

    // Embedding dimension
    size_t m_N;

    ////////////////////////////////////////////////////////////////////////////////
    // User configuration of IPC barrier
    ////////////////////////////////////////////////////////////////////////////////
    Real m_k = 1.0;                   // IPC Barrier Stiffness
    MXd  m_collisionVertexPositions;  // Cached collision vertex positions
};

} // namespace MeshFEM

#endif
