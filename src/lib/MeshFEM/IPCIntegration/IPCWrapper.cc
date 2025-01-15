#ifdef MESHFEM_WITH_IPC_TOOLKIT
#include "IPCWrapper.hh"

#include <ipc/ipc.hpp>
#include <ipc/collisions/collisions.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/potentials/barrier_potential.hpp>

#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/ParallelAssembly.hh>

#include <MeshFEM_export.h>

// The dimension-specific parts of IPCWrapper.
template<size_t N>
struct IPCWrapper : public IPCWrapperBase {
    IPCWrapper(const MXd &collisionVertexPositions, const CombinedCollisionMesh<Real> &cm)
        : m_assembler(cm.fullModelBlockVars) {
        collisionMesh = ipc::CollisionMesh(collisionVertexPositions, cm.edges, cm.faces);
        collisionMesh.can_collide = [&cm](size_t vi, size_t vj) {
            // Obstacles cannot collide
            return !(cm.isObstacleVertex(vi) && cm.isObstacleVertex(vj));
        };
    }

    virtual void build_collision_constraints(const MXd &collisionVertexPositions) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("build_collision_constraints");
        if (candidateCache) collisionConstraints.build(*candidateCache, collisionMesh, collisionVertexPositions, dhat);
        else                collisionConstraints.build(                 collisionMesh, collisionVertexPositions, dhat);
    }

    virtual double compute_potential(const MXd &collisionVertexPositions) const override {
        ipc::BarrierPotential barrierPotential(dhat);
        return barrierPotential(collisionConstraints, collisionMesh, collisionVertexPositions);
    }

    virtual double compute_collision_tightInclusion_stepsize(const MXd &collisionVertexPositions, const MXd &steppedCollisionVertexPositions) const override {
#if 0
        const auto ccd_begin = std::chrono::steady_clock::now();
        double alpha = ipc::compute_collision_free_stepsize(collisionMesh, collisionVertexPositions, steppedCollisionVertexPositions);
        const auto ccd_end = std::chrono::steady_clock::now();
        std::cout << "CCD duration: " << std::chrono::duration<double>(ccd_end - ccd_begin).count() << " seconds." << std::endl;
        // const auto ccd_begin = std::chrono::steady_clock::now();
#else
        BENCHMARK_START_TIMER_SECTION("candidates.build");
        candidateCache = std::make_unique<ipc::Candidates>();
        ipc::Candidates &candidates = *candidateCache;
		candidates.build(
			collisionMesh,
			collisionVertexPositions,
			steppedCollisionVertexPositions,
			/* inflation_radius = */ dhat / 2, ipc::BroadPhaseMethod::HASH_GRID);
        BENCHMARK_STOP_TIMER_SECTION("candidates.build");
        // const auto candidate_end = std::chrono::steady_clock::now();

#if 0
        double alpha = candidates.compute_collision_free_stepsize(
            collisionMesh, collisionVertexPositions, steppedCollisionVertexPositions, /* dmin = */ 1e-6, /* tolerance = */ 1e-6, /* max_iterations = */ 1e6);
#else
        BENCHMARK_START_TIMER_SECTION("compute_collision_free_stepsize");
        double dmin = 0.0;
        double ccd_tolerance = 2e-8;
        size_t max_iteration = 1e6;
        std::cout << "candidates.compute_collision_free_stepsize with candidate size " << candidates.size() << " and step length " << (steppedCollisionVertexPositions - collisionVertexPositions).norm() << std::endl;
        double alpha = candidates.compute_collision_free_stepsize(
            collisionMesh, collisionVertexPositions, steppedCollisionVertexPositions, /* dmin = */ dmin, /* tolerance = */ ccd_tolerance, /* max_iterations = */ max_iteration); 
        BENCHMARK_STOP_TIMER_SECTION("compute_collision_free_stepsize");
#endif

        // const auto ccd_end = std::chrono::steady_clock::now();
        // double candidate_time = std::chrono::duration<double>(candidate_end - ccd_begin).count();
        // double   compute_time = std::chrono::duration<double>(ccd_end - candidate_end).count();

        // std::cout << "CCD duration " << candidate_time + compute_time << ": " << candidate_time << ", " << compute_time << std::endl;
#endif
        return alpha;
    }

    virtual double compute_collision_cfl_stepsize(const MXd &collisionVertexPositions, const MXd &steppedCollisionVertexPositions) const override {
        candidateCache = std::make_unique<ipc::Candidates>();
        ipc::Candidates &candidates = *candidateCache;
        candidates.build(collisionMesh, collisionVertexPositions, steppedCollisionVertexPositions, dhat / 2);
        return candidates.compute_cfl_stepsize(collisionMesh, collisionVertexPositions, steppedCollisionVertexPositions, dhat);
    }

    virtual double initial_barrier_stiffness(const MXd &collisionVertexPositions, double bboxDiagonal, double mass, const VXd &primaryGradient, const VXd &contactPotentialGradient, double weight) override {
        prevMinDistanceSq = collisionConstraints.compute_minimum_distance(collisionMesh, collisionVertexPositions);
        ipc::BarrierPotential barrierPotential(dhat);
        double barrierStiffness = ipc::initial_barrier_stiffness(bboxDiagonal, barrierPotential.barrier(), dhat, mass, primaryGradient, contactPotentialGradient, maxBarrierStiffness);
        barrierStiffness /= weight;
        maxBarrierStiffness /= weight;
        return barrierStiffness;
    }

    virtual double update_barrier_stiffness(const MXd &collisionVertexPositions, double k, double bboxDiagonal) override {
        double minDistanceSq = collisionConstraints.compute_minimum_distance(collisionMesh, collisionVertexPositions);
        double k_new = ipc::update_barrier_stiffness(prevMinDistanceSq, minDistanceSq, maxBarrierStiffness, k, bboxDiagonal);
        prevMinDistanceSq = minDistanceSq;
        return k_new;
    }

    virtual VXd compute_potential_gradient(const MXd &cvPositions) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("IPC.compute_potential_gradient");
        ipc::BarrierPotential barrierPotential(dhat);
#if 1
        VXd result(VXd::Zero(cvPositions.size()));
        m_assembler.assembleGradient(result, collisionConstraints.size(),
                [&](size_t ci) { return barrierPotential.gradient(collisionConstraints[ci], collisionConstraints[ci].dof(cvPositions, collisionMesh.edges(), collisionMesh.faces())); },
                [&](size_t ci) { return constraintStencil(ci); });
        return result;
#else
        return barrierPotential.gradient(collisionConstraints, collisionMesh, cvPositions);
#endif
    }

    using StencilMembers = ElementBlockVarsWithSizeRange<1, 4>;

    // Get the vertices *within the collision mesh* participating in the constraint.
    StencilMembers constraintStencil(size_t ci) const {
        const auto &c = collisionConstraints[ci];
        std::array<long, 4> vertex_ids = c.vertex_ids(collisionMesh.edges(), collisionMesh.faces());
        StencilMembers result(c.num_vertices());
        for (size_t i = 0; i < result.numVars; ++i) result[i] = vertex_ids[i];
        return result;
    }

    // Get the block variables *for the full problem* participating in the constraint.
    StencilMembers constraintStencil(size_t ci, const Eigen::VectorXi &blockVarForCollisionMeshVertex) const {
        const auto &c = collisionConstraints[ci];
        const size_t nv = c.num_vertices();
        // Get the collision mesh vertices participating in constraint `c`.
        std::array<long, 4> vertex_ids = c.vertex_ids(collisionMesh.edges(), collisionMesh.faces());
        // Convert to block variables of the global system.
        StencilMembers result(nv);
        size_t back = 0;
        for (long vi : vertex_ids) {
            if (vi < 0) continue;
            int bvar = blockVarForCollisionMeshVertex[vi];
            if (bvar != -1) result[back++] = bvar;
        }
        result.resize(back);
        return result;
    }

    virtual void hessian(NewtonHessian &H, const MXd &cvPositions, const Eigen::VectorXi &blockVarForCollisionMeshVertex, double k, bool projectionMask) const override {
        m_assembler.assembleHessian(H, collisionConstraints.size(), [&](size_t ci) {
                ipc::BarrierPotential barrierPotential(dhat);
                return (k * barrierPotential.hessian(collisionConstraints[ci], collisionConstraints[ci].dof(cvPositions, collisionMesh.edges(), collisionMesh.faces()), projectionMask)).eval();
            }, [this, &blockVarForCollisionMeshVertex](size_t ci) { return constraintStencil(ci, blockVarForCollisionMeshVertex); });
    }

    virtual std::unique_ptr<BlockCSCHessianBase> block_hessian_sparsity_pattern(const Eigen::VectorXi &blockVarForCollisionMeshVertex) const override {
        return m_assembler.blockSparsityPattern(collisionConstraints.size(), [this, &blockVarForCollisionMeshVertex](size_t ci) { return constraintStencil(ci, blockVarForCollisionMeshVertex); });
    }

    // Returns `NEW_ENTRIES` (size_t max) if even a single new entry becomes nonzero in the contact block sparsity pattern;
    // otherwise returns the number of blocks that have disappeared (if any).
    virtual size_t detect_contact_set_change(const BlockCSCHessianBase &block_Hsp, const Eigen::VectorXi &blockVarForCollisionMeshVertex) const override {
        return m_assembler.detectChangedEntries(block_Hsp,
                collisionConstraints.size(),
                [&](size_t ci) { return constraintStencil(ci, blockVarForCollisionMeshVertex); });
    }

    virtual size_t numCollisionConstraints() const override { return collisionConstraints.size(); }
    virtual void resetCandidateCache() override { candidateCache.reset(); }

    ipc::CollisionMesh collisionMesh;
    ipc::Collisions collisionConstraints;
    mutable std::unique_ptr<ipc::Candidates> candidateCache;

private:
    SystemAssembler<N> m_assembler;
};

std::unique_ptr<IPCWrapperBase> make_ipc_wrapper(const CombinedCollisionMesh<Real> &cm, const Eigen::MatrixXd &collisionVertexPositions) {
    const size_t N = cm.N;
    if (N == 2) return std::make_unique<IPCWrapper<2>>(collisionVertexPositions, cm);
    if (N == 3) return std::make_unique<IPCWrapper<3>>(collisionVertexPositions, cm);
    throw std::runtime_error("Unexpected N");
}

#endif // MESHFEM_WITH_IPC_TOOLKIT
