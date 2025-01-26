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
        // {
        //     static size_t counter = 0;
        //     if (counter == 0) {
        //         std::ofstream("cm_edges.txt") << collisionMesh.edges();
        //         std::ofstream("cm_faces.txt") << collisionMesh.faces();
        //     }
        //     std::string name = "debug_ccd_" + std::to_string(counter++);
        //     std::ofstream(name + "_x0.txt") << std::setprecision(19) << collisionVertexPositions;
        //     std::ofstream(name + "_x1.txt") << std::setprecision(19) << steppedCollisionVertexPositions;
        // }
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
        size_t max_iteration = 1e6;
        double alpha = candidates.compute_collision_free_stepsize(
            collisionMesh, collisionVertexPositions, steppedCollisionVertexPositions, /* dmin = */ dmin, /* tolerance = */ ccdTol, /* max_iterations = */ max_iteration);
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

    virtual double initial_barrier_stiffness(const MXd &collisionVertexPositions, double mass, const VXd &primaryGradient, const VXd &contactPotentialGradient, double dtSq) override {
        prevMinDistanceSq = collisionConstraints.compute_minimum_distance(collisionMesh, collisionVertexPositions);
        ipc::BarrierPotential barrierPotential(dhat);
        double bboxDiagonal = computeBboxDiagonal(collisionVertexPositions);
        // std::cout.precision(20);
        // std::cout << "bbox diagonal: " << bboxDiagonal << std::endl;
        // std::cout << "avgmass: " << mass << std::endl;
        // std::cout << "grad_energy: " << dtSq * primaryGradient.norm() << std::endl;
        // std::cout << "grad_barrier: " << contactPotentialGradient.norm() << std::endl;
        // std::cout << "num collision constrains: " << numCollisionConstraints() << std::endl;
        
        double barrierStiffness = ipc::initial_barrier_stiffness(bboxDiagonal, barrierPotential.barrier(), dhat, mass, dtSq * primaryGradient, contactPotentialGradient, maxBarrierStiffness);
        // std::cout << "barrierStiffness: " << barrierStiffness << std::endl;
        barrierStiffness /= dtSq;
        maxBarrierStiffness /= dtSq;
        // std::cout << "barrierStiffness wo weight: " << barrierStiffness << std::endl;
        return barrierStiffness;
    }

    virtual double update_barrier_stiffness(const MXd &collisionVertexPositions, double k) override {
        double minDistanceSq = collisionConstraints.compute_minimum_distance(collisionMesh, collisionVertexPositions);
        double bboxDiagonal = computeBboxDiagonal(collisionVertexPositions);
        double k_new = ipc::update_barrier_stiffness(prevMinDistanceSq, minDistanceSq, maxBarrierStiffness, k, bboxDiagonal);
        prevMinDistanceSq = minDistanceSq;
        return k_new;
    }

    // compute bounding box diagonal 
    double computeBboxDiagonal(const MXd &collisionVertexPositions) {
        return (collisionVertexPositions.colwise().maxCoeff() - collisionVertexPositions.colwise().minCoeff()).norm();
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
        struct CustomHEAD {
            CustomHEAD(const IPCWrapper &ipc, bool pmask, const MXd &cvp, Real stiffness, int ci, const Eigen::VectorXi &bvfcmv) : evars(ipc.constraintStencil(ci)) {
                ipc::BarrierPotential barrierPotential(ipc.dhat);
                H_e = stiffness * barrierPotential.hessian(ipc.collisionConstraints[ci],
                                                    ipc.collisionConstraints[ci].dof(cvp, ipc.collisionMesh.edges(), ipc.collisionMesh.faces()), pmask);

                // We need to remove from the stencil variables corresponding to the obstacles.
                // These are variables for which `blockVarForCollisionMeshVertex` is -1.
                // We also need to skip over those blocks of the per-element Hessian,
                // which we do with the `local_block_for_evar` index remapping array.
                size_t back = 0;
                local_block_for_evar.resize(evars.numVars);
                for (size_t v = 0; v < evars.numVars; ++v) {
                    int bvar = bvfcmv[evars[v]];
                    if (bvar == -1) continue;
                    local_block_for_evar[back] = v;
                    evars[back++] = bvar;
                }
                local_block_for_evar.resize(back);
                evars.resize(back);
            }

            using MNd = Eigen::Matrix<Real, N, N>;
            MNd block(size_t a, size_t b, size_t /* bsa */, size_t /* bsb */) const { return block(a, b); }
            MNd block(size_t a, size_t b) const { return H_e.template block<N, N>(N * local_block_for_evar[a / N], N * local_block_for_evar[b / N]); } // (a, b) are scalar offsets...

            ipc::MatrixMax12d H_e;

            StencilMembers local_block_for_evar;
            StencilMembers evars;
        };
        m_assembler.assembleHessian(H, collisionConstraints.size(), [&](size_t ci) { return CustomHEAD(*this, projectionMask, cvPositions, k, ci, blockVarForCollisionMeshVertex); });
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
