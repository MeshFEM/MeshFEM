#ifndef IPCWRAPPER_HH
#define IPCWRAPPER_HH

#include <MeshFEM/ElasticObject.hh>
#include "Obstacle.hh"

using ObstaclesCollection = std::vector<std::shared_ptr<Obstacle>>;

// This CombinedCollisionMesh appends the obstacles to the ElasticObject's
// CollisionMesh.
template<typename _Real>
struct CombinedCollisionMesh {
    using Real = _Real;
    using VXi  = Eigen::VectorXi;
    using EO   = ElasticObject<Real>;
    using MXi  = Eigen::MatrixXi;
    using MXd  = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using MXdRowMajor = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VXd  = Eigen::VectorXd;

    using VMaxd = VecMaxN_T<Real, 3>;

    CombinedCollisionMesh(const EO &eo, const ObstaclesCollection &obsts) : m_obsts(obsts) {
        m_eocm = eo.getCollisionMesh();
        N = m_eocm.N;
        numEONodes = eo.numVars() / N;
        fullModelBlockVars = m_eocm.fullModelBlockVars;
        bbox = m_eocm.bbox;
        edges = m_eocm.edges;
        if (N == 3) faces = m_eocm.faces;
        m_obstaclesVertices = MXd();

        // Counting pass
        size_t nov = 0, noe = 0, nof = 0;
        for (auto &o : m_obsts) {
            if (o->dimension() != N) throw std::runtime_error("Obstacle vertex dimension mismatch");
            m_numObstsVertices.push_back(o->numVertices());
            nov += o->numVertices();
            noe += o->numEdges();
            if (N == 3) nof += o->numFaces();
        }

        m_obstaclesVertices.resize(nov, N);
        edges  .conservativeResize(edges.rows() + noe, 2);
        faces  .conservativeResize(faces.rows() + nof, 3);

        // Appending pass
        size_t vtxOffset = numEOCollisionVertices(); // Index of the first vertex of the obstacle after appending.
        size_t ovBack = 0, eBack = m_eocm.edges.rows(), fBack = m_eocm.faces.rows();
        for (auto &o : m_obsts) { m_obstaclesVertices.middleRows(ovBack, o->numVertices()) = o->getVertices();
            ovBack += o->numVertices();

            edges.middleRows(eBack, o->numEdges()) = o->getEdges().array() + vtxOffset;
            eBack += o->numEdges();

            if (N == 3) {
                faces.middleRows(fBack, o->numFaces()) = o->getFaces().array() + vtxOffset;
                fBack += o->numFaces();
            }

            bbox.unionBox(o->getBBox());
            vtxOffset += o->numVertices();
        }
        nodeForCollisionMeshVertex = getNodeForCombinedCollisionMeshVertex();
    }

    const typename EO::CollisionMesh &getElasticObjectCollisionMesh() const { return m_eocm; }

    // Number of ElasticObject vertices which will be used for finite element computations
    size_t numEOCollisionVertices() const { return m_eocm.numCollisionVertices(); }
    // Number of obstacle vertices
    size_t numObstaclesVertices() const { return m_obstaclesVertices.rows(); }

    bool isObstacleVertex(size_t vi) const { return vi >= numEOCollisionVertices(); }

    size_t numCombinedCollisionVertices() const { return numEOCollisionVertices() + numObstaclesVertices(); }
    size_t numCombinedNodes() const { return numEONodes + numObstaclesVertices(); }
    // Concatenate nodeForCollisionMeshVertex with vector of -1 with the number of
    // obstacle vertices
    VXi getNodeForCombinedCollisionMeshVertex() const {
        VXi result(numCombinedCollisionVertices());
        result << m_eocm.nodeForCollisionMeshVertex,
                  VXi::Constant(numObstaclesVertices(), -1);
        return result;
    }

    // Get the position of the collision mesh vertices from the simulation
    // `vars` and the passed obstacle vertex positions `obstVars`.
    MXd extractPositions(const Eigen::Ref<const VXd> &vars, const MXd &obstPositions) {
        if (size_t(vars.size()) != N * numEONodes)                  throw std::runtime_error("Unexpected vars size.");
        if (size_t(obstPositions.rows()) != numObstaclesVertices()) throw std::runtime_error("Unexpected obstacle vertex positions size.");
        const size_t nccv = numCombinedCollisionVertices();
        MXd result(nccv, N);
        result << m_eocm.extractVectorField(vars),
                  obstPositions;
        return result;
    }

    MXd vertexPositionsForVars(const VXd &vars) { return extractPositions(vars, m_obstaclesVertices); }

    // PolyFEM includes the obstacle vertex positions at the end of the vars vector.
    MXd vertexPositionsForPolyfemVars(const VXd &vars) {
        if (size_t(vars.size()) != N * numCombinedNodes()) throw std::runtime_error("Unexpected PolyFEM vars size.");
        return extractPositions(vars.head(N * numEONodes),
                                Eigen::Map<const MXdRowMajor>(vars.data() + N * numEONodes, numObstaclesVertices(), N));
    }

    // Compute the new bounding box for combined collision mesh
    Real getBboxDiagonal() const { return bbox.diagonal(); }

    // Change the position of obstacle in its linear trajectory or move obstacle with time t
    void updateObstaclePosition(double t) {
        m_obstaclesVertices.setZero();
        size_t cnt = 0;
        for (const auto &obst : m_obsts) {
            obst->updatePositionForTime(t);
            m_obstaclesVertices.middleRows(cnt, obst->numVertices()) = obst->getVertices();
            cnt += obst->numVertices();
        }
    }

    // Obstacle vertex positions
    const MXd &getObstaclesVertices() const { return m_obstaclesVertices; }

    // Set the net force generated by contact for each obstacle
    void setObstaclesForce(const Eigen::Ref<const VXd> &grad_vars) {
        size_t cnt = 0;
        for (size_t i = 0; i < m_numObstsVertices.size(); i++){
            m_obsts[i]->setForce(grad_vars.segment(cnt * N, m_numObstsVertices[i]*N));
            cnt += m_numObstsVertices[i];
        }
    }

    size_t fullModelBlockVars;
    VXi nodeForCollisionMeshVertex;
    MXi edges, faces;
    size_t N;
    BBox<VMaxd> bbox;
    size_t numEONodes;

private:
    typename EO::CollisionMesh m_eocm; // Elastic Object Collision Mesh
    ObstaclesCollection m_obsts; // Vector of Obstacles
    std::vector<size_t> m_numObstsVertices;
    MXd m_obstaclesVertices;
};

// Forward declaration of struct holding all IPC state and functionality that
// requires IPC headers
struct IPCWrapperBase {
    using MXd = Eigen::MatrixXd;
    using VXd = Eigen::VectorXd;

    virtual double initial_barrier_stiffness(const MXd &collisionVertexPositions, double bboxDiagonal, double mass, const VXd &primaryGradient, const VXd &contactPotentialGradient, double weight) = 0;
    virtual double  update_barrier_stiffness(const MXd &collisionVertexPositions, double k, double bboxDiagonal) = 0;

    virtual void build_collision_constraints(const MXd &collisionVertexPositions) = 0;
    virtual double compute_collision_tightInclusion_stepsize(const MXd &collisionVertexPositions, const MXd &steppedCollisionVertexPositions) const = 0;
    virtual double compute_collision_cfl_stepsize(const MXd &collisionVertexPositions, const MXd &steppedCollisionVertexPositions) const = 0;

    virtual double compute_potential(const MXd &collisionVertexPositions) const = 0;
    virtual VXd compute_potential_gradient(const MXd &cvPositions) const = 0;
    virtual void              hessian(NewtonHessian &H, const MXd &cvPositions, const Eigen::VectorXi &blockVarForCollisionMeshVertex, double k, bool projectionMask) const = 0;

    virtual std::unique_ptr<BlockCSCHessianBase> block_hessian_sparsity_pattern(const Eigen::VectorXi &blockVarForCollisionMeshVertex) const = 0;
    virtual size_t detect_contact_set_change(const BlockCSCHessianBase &block_Hsp, const Eigen::VectorXi &blockVarForCollisionMeshVertex) const = 0;

    virtual size_t numCollisionConstraints() const = 0;
    virtual void resetCandidateCache() = 0;

    virtual ~IPCWrapperBase() { }

    double dhat = 0; // Barrier distance
    double maxBarrierStiffness = 0;
    double prevMinDistanceSq = 0; // Previous minimum squared distance between non-adjacent collision mesh primitives

    double ccdTol = 1.0e-6; // CCD tolerance required in IPCWrapper and can set by user
};

std::unique_ptr<IPCWrapperBase> make_ipc_wrapper(const CombinedCollisionMesh<Real> &cm, const Eigen::MatrixXd &collisionVertexPositions);

#endif /* end of include guard: IPCWRAPPER_HH */
