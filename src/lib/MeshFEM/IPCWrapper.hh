#ifndef IPCWRAPPER_HH
#define IPCWRAPPER_HH

#include "ElasticObject.hh"
#include "Obstacle.hh"

using ObstaclesCollection = std::vector<std::shared_ptr<Obstacle>>;

// This Combined Collision Mesh structure merge ElasticObject CollisionMesh with
// Obstacle. Therefore it is: CombinedCollisionMesh = [EOCollisionMesh, Obstacles]
template<typename _Real>
struct CombinedCollisionMesh {
    using Real = _Real;
    using VXi  = Eigen::VectorXi;
    using EO   = ElasticObject<Real>;
    using MXi  = Eigen::MatrixXi;
    using MXd  = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
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

        size_t numObstVertices = 0;
        // Merge obstacles elements vertices, edges, and faces
        for (auto &obst : m_obsts ){
            m_numObstsVertices.push_back(obst->numVertices());

            m_obstaclesVertices.conservativeResize(numObstVertices + obst->numVertices(), N);
            if (size_t(obst->getVertices().cols()) != N) throw std::runtime_error("Obstacle vertex dimension mismatch");
            m_obstaclesVertices.bottomRows(obst->numVertices()) = obst->getVertices();

            MXi obstEdge = obst->getEdges();
            edges.conservativeResize(edges.rows() + obstEdge.rows(), edges.cols());
            edges.bottomRows(obstEdge.rows()) = obstEdge.array() + numEOCollisionVertices() + numObstVertices;

            if (N == 3) {
                MXi obstFaces = obst->getFaces();
                faces.conservativeResize(faces.rows() + obstFaces.rows(), faces.cols());
                faces.bottomRows(obstFaces.rows()) = obstFaces.array() + numEOCollisionVertices() + numObstVertices;
            }

            bbox.unionBox(obst->getBBox());
            numObstVertices += obst->numVertices();
        }
        nodeForCollisionMeshVertex = getNodeForCombinedCollisionMesh();

    }

    // Number of ElasticObject vertices which will be used for finite element computations
    size_t numEOCollisionVertices() const { return m_eocm.numCollisionVertices(); }
    // Number of obstacle vertices
    size_t numObstaclesVertices() const { return m_obstaclesVertices.rows(); }

    size_t numCombinedCollisionVertices() const { return numEOCollisionVertices() + numObstaclesVertices(); }
    size_t numCombinedVertices() const {return numEONodes + numObstaclesVertices(); }
    // Concatenate nodeForCollisionMeshVertex with vector of -1 with the number of
    // obstacle vertices
    VXi getNodeForCombinedCollisionMesh() const {
        VXi result;
        result.setConstant(numCombinedCollisionVertices(), -1);
        result.head(numEOCollisionVertices()) = m_eocm.nodeForCollisionMeshVertex;
        return result;
    }

    // This method extract fields from ElasticObject collision mesh and
    // merge them with the corresponding field for obstacle vertices
    MXd mergeCombinedCollisionFields(const VXd &vars, const MXd &obstVars) {
        const size_t nccv = numCombinedCollisionVertices();
        const size_t ncv  = numEOCollisionVertices();
        MXd result(nccv, N);
        // Extract field from ElasticObject collision mesh
        result.topRows(ncv) = m_eocm.getCollisionFields(vars);
        // If Obstacle exists, merge them with obstacle field
        if (obstVars.rows()) result.bottomRows(numObstaclesVertices()) = obstVars;
        return result;
    }

    MXd getCombinedCollisionFields(const VXd &vars) {
        const size_t nccv = numCombinedCollisionVertices();
        const size_t ncv  = numEOCollisionVertices();
        MXd result(nccv, N);
        // Extract field from ElasticObject collision mesh
        result.topRows(ncv) = m_eocm.getCollisionFields(vars.head(numEONodes * N));
        // If Obstacle exists, merge them with obstacle field
        // VXd obstVars = vars.tail(numObstaclesVertices() * N);
        if (size_t(vars.size()) > (numEONodes * N)) result.bottomRows(numObstaclesVertices()) = getObstaclesVertices(); //obstVars.reshaped(numObstaclesVertices(),N);
        return result;
    }

    MXd vertexPositionsForVars(const VXd &vars) {
        return mergeCombinedCollisionFields(vars, m_obstaclesVertices);
    }

    // Compute the new bounding box for combined collision mesh
    Real getBboxDiagonal() {
        return bbox.diagonal();
    }

    // Change the position of obstacle in its linear trajectory or move obstacle with time t
    void updateObstaclePosition(double dt) {
        m_obstaclesVertices.setZero();
        size_t cnt = 0;
        for (auto obst: m_obsts){
            obst->moveForward(dt);
            m_obstaclesVertices.middleRows(cnt, obst->numVertices()) = obst->getVertices();
            cnt += obst->numVertices();
        }

    }

    // Obstacle vertices positions
    const MXd &getObstaclesVertices() const { return m_obstaclesVertices; }

    // Set the net force generated by contact for each obstacles
    void setObstaclesForce(const Eigen::Ref<const VXd> &vars) {
        size_t cnt = 0;
        for (size_t i = 0; i < m_numObstsVertices.size(); i++){
            m_obsts[i]->setForce(vars.segment(cnt * N, m_numObstsVertices[i]*N));
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
    virtual double update_barrier_stiffness(const MXd &collisionVertexPositions, double k, double bboxDiagonal) = 0;

    virtual void build_collision_constraints(const MXd &collisionVertexPositions) = 0;
    virtual double compute_collision_tightInclusion_stepsize(const MXd &collisionVertexPositions, const MXd &steppedCollisionVertexPositions) const = 0;
    virtual double compute_collision_cfl_stepsize(const MXd &collisionVertexPositions, const MXd &steppedCollisionVertexPositions) const = 0;

    virtual double compute_potential(const MXd &collisionVertexPositions) const = 0;
    virtual VXd compute_potential_gradient(const MXd &cvPositions) const = 0;
    virtual void              hessian(SuiteSparseMatrix &H, const MXd &cvPositions, const Eigen::VectorXi &blockVarForCollisionMeshVertex, double k, bool projectionMask) const = 0;

    virtual SuiteSparseMatrix block_hessian_sparsity_pattern(const Eigen::VectorXi &blockVarForCollisionMeshVertex) const = 0;
    virtual SuiteSparseMatrix block_hessian_sparsity_pattern_to_scalar(const SuiteSparseMatrix &block_Hsp) const = 0;
    virtual size_t            detect_contact_set_change(const SuiteSparseMatrix &block_Hsp, const Eigen::VectorXi &blockVarForCollisionMeshVertex) const = 0;

    virtual size_t numCollisionConstraints() const = 0;
    virtual void resetCandidateCache() = 0;

    virtual ~IPCWrapperBase() { }

    double dhat = 0; // Barrier distance
    double maxBarrierStiffness = 0;
    double prevMinDistanceSq = 0; // Previous minimum squared distance between non-adjacent collision mesh primitives
};

std::unique_ptr<IPCWrapperBase> make_ipc_wrapper(const CombinedCollisionMesh<Real> &cm, const Eigen::MatrixXd &collisionVertexPositions);

#endif /* end of include guard: IPCWRAPPER_HH */
