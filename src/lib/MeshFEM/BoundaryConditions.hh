////////////////////////////////////////////////////////////////////////////////
// BoundaryConditions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Represents various boundary conditions and the regions over which they
//      are applied.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/16/2014 04:10:48
////////////////////////////////////////////////////////////////////////////////
#ifndef BOUNDARYCONDITIONS_HH
#define BOUNDARYCONDITIONS_HH
#include <MeshFEM/PeriodicBoundaryMatcher.hh>
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/Types.hh>
#include <MeshFEM/ExpressionVector.hh>
#include <MeshFEM/ComponentMask.hh>

#include <stdexcept>
#include <string>
#include <vector>
#include <set>
#include <array>
#include <map>
#include <list>
#include <utility>
#include <queue>
#include <memory>
#include <iostream>
#include <fstream>
#include <bitset>
#include <cassert>
#include <limits>
#include <bitset>

template<size_t _N>
struct BoundaryCondition {
    BoundaryCondition() {
        region = std::make_shared<Region<VectorND<_N>>>();
        region->minCorner = VectorND<_N>::Zero();
        region->maxCorner = VectorND<_N>::Zero();
    }
    BoundaryCondition(const std::shared_ptr<Region<VectorND<_N>>> r) : region(r) {}
    std::shared_ptr<Region<VectorND<_N>>> region;
    bool containsPoint(const VectorND<_N> &p) const { return region->containsPoint(p); }
    virtual ~BoundaryCondition() { }
};


// A bit of a hack--allow fixing a single component of a single periodic
// boundary node pair's displacement to zero.  This component must be orthogonal
// to the periodic face normal. E.g. fixing the x component on the y = 0 and y =
// 1 faces.
template<size_t _N>
class PeriodicPairDirichletCondition {
public:
    PeriodicPairDirichletCondition(size_t c, size_t f)
        : m_faceSpecifier(f) { m_component.set(c); }

    const ComponentMask &component() const { return m_component; }
    size_t faceSpecifier() const { return m_faceSpecifier; }
    bool    hasCondition() const { return m_component.hasAny(_N); }

    // Get a single valid, matching node pair that can implement this condition.
    // Guarantees to always return the same pair (and the pair is cached).
    // Be careful not to reuse the same PeriodicPairDirichletCondition with
    // different meshes!
    template<typename Mesh>
    std::pair<size_t, size_t> pair(const Mesh &mesh, Real epsilon = 1e-5) {
        if (!hasCondition()) std::runtime_error("Tried to read empty PeriodicPairDirichletCondition");
        assert(m_faceSpecifier < _N);
        BBox<VectorND<_N>> bbox = mesh.boundingBox();
        if (!cached) {
            VectorND<_N> pointToMatch;
            size_t i;
            for (i = 0; i < mesh.numBoundaryNodes(); ++i) {
                auto vn = mesh.boundaryNode(i).volumeNode();
                if (std::abs(vn->p[m_faceSpecifier] - bbox.minCorner[m_faceSpecifier]) <= epsilon) {
                    pointToMatch = vn->p;
                    pointToMatch[m_faceSpecifier] = bbox.maxCorner[m_faceSpecifier];
                    m_pair.first = i;
                    break;
                }
            }
            if (i == mesh.numBoundaryNodes())
                throw std::runtime_error("No vertices on the periodic pair face.");
            for (i = 0; i < mesh.numBoundaryNodes(); ++i) {
                auto vn = mesh.boundaryNode(i).volumeNode();
                if ((vn->p - pointToMatch).norm() <= epsilon) {
                    m_pair.second = i;
                    break;
                }
            }
            if (i == mesh.numBoundaryNodes())
                throw std::runtime_error("Couldn't match vertex in periodic pair Dirichlet condition");
        }

        return m_pair;
    }
private:
    ComponentMask m_component;
    size_t m_faceSpecifier;
    bool cached = false;
    std::pair<size_t, size_t> m_pair;
};


template<size_t _N>
using CondPtr      = std::shared_ptr<BoundaryCondition<_N> >;
template<size_t _N>
using ConstCondPtr = std::shared_ptr<const BoundaryCondition<_N> >;


// Initially, only important thing is the contact region itself. In the future, a friction coefficient may be added and
// some other parameters
template<size_t _N>
struct ContactCondition : public BoundaryCondition<_N> {
    ContactCondition(const std::shared_ptr<Region<VectorND<_N>>> &r) : BoundaryCondition<_N>(r) { }
    virtual ~ContactCondition() { }
};

// Initially, only important thing is the contact region itself. In the future, a friction coefficient may be added and
// some other parameters
template<size_t _N>
struct ContactElementsCondition : public BoundaryCondition<_N> {
    ContactElementsCondition(std::set<int> indices) {
        m_contactIndices = indices;
    }

    bool containElement(int index) const {
        if (m_contactIndices.find(index) == m_contactIndices.end()) {
            return false;
        }
        else {
            return true;
        }
    }

    virtual ~ContactElementsCondition() { }
private:

    std::set<int> m_contactIndices;
};


// Fracture here represents contact between two regions of the object with same material
template<size_t _N>
struct FractureCondition : public BoundaryCondition<_N> {
    FractureCondition(const std::shared_ptr<Region<VectorND<_N>>> &r) : BoundaryCondition<_N>(r) { }
    virtual ~FractureCondition() { }
};

// Fracture here represents contact between two regions of the object with same material
// In the FractureElementsCondition case, we don't have a region but explicitly the indices
// of the touching boundaries
template<size_t _N>
struct FractureElementsCondition : public BoundaryCondition<_N> {
    FractureElementsCondition(std::set<UnorderedPair> pairs) {
        m_contactPairs = pairs;
    }

    bool ContainPair(UnorderedPair pair) const {
        if (m_contactPairs.find(pair) == m_contactPairs.end()) {
            return false;
        }
        else {
            return true;
        }
    }

    virtual ~FractureElementsCondition() { }
private:

    std::set<UnorderedPair> m_contactPairs;
};


enum class NeumannType { Pressure, Traction, Force };
// For the NeumannType::Force case, the force vector is stored in the "traction"
// field, and it is divided by the region's boundary area at application time.
template<size_t _N>
struct NeumannCondition : public BoundaryCondition<_N> {
    NeumannCondition(const std::shared_ptr<Region<VectorND<_N>>> &r, Real p)
        : BoundaryCondition<_N>(r), type(NeumannType::Pressure),
          m_isExpr(false) { m_vecValue[0] = p; }

    NeumannCondition(const std::shared_ptr<Region<VectorND<_N>>> &r, const VectorND<_N> &t,
                     NeumannType _type)
        : BoundaryCondition<_N>(r), type(_type), m_vecValue(t),
          m_isExpr(false) { }

    NeumannCondition(const std::shared_ptr<Region<VectorND<_N>>> &r, const ExpressionVector &ev,
                     NeumannType _type)
        : BoundaryCondition<_N>(r), type(_type), m_isExpr(true),
          m_exprVecValue(ev) {
        if (m_exprVecValue.size() != _N)
            throw std::runtime_error("Bad expression vector length");
    }

    VectorND<_N> traction(const ExpressionEnvironment &env = ExpressionEnvironment()) const {
        assert((type == NeumannType::Traction) ||
               (type == NeumannType::Force && !m_isExpr));
        if (m_isExpr) return m_exprVecValue.eval<_N>(env);
        else          return m_vecValue;
    }

    Real pressure(const ExpressionEnvironment &/* env */ = ExpressionEnvironment()) const {
        assert(type == NeumannType::Pressure);
        if (m_isExpr)
            throw std::runtime_error("Unimplemented");
        return m_vecValue[0];
    }

    NeumannType type;

    virtual ~NeumannCondition() { }

private:
    VectorND<_N> m_vecValue;
    VectorND<_N> m_traction;
    bool m_isExpr;
    ExpressionVector m_exprVecValue;
};

template<size_t _N>
struct DirichletCondition : public BoundaryCondition<_N> {
    DirichletCondition(const std::shared_ptr<Region<VectorND<_N>>> &r, const VectorND<_N> &d, const ComponentMask &m)
        : BoundaryCondition<_N>(r), componentMask(m), m_isExpr(false), m_displacement(d) { }

    DirichletCondition(const std::shared_ptr<Region<VectorND<_N>>> &r, const ExpressionVector &ev,
                       const ComponentMask &m)
        : BoundaryCondition<_N>(r), componentMask(m), m_isExpr(true),
          m_displacementExpr(ev) {
        if (m_displacementExpr.size() != _N)
            throw std::runtime_error("Bad expression vector length");
    }

    VectorND<_N> displacement(const ExpressionEnvironment &env = ExpressionEnvironment()) const {
        if (m_isExpr) return m_displacementExpr.eval<_N>(env);
        else          return m_displacement;
    }

    virtual ~DirichletCondition() { }

    ComponentMask componentMask; // 1 if condition affects component

private:
    bool m_isExpr;
    VectorND<_N> m_displacement;
    ExpressionVector m_displacementExpr;
};

// Behaves just like Dirichlet
// WARNING: will dynamically cast to DirichletCondition, so care must be taken
// not to interpret target conditions as Dirichlet.
template<size_t _N>
struct TargetCondition : public DirichletCondition<_N> {
    using DirichletCondition<_N>::DirichletCondition;
};

template<size_t _N>
struct NeumannElementsCondition : public BoundaryCondition<_N> {
    NeumannElementsCondition(NeumannType t,
                             const std::vector<UnorderedTriplet> &element_corners,
                             const std::vector<VectorND<_N>> &values) {
        assert(element_corners.size() == values.size());
        for (size_t i = 0; i < element_corners.size(); ++i) {
            if      (t == NeumannType::Traction) m_vals[element_corners[i]] = Value(values[i]);
            else if (t == NeumannType::Pressure) m_vals[element_corners[i]] = Value(values[i][0]);
            else if (t == NeumannType::Force)    m_vals[element_corners[i]] = Value(values[i], t);
        }
    }

    struct Value {
        Value(Real p = 0.0) : type(NeumannType::Pressure) { m_val[0] = p; }
        Value(const VectorND<_N> &t, NeumannType inputType = NeumannType::Traction) : type(inputType), m_val(t) { }
        NeumannType type;

        Real pressure() const {
            if (type != NeumannType::Pressure)
                throw std::runtime_error("Neumann condition isn't pressure.");
            return m_val[0];
        }

        const VectorND<_N> &traction() const {
            if (type != NeumannType::Traction)
                throw std::runtime_error("Neumann condition isn't traction.");
            return m_val;
        }

        const VectorND<_N> &force() const {
            if (type != NeumannType::Force)
                throw std::runtime_error("Neumann condition isn't force.");
            return m_val;
        }

    private:
        VectorND<_N> m_val;
    };

    void setValue(Real pressure, size_t v0, size_t v1, size_t v2 = 0) {
        UnorderedTriplet elem(v0, v1, v2);
        m_vals[elem] = Value(pressure);
    }

    void setValue(const VectorND<_N> &traction, size_t v0, size_t v1, size_t v2 = 0) {
        UnorderedTriplet elem(v0, v1, v2);
        m_vals[elem] = Value(traction);
    }

    void setValue(const VectorND<_N> &force) {
        for (auto it = m_vals.begin(); it != m_vals.end(); it++) {
            m_vals[it->first] = Value(force, NeumannType::Force);
        }
    }

    const Value &getValue(const UnorderedTriplet &elem) const {
        return m_vals.at(elem);
    }

    const Value &getValue(size_t v0, size_t v1, size_t v2 = 0) const {
        UnorderedTriplet elem(v0, v1, v2);
        return getValue(elem);
    }

    bool hasValueForElement(const UnorderedTriplet &elem) const {
        return m_vals.count(elem) == 1;
    }

    bool hasValueForElement(size_t v0, size_t v1, size_t v2 = 0) const {
        UnorderedTriplet elem(v0, v1, v2);
        return hasValueForElement(elem);
    }

    /*! Number of elements this condition affects. */
    size_t numElements() const { return m_vals.size(); }

    virtual ~NeumannElementsCondition() { }

private:
    std::map<UnorderedTriplet, Value> m_vals;
};

template<size_t _N>
struct DirichletNodesCondition : public BoundaryCondition<_N> {
    DirichletNodesCondition(std::vector<size_t> nidxs, std::vector<VectorND<_N>> ndisps, const ComponentMask &m)
        : componentMask(m), indices(nidxs), displacements(ndisps) { }

    // All nodes in the condition get the same mask
    ComponentMask componentMask;
    std::vector<size_t> indices;
    std::vector<VectorND<_N>> displacements;
    virtual ~DirichletNodesCondition() { }
};

template<size_t _N>
struct DirichletElementsCondition : public BoundaryCondition<_N> {
    DirichletElementsCondition(
            std::vector<IVectorND<_N>> nidxs, const VectorND<_N> &d, const ComponentMask &m)
        : componentMask(m), m_corners(nidxs), m_isExpr(false), m_displacement(d)
    { sortIndices(); }

    DirichletElementsCondition(
            std::vector<IVectorND<_N>> nidxs, const ExpressionVector &ev, const ComponentMask &m)
        : componentMask(m), m_corners(nidxs), m_isExpr(true), m_displacementExpr(ev)
    { sortIndices(); }

    virtual ~DirichletElementsCondition() { }

    bool containsElement(IVectorND<_N> idx) const {
        std::sort(idx.begin(), idx.end());
        return std::binary_search(m_corners.begin(), m_corners.end(), idx);
    }

    VectorND<_N> displacement(const ExpressionEnvironment &env = ExpressionEnvironment()) const {
        if (m_isExpr) return m_displacementExpr.eval<_N>(env);
        else          return m_displacement;
    }

    // All nodes in the condition get the same mask
    ComponentMask componentMask;
private:
    std::vector<IVectorND<_N>> m_corners;
    bool m_isExpr;
    VectorND<_N> m_displacement;
    ExpressionVector m_displacementExpr;

    void sortIndices() {
        for (auto &idx : m_corners) { std::sort(idx.begin(), idx.end()); }
        std::sort(m_corners.begin(), m_corners.end());
    }
};

// Behaves just like Dirichlet
// WARNING: will dynamically cast to DirichletCondition, so care must be taken
// not to interpret target conditions as Dirichlet.
template<size_t _N>
struct TargetNodesCondition : public DirichletNodesCondition<_N> {
    using DirichletNodesCondition<_N>::DirichletNodesCondition;
};

// Delta function applied to all volume/boundary nodes appearing in region.
template<size_t _N>
struct DeltaForceCondition : public BoundaryCondition<_N> {
    DeltaForceCondition(const std::shared_ptr<Region<VectorND<_N>>> &r, const VectorND<_N> &f)
        : BoundaryCondition<_N>(r), m_isExpr(false), m_force(f) { }

    DeltaForceCondition(const std::shared_ptr<Region<VectorND<_N>>> &r, const ExpressionVector &ev)
        : BoundaryCondition<_N>(r), m_isExpr(true), m_forceExpr(ev) {
        if (ev.size() != _N) throw std::runtime_error("Bad expression vector length");
    }

    VectorND<_N> force(const ExpressionEnvironment &env = ExpressionEnvironment()) const {
        if (m_isExpr) return m_forceExpr.eval<_N>(env);
        else          return m_force;
    }

    virtual ~DeltaForceCondition() { }
private:
    bool m_isExpr;
    VectorND<_N> m_force;
    ExpressionVector m_forceExpr;
};

template<size_t _N>
struct DeltaForceNodesCondition : public BoundaryCondition<_N> {
    DeltaForceNodesCondition(std::vector<size_t> nidxs, std::vector<VectorND<_N>> nforces)
        : indices(nidxs), forces(nforces) { }

    std::vector<size_t> indices;
    std::vector<VectorND<_N>> forces;
    virtual ~DeltaForceNodesCondition() { }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Condition I/O
////////////////////////////////////////////////////////////////////////////////
template<size_t _N> MESHFEM_EXPORT void writeBoundaryConditions(const std::string &cpath, const std::vector<ConstCondPtr<_N> > &conds);
template<size_t _N> MESHFEM_EXPORT void writeBoundaryConditions(std::ostream &os,         const std::vector<ConstCondPtr<_N> > &conds);
template<size_t _N> MESHFEM_EXPORT std::vector<CondPtr<_N> > readBoundaryConditions(const std::string &cpath, const BBox<VectorND<_N>> &bbox, bool &noRigidMotion);
template<size_t _N> MESHFEM_EXPORT std::vector<CondPtr<_N> > readBoundaryConditions(std::istream &is,         const BBox<VectorND<_N>> &bbox, bool &noRigidMotion);
template<size_t _N> MESHFEM_EXPORT std::vector<CondPtr<_N> > readBoundaryConditions(const std::string &cpath, const BBox<VectorND<_N>> &bbox, bool &noRigidMotion, std::vector<PeriodicPairDirichletCondition<_N>> &pp, ComponentMask &pinTranslation);
template<size_t _N> MESHFEM_EXPORT std::vector<CondPtr<_N> > readBoundaryConditions(std::istream &is,         const BBox<VectorND<_N>> &bbox, bool &noRigidMotion, std::vector<PeriodicPairDirichletCondition<_N>> &pp, ComponentMask &pinTranslation);

////////////////////////////////////////////////////////////////////////////////
// Periodic boundary condition implementation
// (Nothing to read from input files--just specified either in code or command
//  line switch.)
////////////////////////////////////////////////////////////////////////////////
template<size_t _N>
class PeriodicCondition {
public:
    static constexpr size_t NO_PAIR = std::numeric_limits<size_t>::max();
    static constexpr size_t NO_DOF  = std::numeric_limits<size_t>::max();

    template<typename Mesh>
    PeriodicCondition(const Mesh &mesh, Real epsilon = 1e-7, bool ignoreMismatch = false, const std::vector<size_t> &ignoreDims = std::vector<size_t>())
        : m_ignoreDims(ignoreDims)
    {
        BBox<VectorND<_N>> cell = mesh.boundingBox();

        std::vector<VectorND<_N>> bdryPts;
        bdryPts.reserve(mesh.numBoundaryNodes());
        for (auto bn : mesh.boundaryNodes()) bdryPts.push_back(bn.volumeNode()->p);

        PeriodicBoundaryMatcher::determineCellBoundaryFaceMembership(bdryPts,
                cell, m_periodicBoundariesForBoundaryNode, epsilon);

        // Remove boundary vertices on ignored cell faces by removing appropriate cell face memberships.
        if (ignoreDims.size() > 0) {
            std::vector<size_t> periodicDims;
            for (size_t d = 0; d < _N; d++) {
                if (std::find(ignoreDims.begin(), ignoreDims.end(), d) == ignoreDims.end()) {
                    periodicDims.push_back(d);
                }
            }

            assert(m_periodicBoundariesForBoundaryNode.size() == bdryPts.size());
            std::vector<bool> onSignificantDim(m_periodicBoundariesForBoundaryNode.size(), false);
            for (size_t i = 0; i < m_periodicBoundariesForBoundaryNode.size(); i++) {
                for (size_t dim : periodicDims) {
                    onSignificantDim[i] = onSignificantDim[i] | m_periodicBoundariesForBoundaryNode[i].onMinFace(dim);
                    onSignificantDim[i] = onSignificantDim[i] | m_periodicBoundariesForBoundaryNode[i].onMaxFace(dim);
                }
                if (onSignificantDim[i]) {
                    // Only remove membership from ignored faces
                    for (size_t d : ignoreDims) {
                        m_periodicBoundariesForBoundaryNode[i].membership[d] = false;
                        m_periodicBoundariesForBoundaryNode[i].membership[d + _N] = false;
                    }
                }
                else {
                    // Remove membership from all faces
                    for (size_t d = 0; d < _N; d++) {
                        m_periodicBoundariesForBoundaryNode[i].membership[d] = false;
                        m_periodicBoundariesForBoundaryNode[i].membership[d + _N] = false;
                    }
                }
            }
        }

        // Determine identified boundary nodes.
        std::vector<std::vector<size_t> > bdryNodeSets;
        std::vector<size_t              > bdryNodeSetForBdryNode;
        if (ignoreMismatch) {
            PeriodicBoundaryMatcher::matchPermittingMismatch(bdryPts, cell,
                    m_periodicBoundariesForBoundaryNode,
                    bdryNodeSets, bdryNodeSetForBdryNode, epsilon);
        }
        else {
            PeriodicBoundaryMatcher::match(bdryPts, cell,
                    m_periodicBoundariesForBoundaryNode,
                    bdryNodeSets, bdryNodeSetForBdryNode, epsilon);
        }

        // Mark periodic boundary elements: those with all corners on the same
        // periodic boundary.
        m_isPeriodicBoundaryElement = PeriodicBoundaryMatcher::determineCellFaceBoundaryElements(mesh,
                m_periodicBoundariesForBoundaryNode);

        // Determine the "DoF index" for every node in the mesh. For internal
        // nodes, these are all unique. On the periodic boundary, these will be
        // shared by identified nodes. These indices are created assuming one
        // variable per node. For 3D elasticity, there will actually be three
        // DOFs per node i, with indices
        //   [ 3 * m_dofForNode[i] + 0, 3 * m_dofForNode[i] + 1,
        //     3 * m_dofForNode[i] + 2 ]
        m_dofForNode.assign(mesh.numNodes(), size_t(NO_DOF));
        m_nodesForDoF.clear(), m_nodesForDoF.reserve(mesh.numInternalNodes() +
                                                     bdryNodeSets.size());

        // Assign DoF indices to each node (and its identified nodes) in order.
        for (auto n : mesh.nodes()) {
            if (m_dofForNode[n.index()] != NO_DOF) continue;
            auto bn = n.boundaryNode();
            if (bn) {
                // Create a DoF shared between bn and all its identified nodes.
                // Note: invalidates entry of bdryNodeSets!!!
                auto &ns = bdryNodeSets[bdryNodeSetForBdryNode[bn.index()]];
                for (size_t &ni : ns) {
                    ni = mesh.boundaryNode(ni).volumeNode().index();
                    assert(m_dofForNode.at(ni) == NO_DOF);
                    m_dofForNode[ni] = m_nodesForDoF.size();
                }
                m_nodesForDoF.emplace_back(std::move(ns));
            }
            else {
                // Create a DoF with a single associated node for each internal node.
                assert(m_dofForNode.at(n.index()) == NO_DOF);
                m_dofForNode.at(n.index()) = m_nodesForDoF.size();
                m_nodesForDoF.emplace_back(1, n.index());
            }
        }

        // All nodes should have been assigned valid DOFs
        for (size_t i = 0; i < mesh.numNodes(); ++i) {
            assert(m_dofForNode[i] != NO_DOF);
            assert(m_dofForNode[i] < numPeriodicDoFs());
        }
    }

    // Constructor reading the periodic boundary conditions from a file.
    // This is a hack that only loads periodic nodes--i.e., it does not
    // mark periodic boundary elements.
    template<typename Mesh>
    PeriodicCondition(const Mesh &mesh, const std::string &pcFile) {
        // The periodic condition file contains pairs of nodes
        // that are periodically identified.
        std::cerr << "WARNING: periodic boundary condition files are a temporary hack." << std::endl;

        std::ifstream file(pcFile);
        if (!file.is_open()) throw std::runtime_error("Couldn't open " + pcFile);

        std::vector<std::vector<size_t>> adj(mesh.numNodes());
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream ls(line);
            size_t a, b;
            ls >> a >> b;
            adj[a].push_back(b);
            adj[b].push_back(a);
        }

        // Assign unique dofs for each connected component in the
        // identified node graph represented by adjacency list `adj`
        std::queue<size_t> bfsQueue;
        m_dofForNode.assign(mesh.numNodes(), size_t(NO_DOF));
        m_nodesForDoF.reserve(mesh.numNodes());
        m_nodesForDoF.clear();
        for (const auto n : mesh.nodes()) {
            if (m_dofForNode[n.index()] != NO_DOF) continue;
            size_t dof = m_nodesForDoF.size();
            m_nodesForDoF.emplace_back(1, n.index());
            auto &nodeSet = m_nodesForDoF.back();

            bfsQueue.push(n.index());
            m_dofForNode[n.index()] = dof;
            while (!bfsQueue.empty()) {
                size_t u = bfsQueue.front();
                bfsQueue.pop();
                for (const size_t v : adj[u]) {
                    size_t &dof_v = m_dofForNode[v];
                    if (dof_v != NO_DOF) {
                        assert(dof_v == dof);
                        continue;
                    }
                    bfsQueue.push(v);
                    dof_v = dof;
                    nodeSet.push_back(v);
                }
            }
        }

        m_isPeriodicBoundaryElement.assign(mesh.numBoundaryElements(), false);
    }

    // Guaranteed monotonically increasing with (lowest identified) node index
    const std::vector<size_t> &periodicDoFsForNodes() const {
        return m_dofForNode;
    }

    const std::vector<size_t> &getIgnoreDims() const { return m_ignoreDims; }

    // Check if a given boundary element is periodic
    bool isPeriodicBE(size_t be) const {
        return m_isPeriodicBoundaryElement.at(be);
    }

    // Check if a *volume* node is periodic
    bool isPeriodicNode(int vni) const {
        return identifiedNodes(vni).size() > 1;
    }

    // The *volume* nodes identified with the vi^th *volume* node
    // (including vni itself).
    const std::vector<size_t> &identifiedNodes(int vni) const {
        return m_nodesForDoF.at(m_dofForNode.at(vni));
    }

    // Return 0 if boundary node bni is not on the d^th min or max cell face
    // Return -1 if it's on the min face
    // Return  1 if it's on the max face
    int bdryNodeOnMinOrMaxPeriodCellFace(size_t bni, size_t d) const {
        assert(d < _N);
        const auto &bdry = m_periodicBoundariesForBoundaryNode.at(bni);
        if (bdry.onMinFace(d)) return -1;
        if (bdry.onMaxFace(d)) return  1;
        return 0;
    }

    // Determines whether the set of periodic cell faces on which node "a"
    // lies is contained in the set of periodic cell faces on which node "b"
    // lies.
    // This is a partial order: e.g. a pair of nodes each lying on a single,
    // distinct periodic cell face cannot be compared.
    // Returns true if "a <= b"   (either a < b or a ==  b)
    //         false otherwise    (either b < a or a and b cannot be compared)
    bool bdryNodePeriodCellFacesPartialOrderLessEq(size_t a, size_t b) const {
        return m_periodicBoundariesForBoundaryNode.at(a)
            <= m_periodicBoundariesForBoundaryNode.at(b);
    }

    size_t numPeriodicDoFs() const { return m_nodesForDoF.size(); }

private:
    std::vector<bool> m_isPeriodicBoundaryElement;
    // Which periodic boundaries is a boundary node on?
    std::vector<PeriodicBoundaryMatcher::FaceMembership<_N>> m_periodicBoundariesForBoundaryNode;
    std::vector<size_t> m_dofForNode; // Guaranteed monotonically increasing with (lowest identified) node index
    std::vector<std::vector<size_t>> m_nodesForDoF;
    std::vector<size_t> m_ignoreDims;
};

#endif /* end of include guard: BOUNDARYCONDITIONS_HH */
