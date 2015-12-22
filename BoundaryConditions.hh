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
#include "CollisionGrid.hh"
#include "Geometry.hh"
#include "Types.hh"
#include "ExpressionVector.hh"
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <list>
#include <utility>
#include <queue>
#include <memory>
#include <iostream>
#include <bitset>
#include <cassert>
#include <limits>

template<size_t _N>
struct BoundaryCondition {
    BoundaryCondition() { }
    BoundaryCondition(const BBox<VectorND<_N>> &r) : region(r) { }
    BBox<VectorND<_N>> region;
    bool containsPoint(const VectorND<_N> &p) const { return region.containsPoint(p); }
    virtual ~BoundaryCondition() { }
};

class ComponentMask {
public:
    ComponentMask(const std::string &components = "") {
        setComponentString(components);
    }

    void setComponentString(const std::string &components) {
        static const std::map<std::string, std::bitset<3>> cmasks = {
            {   "x", std::bitset<3>("001") }, {  "y", std::bitset<3>("010") },
            {   "z", std::bitset<3>("100") }, { "xy", std::bitset<3>("011") },
            {  "yz", std::bitset<3>("110") }, { "xz", std::bitset<3>("101") },
            { "xyz", std::bitset<3>("111") }, {   "", std::bitset<3>("000") } };
        if (cmasks.count(components) == 0)
            throw std::runtime_error("invalid component specifier: " + components);
        m_active = cmasks.at(components);
    }

    bool has(size_t c) const { return m_active.test(c); }
    bool hasX()        const { return m_active[0]; }
    bool hasY()        const { return m_active[1]; }
    bool hasZ()        const { return m_active[2]; }
    bool hasAny(size_t dim) const { return count(dim) > 0; }
    bool hasAll(size_t dim) const { return count(dim) == dim; }
    // Number of active components for dimension (2 or 3)
    size_t count(size_t dim) const {
        if (dim == 3)      return m_active.count();
        else if (dim == 2) return m_active.count() - (hasZ() ? 1 : 0);
        else throw std::runtime_error("Illegal dimension");
    }

    void set()           { m_active.set(); }
    void set(size_t c)   { m_active.set(c); }
    void clear()         { m_active.reset(); }
    void clear(size_t c) { m_active.reset(c); }

    bool operator==(const ComponentMask &b) const { return m_active == b.m_active; }
    bool operator!=(const ComponentMask &b) const { return m_active != b.m_active; }

    // Apply the mask to a vector, clearing any component not set in the mask.
    template<int N>
    VectorND<N> apply(const VectorND<N> &v) const {
        VectorND<N> result(v);
        for (size_t c = 0; c < N; ++c)
            if (!has(c)) result[c] = 0;
        return result;
    }

    std::string componentString() const {
        std::string result;
        if (hasX()) result += "x";
        if (hasY()) result += "y";
        if (hasZ()) result += "z";
        return result;
    }

    friend std::ostream &operator<<(std::ostream &os, const ComponentMask &cm) {
        if (cm.hasX()) os << "x";
        if (cm.hasY()) os << "y";
        if (cm.hasZ()) os << "z";
        return os;
    }

private:
    std::bitset<3> m_active;
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
    // Be careful not to requse the same PeriodicPairDirichletCondition with
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
                if (std::abs(vn->p[m_faceSpecifier] - bbox.minCorner[m_faceSpecifier]) < epsilon) {
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
                if ((vn->p - pointToMatch).norm() < epsilon) {
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

enum class NeumannType { Pressure, Traction, Force };
// For the NeumannType::Force case, the force vector is stored in the "traction"
// field, and it is divided by the region's boundary area at application time.
template<size_t _N>
struct NeumannCondition : public BoundaryCondition<_N> {
    NeumannCondition(const BBox<VectorND<_N>> &region, Real p)
        : BoundaryCondition<_N>(region), type(NeumannType::Pressure),
          m_isExpr(false) { m_vecValue[0] = p; }

    NeumannCondition(const BBox<VectorND<_N>> &region, const VectorND<_N> &t,
                     NeumannType _type)
        : BoundaryCondition<_N>(region), type(_type), m_vecValue(t),
          m_isExpr(false) { }

    NeumannCondition(const BBox<VectorND<_N>> &region, const ExpressionVector &ev,
                     NeumannType _type)
        : BoundaryCondition<_N>(region), type(_type), m_isExpr(true),
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
private:
    VectorND<_N> m_vecValue;
    VectorND<_N> m_traction;
    bool m_isExpr;
    ExpressionVector m_exprVecValue;
    virtual ~NeumannCondition() { }
};

template<size_t _N>
struct DirichletCondition : public BoundaryCondition<_N> {
    DirichletCondition(const BBox<VectorND<_N>> &region, const VectorND<_N> &d, const ComponentMask &m)
        : BoundaryCondition<_N>(region), componentMask(m), m_isExpr(false), m_displacement(d) { }

    DirichletCondition(const BBox<VectorND<_N>> &region, const ExpressionVector &ev,
                       const ComponentMask &m)
        : BoundaryCondition<_N>(region), componentMask(m), m_isExpr(true),
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
    TargetCondition(const BBox<VectorND<_N>> &region, const VectorND<_N> &d, const ComponentMask &m)
        : DirichletCondition<_N>(region, d, m) { }
    TargetCondition(const BBox<VectorND<_N>> &region, const ExpressionVector &ev, const ComponentMask &m)
        : DirichletCondition<_N>(region, ev, m) { }
    virtual ~TargetCondition() { }
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
        }
    }

    struct Value {
        Value(Real p = 0.0) : type(NeumannType::Pressure) { m_val[0] = p; }
        Value(const VectorND<_N> &t) : type(NeumannType::Traction), m_val(t) { }
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

// Behaves just like Dirichlet
// WARNING: will dynamically cast to DirichletCondition, so care must be taken
// not to interpret target conditions as Dirichlet.
template<size_t _N>
struct TargetNodesCondition : public BoundaryCondition<_N> {
    TargetNodesCondition(std::vector<size_t> nidxs, std::vector<VectorND<_N>> ndisps, const ComponentMask &m)
        : componentMask(m), indices(nidxs), displacements(ndisps) { }

    // All nodes in the condition get the same mask
    ComponentMask componentMask;
    std::vector<size_t> indices;
    std::vector<VectorND<_N>> displacements;
    virtual ~TargetNodesCondition() { }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Condition I/O
////////////////////////////////////////////////////////////////////////////////
template<size_t _N> void writeBoundaryConditions(const std::string &cpath, const std::vector<ConstCondPtr<_N> > &conds);
template<size_t _N> void writeBoundaryConditions(std::ostream &os,         const std::vector<ConstCondPtr<_N> > &conds);
template<size_t _N> std::vector<CondPtr<_N> > readBoundaryConditions(const std::string &cpath, const BBox<VectorND<_N>> &bbox, bool &noRigidMotion);
template<size_t _N> std::vector<CondPtr<_N> > readBoundaryConditions(std::istream &is,         const BBox<VectorND<_N>> &bbox, bool &noRigidMotion);
template<size_t _N> std::vector<CondPtr<_N> > readBoundaryConditions(const std::string &cpath, const BBox<VectorND<_N>> &bbox, bool &noRigidMotion, std::vector<PeriodicPairDirichletCondition<_N>> &pp);
template<size_t _N> std::vector<CondPtr<_N> > readBoundaryConditions(std::istream &is,         const BBox<VectorND<_N>> &bbox, bool &noRigidMotion, std::vector<PeriodicPairDirichletCondition<_N>> &pp);

////////////////////////////////////////////////////////////////////////////////
// Periodic boundary condition implementation
// (Nothing to read from input files--just specified either in code or command
//  line switch.)
////////////////////////////////////////////////////////////////////////////////
template<size_t _N>
class PeriodicCondition {
public:
    static constexpr size_t NO_PAIR = std::numeric_limits<size_t>::max();

    template<typename Mesh>
    PeriodicCondition(const Mesh &mesh, Real epsilon = 1e-5) {
        BBox<VectorND<_N>> cell = mesh.boundingBox();
        // Choose a cell size on the order of epsilon. This should be safe since
        // the max node coordinate shouldn't anyhere near large enough that
        // dividing by epsilon causes an overflow. We don't want any larger than
        // epsilon because then we'd have to check for many (empty) boxes.
        CollisionGrid<Real, VectorND<_N>> cgrid(epsilon);
        // Match boundary nodes on opposite faces of the periodic cell
        std::vector<std::pair<int, int> > pairs;

        // To determine if a boundary face is on the periodic boundary, we
        // check if all its nodes are on *the same* periodic boundary
        // (just checking if all nodes are on a periodic boundary will
        //  incorrectly mark some faces near the period cell edges)
        // This map from boundary nodes to the set of periodic boundary it lies
        // on is encoded as an array of integers taking the following values:
        //      0: non-periodic node
        //      1, 2,   4: node is on min x, y, z faces
        //      8, 16, 32: node is on max x, y, z faces
        // (membership in multiple boundaries is implemented by bitwise or,
        //  set intersection is implemented by bitwise and).
        m_periodicBoundariesForNode.assign(mesh.numBoundaryNodes(), 0);

        for (size_t d = 0; d < _N; ++d) {
            cgrid.reset();
            std::vector<size_t> maxfaceNodes;
            for (size_t i = 0; i < mesh.numBoundaryNodes(); ++i) {
                auto vn = mesh.boundaryNode(i).volumeNode();
                if (std::abs(vn->p[d] - cell.minCorner[d]) < epsilon) {
                    cgrid.addPoint(vn->p, vn.index());
                    m_periodicBoundariesForNode[i] |= 1 << d;
                }
                if (std::abs(vn->p[d] - cell.maxCorner[d]) < epsilon) {
                    maxfaceNodes.push_back(vn.index());
                    m_periodicBoundariesForNode[i] |= 1 << (_N + d);
                }
            }
            for (size_t i = 0; i < maxfaceNodes.size(); ++i) {
                size_t ni = maxfaceNodes[i];
                VectorND<_N> query(mesh.node(ni)->p);
                query[d] = cell.minCorner[d];
                auto result = cgrid.getClosestPoint(query, epsilon);
                if (result.first == -1) {
                    std::stringstream ss;
                    auto n = mesh.node(ni);
                    ss << "Couldn't match periodic boundary node " << ni << " "
                       << n->p << "; looking for: " << query << std::endl
                       << "This is a " << ((n.isEdgeNode()) ? "edge" : "vertex")
                       << " node." << std::endl;
                    if (n.isEdgeNode()) {
                        auto edge = mesh.edgeForEdgeNode(n.edgeNodeIndex());
                        ss << "Edge endpoints: "
                           << edge[0] << ", " << edge[1] << std::endl
                           << "(at " << mesh.node(edge[0])->p << " and "
                           << mesh.node(edge[1])->p << ")" << std::endl;
                    }
                    throw std::runtime_error(ss.str());
                }
                assert(result.first >= 0);
                size_t pi = size_t(result.first);
                // Fill out (symmetric) pair adjacency information
                if (m_pair.count(ni) == 0) {
                    assert(m_pair.count(pi) == 0);
                    // Cast needed; can't  pass storage-less constexpr to fill
                    m_pair[ni].fill(size_t(NO_PAIR));
                    m_pair[pi].fill(size_t(NO_PAIR));
                }
                assert(m_pair.count(pi) + m_pair.count(ni) == 2);
                auto &np = m_pair[ni];
                auto &pp = m_pair[pi];
                if ((np[d] != NO_PAIR) || (pp[d] != NO_PAIR))
                    throw std::runtime_error("Non-bijective boundary matching");
                np[d] = pi;
                pp[d] = ni;
            }
        }

        // Mark the periodic boundary elements.
        m_isPeriodicBoundaryElement.resize(mesh.numBoundaryElements());
        for (size_t i = 0; i < mesh.numBoundaryElements(); ++i) {
            auto be = mesh.boundaryElement(i);
            // Determine what periodic boundary this element lies on.
            unsigned char pboundaries = m_periodicBoundariesForNode.at(be.node(0).index());
            for (size_t j = 1; j < be.numNodes(); ++j)
                pboundaries &= m_periodicBoundariesForNode.at(be.node(j).index());
            // It can't be on more than one boundary...
            // (i.e. power or 2 or zero--use bit hack)
            assert((pboundaries & (pboundaries - 1)) == 0);
            m_isPeriodicBoundaryElement[i] = (pboundaries != 0);
        }

        // Determine the "DoF index" for every node in the mesh. For internal
        // nodes, these are all unique. On the periodic boundary, these will be
        // shared by identified nodes. These indices are created assuming one
        // variable per node. For 3D elasticity, there will actually be three
        // DOFs per node i, with indices
        //   [ 3 * m_dofForNode[i] + 0, 3 * m_dofForNode[i] + 1,
        //     3 * m_dofForNode[i] + 2 ]

        // BFS connected components
        // Assign each node in a connected component of identified nodes
        // the same DoF
        m_dofForNode.assign(mesh.numNodes(), -1);
        m_numDoFs = 0;
        for (size_t i = 0; i < mesh.numNodes(); ++i) {
            if (m_dofForNode[i] >= 0) continue;
            m_dofForNode[i] = m_numDoFs++;
            std::queue<size_t> bfsQueue;
            bfsQueue.push(i);
            while (!bfsQueue.empty()) {
                int u = bfsQueue.front(); bfsQueue.pop();
                auto it = m_pair.find(u);
                if (it == m_pair.end()) continue;
                for (size_t v: it->second) {
                    if (v == NO_PAIR) continue;
                    if (m_dofForNode[v] < 0) {
                        assert(size_t(m_dofForNode[u]) == m_numDoFs - 1);
                        m_dofForNode[v] = m_dofForNode[u];
                        bfsQueue.push(v);
                    }
                }
            }
        }
    }

    const std::vector<int> &periodicDoFsForNodes() const {
        return m_dofForNode;
    }

    // Check if a given boundary element is periodic
    bool isPeriodicBE(size_t be) const {
        return m_isPeriodicBoundaryElement.at(be);
    }

    bool isPeriodicVertex(int vi) const {
        return m_pair.count(vi) != 0;
    }

    const std::array<size_t, _N> &periodicPairs(int vi) const {
        return m_pair.at(vi);
    }

    // Return 0 if boundary vertex bvi is not on the d^th min or max cell face
    // Return -1 if it's on the min face
    // Return  1 if it's on the min face
    int bdryVertexOnMinOrMaxPeriodCellFace(size_t bvi, size_t d) const {
        assert(d < _N);
        unsigned char bdry = m_periodicBoundariesForNode.at(bvi);
        if (bdry & (1 << d)) return -1;
        if (bdry & (1 << (_N + d))) return 1;
        return 0;
    }

    size_t numPeriodicDoFs() const { return m_numDoFs; }

private:
    std::vector<bool> m_isPeriodicBoundaryElement;
    std::vector<unsigned char> m_periodicBoundariesForNode;
    size_t m_numDoFs;
    std::vector<int> m_dofForNode;
    // Sparse collection of periodically-paired nodes for each node.
    // (Volume node indices!)
    // Unpaired nodes have no entry in this map, while paired nodes
    // map to an _N-dim array with at least one entry that isn't NO_PAIR.
    std::map<size_t, std::array<size_t, _N>> m_pair;
};

#endif /* end of include guard: BOUNDARYCONDITIONS_HH */
