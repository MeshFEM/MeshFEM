////////////////////////////////////////////////////////////////////////////////
// PeriodicBoundaryMatcher.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Determine the periodic cell face membership for a collection of node
//      points as well as the sets of nodes that are identified with each other.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/05/2016 13:31:56
////////////////////////////////////////////////////////////////////////////////
#ifndef PERIODICBOUNDARYMATCHER_HH
#define PERIODICBOUNDARYMATCHER_HH

#include <vector>
#include <bitset>
#include <limits>
#include <stdexcept>
#include <sstream>

#include "CollisionGrid.hh"
#include "Geometry.hh"

namespace PeriodicBoundaryMatcher {

////////////////////////////////////////////////////////////////////////////////
// Periodic Cell Face Membership
////////////////////////////////////////////////////////////////////////////////
// Encode the set of periodic boundaries a node lies on as a bitset with the
// following bits:
//      0, _N + 0: on min, max x face
//      1, _N + 1: on min, max y face
//    [ 2, _N + 2: on min, max z face ]
template<size_t N>
struct FaceMembership {
    std::bitset<2 * N> membership;

    // Construct membership for p in cell.
    template<class Point>
    FaceMembership(const Point &p, const BBox<VectorND<N>> &cell,
                   Real epsilon = 1e-5) {
        for (size_t d = 0; d < N; ++d) {
            membership[d]     = std::abs(p[d] - cell.minCorner[d]) < epsilon;
            membership[N + d] = std::abs(p[d] - cell.maxCorner[d]) < epsilon;
        }
    }

    bool      onMinFace(size_t d) const { assert(d < N); return membership[    d]; }
    bool      onMaxFace(size_t d) const { assert(d < N); return membership[N + d]; }
    bool onMinOrMaxFace(size_t d) const { return onMinFace(d) || onMinOrMaxFace(d); }
    bool      onAnyFace()         const { return membership.any(); }
    size_t        count() const { return membership.count(); }
    // Verify that the node is not on both the min and max face.
    bool          valid() const { return (membership & (membership >> N)).none(); }
    // More advanced membership queries, useful to determine the "minimal"
    // periodic node, to which the identified nodeset can be linked.
    bool   onAnyMaxFace() const { return (membership >> N).any(); }
    bool   onAnyMinFace() const { return (membership << N).any(); }
    bool  isMinimalNode() const { return !onAnyMaxFace(); }

    FaceMembership &operator&=(const FaceMembership<N> &b) { membership &= b.membership; return *this; }
};

// *Partial* order on boundary membership:
// a <= b if a's membership set is contained in b's.
template<size_t N>
bool operator<=(const FaceMembership<N> &a, const FaceMembership<N> &b) {
    return ((a.membership & b.membership) == a.membership);
}

template<size_t N>
FaceMembership<N> operator&(FaceMembership<N> a, const FaceMembership<N> &b) {
    a &= b;
    return a;
}

// Determine the periodic cell face membership for a collection of node points.
template<size_t N, class PointCollection>
void determineCellBoundaryFaceMembership(const PointCollection &bdryPoints,
        const BBox<VectorND<int(N)>> &cell,
        std::vector<FaceMembership<N>> &faceMembership,
        Real epsilon = 1e-5)
{
    faceMembership.clear(), faceMembership.reserve(bdryPoints.size());
    for (const auto &p : bdryPoints) {
        faceMembership.emplace_back(p, cell, epsilon);
        assert(faceMembership.back().valid());
    }
}

// Determine the periodic cell nodes that are identified with each other.
template<size_t N, class PointCollection>
void match(const PointCollection &bdryPoints,
        const BBox<VectorND<int(N)>> &cell,
        const std::vector<FaceMembership<N>> &faceMembership,
        std::vector<std::vector<size_t>>     &nodeSets,
        std::vector<size_t>                  &nodeSetForNode,
        Real epsilon = 1e-5)
{
    // Choose a cell size on the order of epsilon, but prevent cell sizes so
    // small as to cause index overflows for objects of size up to 100x100
    // centered at the origin: max int ~10^9 ==> cellSize > 10^-7
    CollisionGrid<Real, VectorND<N>> cgrid(std::max(epsilon, 1.0e-7));

    // New simpler approach:
    //   Add all non-minimal bbox points to the cgrid at once, instead of
    //   treating one periodic direction at a time. This lets us avoid
    //   recovering identified node sets using a BFS, at the expense of more
    //   costly collision grid queries.
    //   If we find this is too slow, we can revert to the BFS approach.
    assert(faceMembership.size() == bdryPoints.size());
    int i = -1; // function-global variable for enumerating bdryPoints...
    for (const auto &p : bdryPoints) {
        ++i;
        if (!faceMembership[i].isMinimalNode())
            cgrid.addPoint(p, i);
    }

    static constexpr size_t NONE = std::numeric_limits<size_t>::max();
    nodeSetForNode.assign(bdryPoints.size(), NONE);

    // Determine number of nodesets.
    size_t numNodesets = 0;
    for (i = 0; i < int(bdryPoints.size()); ++i)
        numNodesets += faceMembership[i].isMinimalNode();

    nodeSets.clear(), nodeSets.reserve(numNodesets);

    // Create a new identified node set for each "minimal" node (in only min
    // faces). For a minimal node on d period faces, there will be 2^d nodes in
    // the identified set.
    // Add all corresponding nodes to this set, and mark their node set index.
    i = -1;
    for (const auto &p : bdryPoints) {
        ++i;
        const auto &fm = faceMembership[i];
        if (fm.isMinimalNode()) {
            assert(nodeSetForNode[i] == NONE);
            nodeSetForNode[i] = nodeSets.size();

            size_t numPeriodicFaces = fm.count();
            size_t numIdentifiedNodes = 1 << numPeriodicFaces;
            nodeSets.push_back(std::vector<size_t>(numIdentifiedNodes, NONE));

            auto &ns = nodeSets.back();
            ns[0] = i;

            // Search for the 2^d identified nodes in the collision grid.
            for (size_t n = 1; n < numIdentifiedNodes; ++n) {
                auto query = p;
                size_t idx = 0; // enumerates i's periodic faces
                for (size_t d = 0; d < N; ++d) {
                    if (fm.onMinFace(d))
                        if (n & (1 << idx++)) query[d] = cell.maxCorner[d];
                }
                assert(idx == numPeriodicFaces);

                auto result = cgrid.getClosestPoint(query, epsilon);
                if (result.first < 0) {
                    std::stringstream ss;
                    ss << "Couldn't find " << n << "th periodic-identified node "
                       << "for minimal boundary node " << i << " at " << p
                       << "; looking for " << query << std::endl;
                    throw std::runtime_error(ss.str());
                }
                size_t pair = result.first;
                assert(faceMembership[pair].count() == numPeriodicFaces);
                if (nodeSetForNode.at(pair) != NONE)
                    throw std::runtime_error("Non bijective node set assignment.");
                nodeSetForNode.at(pair) = nodeSetForNode[i];
                ns[n] = pair;
            }
        }
    }

    // Make sure every node is in a set.
    i = -1;
    for (const auto &p : bdryPoints) {
        ++i;
        if (nodeSetForNode[i] == NONE) {
            std::stringstream ss;
            ss << "Unmatched non-minimal boundary node " << i
               << " at " << p << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
}

#if 0
// old BFS-based version
        // Sparse collection of periodically-paired nodes for each node.
        // (VOLUME NODE INDICES!)
        // Unpaired nodes have no entry in this map, while paired nodes
        // map to an _N-dim array with at least one entry that isn't NO_PAIR.
        std::map<size_t, std::array<size_t, _N>> pair;

        // To determine if a boundary face is on the periodic boundary, we
        // check if all its nodes are on *the same* periodic boundary
        // (just checking if all nodes are on a periodic boundary will
        //  incorrectly mark some faces near the period cell edges)
        // This map from boundary nodes to the set of periodic boundary it lies
        // on is encoded as a bitset with the following bits
        //      0, _N + 0: on min, max x face
        //      1, _N + 1: on min, max y face
        //    [ 2, _N + 2: on min, max z face ]
        m_periodicBoundariesForBoundaryNode.clear();
        m_periodicBoundariesForBoundaryNode.resize(mesh.numBoundaryNodes());

        for (size_t d = 0; d < _N; ++d) {
            cgrid.reset();
            std::vector<size_t> maxfaceNodes;
            for (size_t i = 0; i < mesh.numBoundaryNodes(); ++i) {
                auto vn = mesh.boundaryNode(i).volumeNode();
                if (std::abs(vn->p[d] - cell.minCorner[d]) < epsilon) {
                    cgrid.addPoint(vn->p, vn.index());
                    m_periodicBoundariesForBoundaryNode[i].set(d);
                }
                if (std::abs(vn->p[d] - cell.maxCorner[d]) < epsilon) {
                    maxfaceNodes.push_back(vn.index());
                    m_periodicBoundariesForBoundaryNode[i].set(_N + d);
                }
            }
            for (size_t ni : maxfaceNodes) {
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
                if (pair.count(ni) == 0) {
                    assert(pair.count(pi) == 0);
                    // Cast needed; can't  pass storage-less constexpr to fill
                    pair[ni].fill(size_t(NO_PAIR));
                    pair[pi].fill(size_t(NO_PAIR));
                }
                assert(pair.count(pi) + pair.count(ni) == 2);
                auto &np = pair[ni];
                auto &pp = pair[pi];
                if ((np[d] != NO_PAIR) || (pp[d] != NO_PAIR))
                    throw std::runtime_error("Non-bijective boundary matching");
                np[d] = pi;
                pp[d] = ni;
            }
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
        m_dofForNode.assign(mesh.numNodes(), size_t(NO_DOF));
        m_nodesForDoF.clear();
        m_nodesForDoF.reserve(mesh.numNodes()); // very conservative--prevent realloc
        for (size_t i = 0; i < mesh.numNodes(); ++i) {
            if (m_dofForNode[i] != NO_DOF) continue;
            size_t dof = m_nodesForDoF.size();
            m_dofForNode[i] = dof;
            m_nodesForDoF.emplace_back(1, i);
            std::queue<size_t> bfsQueue;
            bfsQueue.push(i);
            while (!bfsQueue.empty()) {
                int u = bfsQueue.front(); bfsQueue.pop();
                assert(size_t(m_dofForNode[u]) == dof);
                auto it = pair.find(u);
                if (it == pair.end()) continue;
                for (size_t v: it->second) {
                    if (v == NO_PAIR) continue;
                    if (m_dofForNode[v] == NO_DOF) {
                        m_dofForNode[v] = dof;
                        m_nodesForDoF.at(dof).push_back(v);
                        bfsQueue.push(v);
                    }
                    else assert(size_t(m_dofForNode[v]) == dof);
                }
            }
        }

        // All nodes should have been assigned valid DOFs
        for (size_t i = 0; i < mesh.numNodes(); ++i) {
            assert(m_dofForNode[i] != NO_DOF);
            assert(m_dofForNode[i] < numPeriodicDoFs());
        }
#endif

}

#endif /* end of include guard: PERIODICBOUNDARYMATCHER_HH */
