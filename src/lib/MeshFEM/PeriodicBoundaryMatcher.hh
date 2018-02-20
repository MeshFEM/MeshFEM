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
#include <iostream>
#include <sstream>
#include <queue>

#include <MeshFEM/CollisionGrid.hh>
#include <MeshFEM/Geometry.hh>

namespace PeriodicBoundaryMatcher {

static constexpr size_t NONE = std::numeric_limits<size_t>::max();

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
            membership[d]     = std::abs(p[d] - cell.minCorner[d]) <= epsilon;
            membership[N + d] = std::abs(p[d] - cell.maxCorner[d]) <= epsilon;
        }
    }

    static FaceMembership AllFaces() {
        FaceMembership result;
        result.membership.set();
        return result;
    }

    bool      onMinFace(size_t d) const { assert(d < N); return membership[    d]; }
    bool      onMaxFace(size_t d) const { assert(d < N); return membership[N + d]; }
    bool onMinOrMaxFace(size_t d) const { return onMinFace(d) || onMaxFace(d); }
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
private:
    FaceMembership() { }
};

// Face membership ASCII output.
template<size_t N>
std::ostream &operator<<(std::ostream &os, const FaceMembership<N> &m) {
    if (N > 0 && m.onMinOrMaxFace(0)) os << (m.onMinFace(0) ? "x" : "X");
    if (N > 1 && m.onMinOrMaxFace(1)) os << (m.onMinFace(1) ? "y" : "Y");
    if (N > 2 && m.onMinOrMaxFace(2)) os << (m.onMinFace(2) ? "z" : "Z");

    if (m.count() == 0) os << "(none)";
    return os;
}

// *Partial* order on boundary membership:
// a <= b if a's membership set is contained in b's.
template<size_t N>
bool operator<=(const FaceMembership<N> &a, const FaceMembership<N> &b) {
    return ((a.membership & b.membership) == a.membership);
}

template<size_t N>
bool operator==(const FaceMembership<N> &a, const FaceMembership<N> &b) {
    return a.membership == b.membership;
}

template<size_t N>
bool operator!=(const FaceMembership<N> &a, const FaceMembership<N> &b) {
    return a.membership != b.membership;
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

// Given the cell face membership of every node, determine which boundary
// elements of the mesh lie on the cell faces (which boundary elements have all
// nodes with a single face in common).
template<size_t N, class _FEMMesh>
std::vector<bool>
determineCellFaceBoundaryElements(const _FEMMesh &mesh, const std::vector<FaceMembership<N>> &faceMemberships) {
    std::vector<bool> isOnCellFace(mesh.numBoundaryElements());
    for (auto be : mesh.boundaryElements()) {
        // Determine what periodic boundary this element lies on.
        auto pboundaries = FaceMembership<N>::AllFaces();
        for (auto bn : be.nodes())
            pboundaries &= faceMemberships.at(bn.index());
        // An element can't be on more than one boundary...
        size_t numBoundaries = pboundaries.count();
        // assert(numBoundaries < 2);
        if (numBoundaries > 1) {
            throw std::runtime_error("Boundary element on more than one cell face.");
        }

        isOnCellFace[be.index()] = numBoundaries > 0;
    }
    return isOnCellFace;
}

// Determine the periodic cell nodes that are identified with each other.
template<size_t N, class PointCollection>
void match(const PointCollection &bdryPoints,
        const BBox<VectorND<int(N)>> &cell,
        const std::vector<FaceMembership<N>> &faceMembership,
        std::vector<std::vector<size_t>>     &nodeSets,
        std::vector<size_t>                  &nodeSetForNode,
        Real epsilon = 1e-7)
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
                       << "for minimal boundary node " << i << " at " << p.transpose()
                       << "; looking for " << query.transpose() << std::endl;


                    double closestDist = std::numeric_limits<double>::max();
                    auto closestPt = bdryPoints.front();
                    for (const auto &pp : bdryPoints) {
                        double dist = (query - pp).norm();
                        if (dist < closestDist) {
                            closestPt = pp;
                            closestDist = dist;
                        }
                    }

                    ss << "Closest candidate at distance " << closestDist << ":\t";
                    ss << closestPt.transpose() << std::endl;

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

// Determine the periodic cell nodes that are identified with each other. This
// version permits mismatches, which can make sense for regular voxel grid
// meshes.
// Matching is slightly more complicated when permitting mismatches: for
// example, we cannot assume that if a node is present on any cell corner it
// will be present on the minimum corner. In other words, we cannot simply loop
// over minimal face nodes and identify them with every matched node as this
// will miss possible pairings not involving the minimal face nodes.
template<size_t N, class PointCollection>
void matchPermittingMismatch(const PointCollection &bdryPoints,
        const BBox<VectorND<int(N)>> &cell,
        const std::vector<FaceMembership<N>> &faceMembership,
        std::vector<std::vector<size_t>>     &nodeSets,
        std::vector<size_t>                  &nodeSetForNode,
        Real epsilon = 1e-7)
{
    assert(bdryPoints.size() == faceMembership.size());
    const size_t numBdryPts = bdryPoints.size();

    // Choose a cell size on the order of epsilon, but prevent cell sizes so
    // small as to cause index overflows for objects of size up to 100x100
    // centered at the origin: max int ~10^9 ==> cellSize > 10^-7
    CollisionGrid<Real, VectorND<N>> cgrid(std::max(epsilon, 1.0e-7));

    // Sparse collection of periodically-paired nodes for each node.
    // (VOLUME NODE INDICES!)
    // Unpaired nodes have no entry in this map, while paired nodes
    // map to an N-dim array with at least one entry that isn't NONE.
    std::map<size_t, std::array<size_t, N>> pair;

    for (size_t d = 0; d < N; ++d) {
        cgrid.reset();
        for (size_t i = 0; i < numBdryPts; ++i) {
            if (faceMembership[i].onMinFace(d))
                cgrid.addPoint(bdryPoints[i], i);
        }
        for (size_t i = 0; i < numBdryPts; ++i) {
            if (!faceMembership[i].onMaxFace(d)) continue;
            VectorND<N> query(bdryPoints[i]);
            query[d] = cell.minCorner[d];
            auto result = cgrid.getClosestPoint(query, epsilon);
            if (result.first == -1) continue; // mismatch!

            // Fill out (symmetric) pair adjacency information
            size_t pi = result.first;
            // Note: when mismatches occur, it's possible for an entry in "pair"
            // to already exist (from a previous d loop) for one of the nodes 
            // but not the other.
            // Initialize new entries where necessary.
            if (pair.count( i) == 0) pair[ i].fill(NONE);
            if (pair.count(pi) == 0) pair[pi].fill(NONE);

            size_t &ip = pair[ i][d],
                   &pp = pair[pi][d];
            if ((ip != NONE) || (pp != NONE))
                throw std::runtime_error("Non-bijective boundary matching");
            ip = pi;
            pp =  i;
        }
    }

    // Group each connected component of the paired node graph into a node set.
    std::queue<size_t> bfsQueue;
    nodeSetForNode.assign(numBdryPts, NONE);
    nodeSets.clear();
    size_t numMismatches = 0;
    for (size_t i = 0; i < numBdryPts; ++i) {
        if (nodeSetForNode[i] != NONE) continue; // visited?
        const size_t nsi = nodeSets.size();
        nodeSetForNode[i] = nsi;
        nodeSets.emplace_back(1, i);
        auto &ns = nodeSets.back();

        bfsQueue.push(i);
        while (!bfsQueue.empty()) {
            size_t u = bfsQueue.front();
            bfsQueue.pop();

            // Loop over adjacencies
            for (size_t d = 0; d < N; ++d) {
                if (!faceMembership[u].onMinOrMaxFace(d)) continue;
                size_t v = NONE;
                if (pair.count(u) == 1) v = pair[u][d];

                if (v == NONE) { ++numMismatches; continue; }
                size_t &nsv = nodeSetForNode[v];
                if (nsv != NONE) {
                    if (nsv != nsi) {
                        std::cerr << "node set conflict. " << nsv << " vs " << nsi << std::endl;
                        std::cerr << "u, v, i, d:\t"
                            << u << ", "
                            << v << ", "
                            << i << ", "
                            << d << std::endl;
                    }
                    assert(nsv == nsi);
                    continue;
                }
                nsv = nsi;
                ns.push_back(v);
                bfsQueue.push(v);
            }
        }
    }

    // All boundary points should have been assigned to node sets
    for (size_t i = 0; i < numBdryPts; ++i) {
        assert(nodeSetForNode[i] != NONE);
        assert(nodeSetForNode[i] < nodeSets.size());
    }

    if (numMismatches > 0)
        std::cerr << "WARNING: detected " << numMismatches << " mismatches in periodic node identification" << std::endl;
}

}

#endif /* end of include guard: PERIODICBOUNDARYMATCHER_HH */
