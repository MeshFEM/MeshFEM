////////////////////////////////////////////////////////////////////////////////
// CurveCleanup.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Clean a single connected, closed, simple curve:
//           1) Collapsing short edges
//           2) Splitting   long edges
//      Operations are performed in this order, so if splitting creates short
//      edges, the new edges will not be collapsed.
//
//      The operations preserve the topology of a periodically or reflectively
//      tiled mesh by constraining vertices on the bbox min/max faces from
//      moving off those faces.
//      If "periodic = true" is passed, the axis-aligned periodicity of the
//      discrete curve mesh itself is also preserved by ensuring paired vertices
//      are collapsed/split simultaneously.
//
//      "Sharp features" are preserved by ensuring no feature vertex (vertex
//      with angle differing significantly from pi) is merged into a non-feature
//      vertex. However, we never let this criterion prevent a merge: two
//      adjacent sharp features are merged to avoid robustness issues with
//      jagged boundaries. Also, we allow feature vertices adjacent to the
//      periodic boundary to merge into the boundary.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/03/2016 15:29:15
////////////////////////////////////////////////////////////////////////////////
#ifndef CURVECLEANUP_HH
#define CURVECLEANUP_HH

#include <stdexcept>
#include <list>
#include <vector>
#include <random>
#include <utility>

#include <nonstd/optional.hpp>

#include <MeshFEM/filters/extract_polygons.hh>
#include <MeshFEM/PeriodicBoundaryMatcher.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Utilities/IteratorMap.hh>
#include <MeshFEM/Utilities/RandomAccessIndexSet.hh>
#include <MeshFEM/Geometry.hh>

// Clean a closed polygonal curve.
// Curve is given in order as curve[0], curve[1], ..., curve[len - 1], curve[0],
// and we clean it in-place.
// We take advantage of this representation to identify edge v0->v1 with vertex v0.
// cellBdryEdgeLen: if set, determines the resolution at which the boundary is
// meshed. Otherwise, the cell boundary edges are left unsubdivided.
//
// permitFeatureMerge: whether to allow a short edge connecting two feature
// vertices to merge.
template<size_t N>
void curveCleanup(std::list<VectorND<int(N)>> &curve,
                  const BBox<VectorND<int(N)>> &cell,
                  Real minLen, Real maxLen,
                  Real featureAngleThreshold, bool periodic = false,
                  nonstd::optional<Real> cellBdryEdgeLen = nonstd::optional<Real>(),
                  const std::vector<Real> &variableMinLen = std::vector<Real>(),
                  Real cellEpsilon = 1e-5, const bool permitFeatureMerge = true) {
    // std::cout << "Simplifying curve of len " << curve.size() << std::endl;
    using Point = VectorND<N>;
    using FMembership = PeriodicBoundaryMatcher::FaceMembership<N>;
    static constexpr size_t NO_PAIR = std::numeric_limits<size_t>::max();

    // Vertices are identified with their outgoing edge.
    using  Curve = std::list<Point>;
    using  VtxIt = typename Curve::iterator;
    using EdgeIt = typename Curve::iterator;

    // Determine periodically-identified edge pairs if periodicity preservation
    // is requested. We only handle the 2D case for now, where a boundary edge
    // has exactly one pair.
    // We ensure that paired edges are always collapsed (erased from curve)
    // simultaneously, so both keys and values of "pair" are valid iterators.
    IteratorMap<EdgeIt, EdgeIt> pair;
    if (periodic) {
        if (N != 2) throw std::runtime_error("Periodic curve cleanup only implemented for 2D");

        std::vector<FMembership> fm;
        std::vector<std::vector<size_t>> nodeSets;
        std::vector<size_t>              nodeSetForNode;
        PeriodicBoundaryMatcher::determineCellBoundaryFaceMembership(curve, cell, fm, cellEpsilon);
        PeriodicBoundaryMatcher::match(curve, cell, fm, nodeSets, nodeSetForNode, cellEpsilon);

        // (Temporary) constant-time random access to list iterators
        std::vector<EdgeIt> edges;
        edges.reserve(curve.size());
        for (auto ei = curve.begin(); ei != curve.end(); ++ei)
            edges.push_back(ei);
        for (size_t ei = 0; ei < curve.size(); ++ei) {
            size_t v0i = ei;
            size_t v1i = (ei + 1) % curve.size();
            // Determine which periodic face the edge is on, if any. Since we're only
            // implementing the 2D case, there should be only one.
            auto eFM = fm[v0i] & fm[v1i];
            if (eFM.count() > 1) throw std::runtime_error("ERROR: edge on more than one periodic cell face.");
            if (eFM.count() == 0) continue;

            // The periodic pair edge should be in the opposite orientation
            // (v1i_pair -> v0i_pair)
            // If v0i or v1i are on a corner, there are multiple pair candidates to check.
            size_t ei_pair = NO_PAIR;
            for (size_t v1i_pair : nodeSets[nodeSetForNode[v1i]]) {
                for (size_t v0i_pair : nodeSets[nodeSetForNode[v0i]]) {
                    if (v0i_pair == (v1i_pair + 1) % curve.size()) {
                        assert(ei_pair == NO_PAIR);
                        ei_pair = v1i_pair;
                    }
                }
            }
            if (ei_pair == NO_PAIR) throw std::runtime_error("Couldn't find periodic-matching edge.");
            pair[edges.at(ei)] = edges.at(ei_pair);
        }

        // Verify all periodic edges are paired and the pairing is symmetric.
        for (size_t ei = 0; ei < edges.size(); ++ei) {
            bool isPeriodicEdge = (fm[ei] & fm[(ei + 1) % curve.size()]).count();
            if (!isPeriodicEdge) continue;
            auto e = edges.at(ei);
            if (pair.count(e) == 0) throw std::runtime_error("Unpaired edge found.");
            if (pair.at(pair.at(e)) != e) throw std::runtime_error("Inconsistent edge pairing");
        }
    }

    // Circular access
    auto next = [&](VtxIt i) { ++i; if (i == curve.end()) return curve.begin(); return i; };
    auto prev = [&](VtxIt i) { if (i == curve.begin()) i = curve.end(); return --i; };

    auto edgeLen = [&](EdgeIt ei) -> Real { return (*next(ei) - *ei).norm(); };

    // Vertices are features if they have an incident angle significantly
    // different from pi.
    // We also consider vertices with greater face membership than either of
    // their neighbors to be features to discourage changing the shape of the
    // "connector" geometry.
    auto isFeature = [&](VtxIt v) -> bool {
        const Point &p = *v, &pn = *(next(v)), &pp = *(prev(v));
        Real theta = angle((pn - p).eval(), (pp - p).eval());
        if (std::abs(theta - M_PI) > featureAngleThreshold) return true;
        FMembership fmp(p, cell, cellEpsilon), fmpn(pn, cell, cellEpsilon), fmpp(pp, cell, cellEpsilon);
        if (!(fmp <= fmpn) || !(fmp <= fmpp)) return true;
        return false;
    };

    // Consider variable (per-edge) minLength when available
    bool hasVariableMinLen = (variableMinLen.size() == curve.size());
    // std::cout << "hasVariableMinLen: " << hasVariableMinLen << std::endl;
    // std::cout << "variableMinLen.size(): " << variableMinLen.size() << std::endl;
    // std::cout << "curve.size(): " << variableMinLen.size() << std::endl;
    auto getMinLen = [&](size_t idx) {
        if (!hasVariableMinLen) return minLen;
        return variableMinLen.at(idx);
    };

    // allEdges[i]: the iterator for edge with index i (warning: contains invalidated edges)
    // edgeIndex[ei]: the index for edge iterator ei
    // shortEdges:    set of indices for short edges
    std::vector<EdgeIt> allEdges; allEdges.reserve(curve.size());
    IteratorMap<EdgeIt, size_t> edgeIndex;
    RandomAccessIndexSet shortEdges(curve.size());

    for (auto ei = curve.begin(); ei != curve.end(); ++ei) {
        size_t idx = allEdges.size();
        edgeIndex.emplace(ei, idx);
        if (edgeLen(ei) < getMinLen(idx)) shortEdges.insert(idx);
        allEdges.push_back(ei);
    }

    // std::cout << shortEdges.size() << " of " << allEdges.size() << " edges are short" << std::endl;

    // Collapse operation: collapse the edge associated with "tail" into
    // collapsePt by moving tip to collapsePt and deleting tail.
    auto collapse = [&](VtxIt tail, Point &collapsePt) {
        VtxIt tip = next(tail);
        *tip = collapsePt;
        edgeIndex.erase(tail);
        curve.erase(tail);

        // The lengths of neighboring edges have changed--add any newly created
        // short edges to the short edge queue.
        EdgeIt eprev = prev(tip);
        size_t  tipIdx = edgeIndex.at(tip),
               prevIdx = edgeIndex.at(eprev);
        if (edgeLen(  tip) < getMinLen(tipIdx))   shortEdges.insert(tipIdx);
        if (edgeLen(eprev) < getMinLen(prevIdx)) shortEdges.insert(prevIdx);
    };

    std::default_random_engine generator;
    // std::cout << "initial curve size: " << curve.size() << std::endl;
    // std::cout << "numshort edges: " << shortEdges.size() << std::endl;
    // bool verbose = false;
    while (!shortEdges.empty()) {
        if (curve.size() <= 4) {
            std::cerr << "WARNING: curve has become extremely short. Bailing out of collapse." << std::endl;
            break;
        }

        // Choose a random short edge to improve quality in case of many merges.
        // If edges are merged in order, the merge point will "walk" around the
        // curve, causing much more extreme merge than desired.
        size_t loc = 0;
        if (shortEdges.size() > 1) {
            std::uniform_int_distribution<int> distribution(0, shortEdges.size() - 1);
            loc = distribution(generator);
        }

        size_t seIdx = shortEdges.indexAtLocation(loc);
        EdgeIt se = allEdges.at(seIdx);
        shortEdges.removeIndexAtLocation(loc);

        // Previous collapses could have lengthened se above the collapse
        // threshold--skip it in this case.
        if (edgeLen(se) >= getMinLen(seIdx)) continue;

        VtxIt v0 = se;
        VtxIt v1 = next(se);

        ////////////////////////////////////////////////////////////////////////
        // Determine the collapse point location based on mergeability criteria.
        ////////////////////////////////////////////////////////////////////////
        // Can we merge 0 -> 1 or 1 -> 0 without changing the tiled topology?
        // v0 can be merged into v1 if v0 is on a subset of the cell faces v1
        // is on. The exception is when this merge would create a non-manifold
        // vertex in the tiled mesh.
        // This exceptional case happens when neither of the two adjacent edges
        // lies on the same cell face.
        // These are hard constraints; never perform merges that violate them.
        FMembership fm0(*v0, cell, cellEpsilon), fm1(*v1, cell, cellEpsilon);
        bool merge01 = (fm0 <= fm1), merge10 = (fm1 <= fm0);

        // Prevent a non-manifold vertex from forming
        VtxIt v2 = next(v1), v_m1 = prev(v0);
        FMembership fm2(*v2, cell, cellEpsilon), fm_m1(*v_m1, cell, cellEpsilon);
        // Determine what cell boundaries this edge and its neighbors lie on
        FMembership       seFM = fm0 & fm1,
                    nextEdgeFM = fm1 & fm2,
                    prevEdgeFM = fm0 & fm_m1;
        // If neither neighbor is on the same cell face, block collapse
        if ((seFM != nextEdgeFM) && (seFM != prevEdgeFM))
            merge01 = merge10 = false;

        // We try to avoid merging a feature vertex into another vertex.
        // However, we never let the existence of features prevent a merge:
        // 1) If the topology preservation constraint forces only one direction
        //    of merging, we always take it (feature vertices near a periodic
        //    boundary merge into the periodic boundary)
        // 2) We merge adjacent feature vertices to their midpoint.
        if (merge01 && merge10) {
            // Allow features to block merges
            merge01 = !isFeature(v0);
            merge10 = !isFeature(v1);
            // If both vertices are features (or both are not) allow the merge
            if ((merge01 == merge10) && permitFeatureMerge)
                merge01  = merge10 = true;
        }

        // Merge into midpoint, vertex 0, or vertex 1 location
        Point collapsePt;
        if (merge01 && merge10) collapsePt = 0.5 * (*v0 + *v1);
        else if (merge01)       collapsePt = *v1;
        else if (merge10)       collapsePt = *v0;
        else                    continue; // no merge possible

        ////////////////////////////////////////////////////////////////////////
        // Perform the collapse (but first do the periodically-corresponding
        // collapse if we need to preserve periodicity).
        ////////////////////////////////////////////////////////////////////////
        if (periodic && seFM.count()) {
            // Since we only support the 2D case, the edge should be on only a
            // single period cell face.
            if (seFM.count() > 1) throw std::runtime_error("ERROR: short edge on more than one periodic cell face.");

            // By periodicity, the periodic pair should also need collapsing;
            // do it now to maintain periodicity.
            auto pse_map_iter = pair.find(se);
            if (pse_map_iter == pair.end()) throw std::runtime_error("Couldn't find periodic edge pair!");
            EdgeIt pse = pse_map_iter->second;
            size_t pseIdx = edgeIndex.at(pse);
            if (!shortEdges.contains(pseIdx)) throw std::runtime_error("ERROR: periodic pair of a short edge isn't short!");

            // Then collapse to the periodic pair of collapsePt:
            Point pairCollapsePt = collapsePt;
            bool flip = false;
            for (size_t d = 0; d < N; ++d) {
                if (seFM.onMinFace(d)) { pairCollapsePt[d] = cell.maxCorner[d]; flip = true; break; }
                if (seFM.onMaxFace(d)) { pairCollapsePt[d] = cell.minCorner[d]; flip = true; break; }
            }
            assert(flip);

            // Remove the short edge record
            shortEdges.remove(pseIdx);

            // Remove from our iterator collections the iterators that are about
            // to be erased (invalidated)
            pair.erase(pse_map_iter);
            auto paired_pse_map_iter = pair.find(pse);
            if (paired_pse_map_iter == pair.end()) throw std::runtime_error("Couldn't find paired edge's periodic pair!");
            pair.erase(paired_pse_map_iter);

            collapse(pse, pairCollapsePt);
        }

        collapse(se, collapsePt);

#if 0
        {
            static size_t iter = 0;
            std::vector<MeshIO::IOVertex> collapsedVertices;
            std::vector<MeshIO::IOElement> collapsedEdges;
            for (const auto &p : curve) {
                collapsedEdges.emplace_back(collapsedVertices.size(), collapsedVertices.size() + 1);
                collapsedVertices.emplace_back(p);
            }
            collapsedEdges.back()[1] = 0;
            MeshIO::save("collapsed_" + std::to_string(++iter) + ".msh", collapsedVertices, collapsedEdges);
        }
#endif
    }

    // Splitting is much easier--periodicity is automatically maintained, and
    // iterators are not invalidated.
    // Subdivide each long edge into segments of length geomMean(minLen, maxLen)
    // Only subdivide cell boundary edges if the cellBdryEdgeLen option was
    // passed.
    std::vector<EdgeIt> longEdges;
    std::vector<bool>   isLECellBoundaryEdge;
    {
        size_t i = 0;
        std::vector<FMembership> fm;
        PeriodicBoundaryMatcher::determineCellBoundaryFaceMembership(curve, cell, fm, cellEpsilon);
        for (auto ei = curve.begin(); ei != curve.end(); ++ei, ++i) {
            if (edgeLen(ei) > maxLen) {
                longEdges.push_back(ei);
                isLECellBoundaryEdge.push_back(
                    ((fm.at(i) & fm.at((i + 1) % curve.size())).onAnyFace()));
            }
        }
    }

    for (size_t li = 0; li < longEdges.size(); ++li) {
        EdgeIt ei = longEdges[li];
        Real targetLen = sqrt(getMinLen(edgeIndex.at(ei)) * maxLen);
        if (isLECellBoundaryEdge[li]) {
            if (cellBdryEdgeLen) targetLen = *cellBdryEdgeLen;
            else                 continue;
        }
        size_t nSubdiv = ceil(edgeLen(ei) / targetLen);

        const auto &p0 = *ei;
        const auto &p1 = *next(ei);
        EdgeIt it = ei;
        it = next(it);
        for (size_t i = 1; i < nSubdiv; ++i) {
            Real alpha = Real(i) / nSubdiv;
            curve.insert(it, (1 - alpha) * p0 + alpha * p1);
        }
    }
}

// Convenience version--operate on vector instead of list.
template<size_t N, typename... Args>
void curveCleanup(std::vector<VectorND<int(N)>> &curve,
                  BBox<VectorND<int(N)>> &cell,
                  Args&&... args)
{
    using Point = VectorND<int(N)>;
    std::list<Point> curveList;
    for (const Point &p : curve) curveList.push_back(p);
    curveCleanup(curveList, cell, std::forward<Args>(args)...);

    curve.clear();
    curve.reserve(curveList.size());
    for (const Point &p : curveList)
        curve.emplace_back(p);
}

// Convenience version--operate on line soup, preserving bbox "cell"
template<size_t N, class PointType, class EdgeType, typename... Args>
void curveCleanup(const std::vector<PointType> &inVertices,
                  const std::vector<EdgeType> &inElements,
                  std::vector<MeshIO::IOVertex> &outVertices,
                  std::vector<MeshIO::IOElement> &outElements,
                  const BBox<VectorND<int(N)>> &cell,
                  Args&&... args)
{
    std::list<std::list<VectorND<N>>> polygons;
    extract_polygons<N>(inVertices, inElements, polygons);

    outVertices.clear(), outElements.clear();
    for (auto &poly : polygons) {
        curveCleanup<N>(poly, cell, std::forward<Args>(args)...);
        size_t offset = outVertices.size();
        for (const auto &p : poly) {
            outElements.emplace_back(outVertices.size(), outVertices.size() + 1);
            outVertices.emplace_back(p);
        }
        outElements.back()[1] = offset;
    }
}

// Convenience version--operate on line soup, inferring dimension.
template<class PointType, class EdgeType, typename... Args>
void curveCleanup(const std::vector<PointType> &inVertices,
                  const std::vector<EdgeType> &inElements,
                  std::vector<MeshIO::IOVertex> &outVertices,
                  std::vector<MeshIO::IOElement> &outElements,
                  Args&&... args)
{
    // Infer dimension from bounding box.
    BBox<Point3D> bbox(inVertices);
    if (bbox.dimensions()[2] < 1e-8) {
        BBox<Point2D> bbox2D(inVertices);
        curveCleanup<2>(inVertices, inElements, outVertices, outElements, bbox2D, std::forward<Args>(args)...);
    }
    else {
        curveCleanup<3>(inVertices, inElements, outVertices, outElements,   bbox, std::forward<Args>(args)...);
    }
}

#endif /* end of include guard: CURVECLEANUP_HH */
