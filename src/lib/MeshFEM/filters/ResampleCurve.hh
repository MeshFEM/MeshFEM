////////////////////////////////////////////////////////////////////////////////
// ResampleCurve.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Clean a single connected, closed, simple curve by resampling it with points
//  evenly spaced in arclength along the curve.
//
//  To preserve feature corners, the curve is first segmented into a list of
//  smooth polylines terminating at feature vertices, and only these smooth
//  curves are resampled. Vertices at the intersection with the bounding box
//  "cell" are treated as feature vertices.
//
//  TODO: Variable length remeshing, merging of short segments (happens with
//  narrowly spaced feature vertices).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/20/2017 12:31:19
////////////////////////////////////////////////////////////////////////////////
#ifndef RESAMPLECURVE_HH
#define RESAMPLECURVE_HH

#include <stdexcept>
#include <list>
#include <vector>
#include <utility>
#include <tuple>
#include <cassert>
#include <iomanip>

#include <nonstd/optional.hpp>

#include <MeshFEM/PeriodicBoundaryMatcher.hh>
#include <MeshFEM/Geometry.hh>

bool isOutsideResamplingRegions(Point3D p, std::vector<Region<Point3D> *> inRegions, std::vector<Region<Point3D> *> outRegions) {

    // marking which points are filtered and which are not
    for (auto outRegion : outRegions) {
        if (outRegion->containsPoint(p))
            return true;
    }

    if (inRegions.size() > 0) {
        for (auto inRegion : inRegions) {
            if (inRegion->containsPoint(p))
                return false;
        }

        return true;
    }

    return false;
}

// Curve is given in order as curve[0], curve[1], ..., curve[len - 1], curve[0],
// and we clean it in-place.
// We take advantage of this representation to identify edge v0->v1 with vertex v0.
// cellBdryEdgeLen: if set, determines the resolution at which the boundary is
// meshed. Otherwise, the cell boundary edges are left unsubdivided.
template<size_t N>
void resampleCurve(std::list<VectorND<int(N)>> &curve,
                  const BBox<VectorND<int(N)>> &cell,
                  Real targetLen,
                  Real featureAngleThreshold,
                  nonstd::optional<Real> cellBdryEdgeLen = nonstd::optional<Real>(),
                  const std::vector<Real> &variableMinLen = std::vector<Real>(),
                  Real cellEpsilon = 1e-5,
                  bool noSkip = false) {
    if (variableMinLen.size() > 0)
        throw std::runtime_error("Curve resampling does not yet support spatial adaptivity.");
    using Point = VectorND<N>;
    using FMembership = PeriodicBoundaryMatcher::FaceMembership<N>;

    // Vertices are identified with their outgoing edge.
    using  Curve = std::list<Point>;
    using  VtxIt = typename Curve::iterator;

    // Circular access
    auto next = [&](VtxIt i) { ++i; if (i == curve.end()) return curve.begin(); return i; };
    auto prev = [&](VtxIt i) { if (i == curve.begin()) i = curve.end(); return --i; };
    auto segmentLen = [&](VtxIt s, VtxIt e) {
        Real len = 0;
		size_t subdivCount = 0;
        do {
            auto n = next(s);
            len += (*n - *s).norm();
            s = n;
			++subdivCount;
        } while (s != e);
        return std::make_tuple(len, subdivCount);
    };

    // Vertices are features if they have an incident angle significantly
    // different from pi.
    // We also consider vertices with greater face membership than one of
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

    // Begin traversal on a feature vertex, if any exists.
    VtxIt start = next(curve.begin());
    while (!isFeature(start) && (start != curve.begin())) {
        start = next(start);
    }

    VtxIt segmentStart = start;
    do {
        // Find the end of this segment (the next feature vertex, if one exists.)
        VtxIt segmentEnd = next(segmentStart);
        while (!isFeature(segmentEnd) && (segmentEnd != start))
            segmentEnd = next(segmentEnd);

        // Determine if this segment lies on a cell boundary
        auto segmentFM = FMembership(*segmentEnd, cell, cellEpsilon);
        for (VtxIt it = segmentStart; it != segmentEnd; it = next(it))
            segmentFM &= FMembership(*it, cell, cellEpsilon);

        // For segments on the cell boundary, either remesh at resolution
        // cellBdryEdgeLen, or leave untouched if cellBdryEdgeLen unspecified
        Real segmentTargetSpacing = targetLen;
        bool skip = false;

        // If noSkip is on, all edges are treated the same way (for non periodic case)
        if (segmentFM.onAnyFace() && !noSkip) {
            if (!cellBdryEdgeLen) skip = true;
            else segmentTargetSpacing = *cellBdryEdgeLen;
        }

        // Remesh this segment with a uniform spacing as close as possible
        // to, but not exceeding, segmentTargetSpacing.

        if (!skip) {
            Real slen;
            size_t currentEdgeCount;
            std::tie(slen, currentEdgeCount) = segmentLen(segmentStart, segmentEnd);

            // Never simplify so much that the polygon could become degenerate.
            size_t minSubdiv = std::min<size_t>(2, currentEdgeCount);
            size_t nSubdiv = std::max<size_t>(ceil(slen / segmentTargetSpacing), minSubdiv);
            Real spacing = slen / nSubdiv;

            // Build up the remeshed curve segment points in newPts
            std::vector<Point> newPts;
            newPts.reserve(nSubdiv - 1);
            Real distToNext = spacing;
            for (VtxIt it = segmentStart, nextIt; /* terminated inside */; it = nextIt) {
                // Loop invariant: distToNext stores the arclength we must travel before
                // placing the next point.
                nextIt = next(it);
                const Point &p = *it, &pn = *nextIt;

                const Real elen = (pn - p).norm();
                Real loc = 0; // position along current endge.
                // Add all resampled points falling on current edge
                while (loc + distToNext < elen) {
                    loc += distToNext;
                    distToNext = spacing;
                    Point pt = p + (loc / elen) * (pn - p);
                    if (newPts.size() == nSubdiv - 1) // In absence of numerical issues, we should generate exactly nSubdiv - 1
                        assert((pt - *segmentEnd).norm() < 1e-8);
                    else
                        newPts.emplace_back(pt);
                }
                distToNext -= elen - loc;
                if (nextIt == segmentEnd) break;
            }
            assert(newPts.size() == nSubdiv - 1);

#if 0
            std::cerr << std::setprecision(19);
            std::cerr << "Original pts:" << std::endl;
            for (VtxIt it = segmentStart; it != segmentEnd; it = next(it))
                std::cerr << "\t" << it->transpose() << std::endl;
            std::cerr << "New pts:" << std::endl;
            for (const auto &p : newPts)
                std::cerr << "\t" << p.transpose() << std::endl;
            std::cerr << std::endl;
#endif

            // Remove the existing vertices in (segmentStart, segmentEnd)
            while (next(segmentStart) != segmentEnd)
                curve.erase(next(segmentStart));

            // Insert the new vertices before segmentEnd.
            for (const auto &p : newPts)
                curve.insert(segmentEnd, p);
        }

        segmentStart = segmentEnd;
    } while (segmentStart != start);
}

// Same resampling technique but choosing resampling regions
template<size_t N>
void resampleCurve(std::list<VectorND<int(N)>> &curve,
                   const BBox<VectorND<int(N)>> &cell,
                   Real targetLen,
                   Real featureAngleThreshold,
                   const std::vector<Region<Point3<double>> *> remeshingRegions,
                   const std::vector<Region<Point3<double>> *> exceptRegions,
                   nonstd::optional<Real> cellBdryEdgeLen = nonstd::optional<Real>(),
                   const std::vector<Real> &variableMinLen = std::vector<Real>(),
                   Real cellEpsilon = 1e-5,
                   bool noSkip = false) {
    if (variableMinLen.size() > 0)
        throw std::runtime_error("Curve resampling does not yet support spatial adaptivity.");
    using Point = VectorND<N>;
    using FMembership = PeriodicBoundaryMatcher::FaceMembership<N>;

    // Vertices are identified with their outgoing edge.
    using  Curve = std::list<Point>;
    using  VtxIt = typename Curve::iterator;

    // Circular access
    auto next = [&](VtxIt i) { ++i; if (i == curve.end()) return curve.begin(); return i; };
    auto prev = [&](VtxIt i) { if (i == curve.begin()) i = curve.end(); return --i; };
    auto segmentLen = [&](VtxIt s, VtxIt e) {
        Real len = 0;
        size_t subdivCount = 0;
        do {
            auto n = next(s);
            len += (*n - *s).norm();
            s = n;
            ++subdivCount;
        } while (s != e);
        return std::make_tuple(len, subdivCount);
    };

    // Vertices are features if they have an incident angle significantly
    // different from pi.
    // We also consider vertices with greater face membership than one of
    // their neighbors to be features to discourage changing the shape of the
    // "connector" geometry.
    auto isFeature = [&](VtxIt v) -> bool {
        const Point &p = *v, &pn = *(next(v)), &pp = *(prev(v));
        Real theta = angle((pn - p).eval(), (pp - p).eval());
        if (std::abs(theta - M_PI) > featureAngleThreshold) return true;

        // If noSkip is on, all edges are treated the same way (for non periodic case)
        if (noSkip) return false;

        FMembership fmp(p, cell, cellEpsilon), fmpn(pn, cell, cellEpsilon), fmpp(pp, cell, cellEpsilon);
        if (!(fmp <= fmpn) || !(fmp <= fmpp)) return true;
        return false;
    };

    // Begin traversal on a feature vertex, if any exists.
    VtxIt start = next(curve.begin());
    while (!isFeature(start) && (start != curve.begin())) {
        start = next(start);
    }

    VtxIt segmentStart = start;
    do {
        // Find the end of this segment (the next feature vertex, if one exists.)
        bool startInRegion = isOutsideResamplingRegions(padTo3D(*segmentStart), remeshingRegions, exceptRegions);
        bool endInRegion = false;
        VtxIt segmentEnd = next(segmentStart);
        while ((segmentEnd != start)) {
            if (isFeature(segmentEnd))
                break;

            endInRegion = isOutsideResamplingRegions(padTo3D(*segmentEnd), remeshingRegions, exceptRegions);
            if (startInRegion && !endInRegion) {
                // It means we have a segment leaving a boundary condition (except region)
                // Our current curve should then not be resampled
                break;
            }
            else if (!startInRegion && endInRegion) {
                // It means we have a segment entering a boundary condition (except region)
                // If we take a step back, we can have a curve that is fully outiside the except region
                // and that could be resampled!
                segmentEnd = prev(segmentEnd);
                if (segmentEnd == segmentStart) {
                    segmentEnd = next(segmentStart);
                    startInRegion = true;
                }
                else {
                    endInRegion = false;
                    break;
                }
            }

            segmentEnd = next(segmentEnd);
        }

        // Determine if this segment lies on a cell boundary
        auto segmentFM = FMembership(*segmentEnd, cell, cellEpsilon);
        for (VtxIt it = segmentStart; it != segmentEnd; it = next(it))
            segmentFM &= FMembership(*it, cell, cellEpsilon);

        // For segments on the cell boundary, either remesh at resolution
        // cellBdryEdgeLen, or leave untouched if cellBdryEdgeLen unspecified
        Real segmentTargetSpacing = targetLen;
        bool skip = false;

        // Idea is to not remesh regions inside boundary conditions (except regions)
        if (startInRegion || endInRegion) {
            skip = true;
        } // noSkip means all faces are treated equaly and it does not matter if they are or not on base cell boundary
        else if (segmentFM.onAnyFace() && !noSkip) {
            if (!cellBdryEdgeLen) skip = true;
            else segmentTargetSpacing = *cellBdryEdgeLen;
        }

        // Remesh this segment with a uniform spacing as close as possible
        // to, but not exceeding, segmentTargetSpacing.

        if (!skip) {
            Real slen;
            size_t currentEdgeCount;
            std::tie(slen, currentEdgeCount) = segmentLen(segmentStart, segmentEnd);

            // Never simplify so much that the polygon could become degenerate.
            size_t minSubdiv = std::min<size_t>(2, currentEdgeCount);
            size_t nSubdiv = std::max<size_t>(ceil(slen / segmentTargetSpacing), minSubdiv);
            Real spacing = slen / nSubdiv;

            // Build up the remeshed curve segment points in newPts
            std::vector<Point> newPts;
            newPts.reserve(nSubdiv - 1);
            Real distToNext = spacing;
            for (VtxIt it = segmentStart, nextIt; /* terminated inside */; it = nextIt) {
                // Loop invariant: distToNext stores the arclength we must travel before
                // placing the next point.
                nextIt = next(it);
                const Point &p = *it, &pn = *nextIt;

                const Real elen = (pn - p).norm();
                Real loc = 0; // position along current endge.
                // Add all resampled points falling on current edge
                while (loc + distToNext < elen) {
                    loc += distToNext;
                    distToNext = spacing;
                    Point pt = p + (loc / elen) * (pn - p);
                    if (newPts.size() == nSubdiv - 1) // In absence of numerical issues, we should generate exactly nSubdiv - 1
                        assert((pt - *segmentEnd).norm() < 1e-8);
                    else
                        newPts.emplace_back(pt);
                }
                distToNext -= elen - loc;
                if (nextIt == segmentEnd) break;
            }
            assert(newPts.size() == nSubdiv - 1);

#if 0
            std::cerr << std::setprecision(19);
            std::cerr << "Original pts:" << std::endl;
            for (VtxIt it = segmentStart; it != segmentEnd; it = next(it))
                std::cerr << "\t" << it->transpose() << std::endl;
            std::cerr << "New pts:" << std::endl;
            for (const auto &p : newPts)
                std::cerr << "\t" << p.transpose() << std::endl;
            std::cerr << std::endl;
#endif

            // Remove the existing vertices in (segmentStart, segmentEnd)
            while (next(segmentStart) != segmentEnd)
                curve.erase(next(segmentStart));

            // Insert the new vertices before segmentEnd.
            for (const auto &p : newPts)
                curve.insert(segmentEnd, p);
        }

        segmentStart = segmentEnd;
    } while (segmentStart != start);
}

#endif /* end of include guard: RESAMPLECURVE_HH */
