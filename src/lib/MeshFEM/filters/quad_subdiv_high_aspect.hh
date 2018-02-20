////////////////////////////////////////////////////////////////////////////////
// quad_subdiv_high_aspect.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Split planar rectangular quads to make them more square:
//      3---------2     3----m1---2
//      |         | ==> |    |    |
//      0---------1     0----m0---1
//      We try to split quads with aspect ratio above a certain treshold while
//      still maintaining a valid quad-tri mesh. This means we can't split a
//      quad if its affected neighbors don't also want to split the shared
//      edge (i.e. if the number of elements wanting it split is 1)
//
//      The algorithm is as follows:
//          1) Determine which quads want to split and count how many elements
//             adjacent each edge want to split the edge.
//          2) BFS-style conflict resolution:
//              Fill a queue with edges having wantSplitCount = 1
//              While queue is nonempty:
//                  take first edge, check wantSplitCount is still 1
//                  set split = false on e, the element that wanted to split it
//                  decrement wantSplitCount on the edges e wanted to split
//                  if an edge's wantSplitCount hits 1, add it to the queue
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/27/2014 18:45:58
////////////////////////////////////////////////////////////////////////////////
#ifndef QUAD_SUBDIV_HIGH_ASPECT_HH
#define QUAD_SUBDIV_HIGH_ASPECT_HH
#include <MeshFEM/Geometry.hh>
#include <map>
#include <vector>
#include <array>
#include <queue>
#include <limits>

struct QSEdgeData {
    // Edge is only created when an element wants to split it, so init count = 1
    QSEdgeData(size_t elem) : midpointIndex(std::numeric_limits<size_t>::max()) {
        addSplittingElement(elem);
    }

    void addSplittingElement(size_t elem) {
        assert(wantSplitCount() < 2);
        m_splittingElements[wantSplitCount()] = elem;
        ++m_wantSplitCount;
    }

    // elem must be one of the current splittingElements
    void removeSplittingElement(size_t elem) {
        size_t idx = 3;
        if      (m_splittingElements[0] == elem) idx = 0;
        else if (m_splittingElements[1] == elem) idx = 1;
        assert(idx < wantSplitCount());
        if (idx == 0) m_splittingElements[0] = m_splittingElements[1];
        --m_wantSplitCount;
    }

    size_t splittingElement(size_t i) const {
        assert(i < m_wantSplitCount);
        return m_splittingElements[i];
    }

    size_t wantSplitCount() const { return m_wantSplitCount; }

    size_t midpointIndex; // Index of the vertex created for this edge
private:
    size_t m_wantSplitCount = 0; // How many elements want to split this edge (0, 1, or 2)
    std::array<size_t, 2> m_splittingElements; // which elements want to split the edge
};

struct QSElementData {
    QSElementData() : splitPair(-1) { }
    int splitPair;  // Which pair of edges wants to split:
                    //     -1: no split
                    //      0: edges 0->1, 2->3
                    //      1: edges 1->2, 3->0
    int blockSplitPair; // TODO: 3 states: wants split, neutral, blocks split
    bool wantsSplit() const { return splitPair == 0 || splitPair == 1; }
    // Get the edges this element wants to split 
    template<class Element>
    void getSplitEdges(const Element &e, UnorderedPair &e0,
                       UnorderedPair &e1) const {
        assert(e.size() == 4);
        assert(wantsSplit());
        e0.set(e[splitPair + 0], e[ splitPair + 1     ]);
        e1.set(e[splitPair + 2], e[(splitPair + 3) % 4]);
    }
};

// quadIdx: index of the quad from which each output element originated
//          This can be propagated across several subdivisions by passing the
//          same array for each call.
// return: true if a subdivision was performed (so we can iterate)
template<class Vertex, class Element>
bool quad_subdiv_high_aspect(
        const std::vector<Vertex>  &inVertices, const std::vector<Element> &inElements,
        std::vector<Vertex>  &outVertices, std::vector<Element> &outElements,
        std::vector<size_t> &quadIdx, Real aspectThreshold = 2)
{
    outVertices = inVertices;
    outElements.clear(), outElements.reserve(4 * inElements.size());

    std::vector<size_t> oldQuadIdx(quadIdx);
    if (oldQuadIdx.size() == 0) {
        for (size_t i = 0; i < inElements.size(); ++i)
            oldQuadIdx.push_back(i);
    }
    if (oldQuadIdx.size() != inElements.size())
        throw std::runtime_error("Invalid quadIdx");
    quadIdx.clear(), quadIdx.reserve(4 * inElements.size());

    if (aspectThreshold <= sqrt(2) + 1e-8)
        throw std::runtime_error("Aspect ratio threshold must be > sqrt(2) for improvement/convergence");

    Element newQuad(4);

    // 1) Determine which quads want to split and count how many elements
    //    adjacent each edge want to split the edge.
    std::vector<QSElementData> elemData(inElements.size());
    std::map<UnorderedPair, QSEdgeData> edgeData;
    for (size_t i = 0; i < inElements.size(); ++i) {
        auto e = inElements[i];
        if (e.size() != 4) continue;
        auto &d = elemData[i];
        // TODO: check for non-planar and non-rectangular cases!!!
        Point3D e0 = Point3D(inVertices[e[1]].point) - Point3D(inVertices[e[0]].point);
        Point3D e1 = Point3D(inVertices[e[2]].point) - Point3D(inVertices[e[1]].point);
        if (e0.norm() > (aspectThreshold * e1.norm())) d.splitPair = 0;
        if (e1.norm() > (aspectThreshold * e0.norm())) d.splitPair = 1;
        if (!d.wantsSplit()) continue;
        UnorderedPair edges[2];
        d.getSplitEdges(e, edges[0], edges[1]);
        for (size_t ei = 0; ei < 2; ++ei) {
            auto it = edgeData.find(edges[ei]);
            if (it == edgeData.end())
                edgeData.insert(std::make_pair(edges[ei], QSEdgeData(i)));
            else it->second.addSplittingElement(i);
        }
    }

    // 2) BFS-style conflict resolution
    std::queue<QSEdgeData *> edgeQueue;
    for (auto &entry :  edgeData) {
        if (entry.second.wantSplitCount() == 1)
            edgeQueue.push(&(entry.second));
    }
    while (!edgeQueue.empty()) {
        QSEdgeData *ed = edgeQueue.front();
        edgeQueue.pop();
        if (ed->wantSplitCount() != 1) continue; // already resolved
        size_t e = ed->splittingElement(0);
        QSElementData &preventedElemData = elemData.at(e);
        assert(preventedElemData.wantsSplit());
        UnorderedPair edges[2];
        // if (!preventedElemData.wantsSplit()) continue; // already resolved
        preventedElemData.getSplitEdges(inElements.at(e), edges[0], edges[1]);
        preventedElemData.splitPair = -1; // Can't split :(
        // Decrement edges' wantSplitCount, adding to the queue if they hit 1
        for (size_t i = 0; i < 2; ++i) {
            auto &edgeDat = edgeData.at(edges[i]);
            edgeDat.removeSplittingElement(e);
            if (edgeDat.wantSplitCount() == 1)
                edgeQueue.push(&edgeDat);
        }
    }

    bool subdivided = false;
    for (size_t i = 0; i < inElements.size(); ++i) {
        auto e = inElements[i];
        auto ed = elemData[i];
        if (e.size() != 4 || !ed.wantsSplit()) {
            quadIdx.push_back(oldQuadIdx[i]);
            outElements.push_back(e);
            continue;
        }

        subdivided = true;

        // splitPair is an index offset that effectively rotates our picture
        // by 90 degress to always look like:
        //      3---------2     3----m1---2
        //      |         | ==> | q0 | q1 |
        //      0---------1     0----m0---1

        // Midpoint vertices
        Point3D m[2] = { (Point3D(inVertices[e[0 + ed.splitPair]].point) + Point3D(inVertices[e[ 1 + ed.splitPair     ]].point)) / 2,
                         (Point3D(inVertices[e[2 + ed.splitPair]].point) + Point3D(inVertices[e[(3 + ed.splitPair) % 4]].point)) / 2, };

        // Look up or generate new midpoint vertices.
        size_t midx[2];
        UnorderedPair edges[2];
        ed.getSplitEdges(e, edges[0], edges[1]);
        for (size_t c = 0; c < 2; ++c) {
            midx[c] = edgeData.at(edges[c]).midpointIndex;
            if (midx[c] == std::numeric_limits<size_t>::max()) {
                midx[c] = outVertices.size();
                outVertices.push_back(m[c]);
                edgeData.at(edges[c]).midpointIndex = midx[c];
            }
            else { assert(midx[c] < outVertices.size()); }
        }

        // Generate both new quads in ccw order
        for (size_t q = 0; q < 2; ++q) {
            newQuad[0] = e[2 * q + ed.splitPair];
            newQuad[1] = midx[q];
            newQuad[2] = midx[(q + 1) % 2];
            newQuad[3] = e[(2 * q + 3 + ed.splitPair) % 4];
            outElements.push_back(newQuad);
            quadIdx.push_back(oldQuadIdx[i]);
        }
    }

    return subdivided;
}

#endif /* end of include guard: QUAD_SUBDIV_HIGH_ASPECT_HH */
