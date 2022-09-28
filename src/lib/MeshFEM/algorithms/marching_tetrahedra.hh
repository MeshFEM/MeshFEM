////////////////////////////////////////////////////////////////////////////////
// marching_tetrahedra.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Code to extract contours and sub-levelsets of a piecewise linear scalar
//  field defined on a tetrahedral mesh.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/29/2022 17:26:14
////////////////////////////////////////////////////////////////////////////////
#ifndef MARCHING_TETRAHEDRA_HH
#define MARCHING_TETRAHEDRA_HH

#include <vector>
#include <queue>
#include <limits>
#include <MeshFEM/SimplicialMesh.hh>
#include <MeshFEM/MeshIO.hh>

// va, vb, barycentric coordinates--useful for sampling functions at the
using ContourSamplePtInfo = std::tuple<size_t, size_t, double>;

// Extract the contour `f(x) == 0` or the sublevel set `f(x) <= 0` (depending on `sublevelset`)
// for the piecewise linear scalar field `f` defined over tetrahedral mesh `m`.
// If `lerp` is `true`, linear interpolation is used to place the sample point on each edge;
// otherwise the edge midpoint is used.
// Returns the number of triangles making up the contour itself (which is
// `outElements.size()` when `sublevelset == false`).
template<class Mesh>
std::enable_if_t<Mesh::K == 3, size_t> marching_tetrahedra(const Mesh &m, const Eigen::VectorXd &f,
            std::vector<MeshIO::IOVertex> &outVertices, std::vector<MeshIO::IOElement> &outElements,
            std::vector<ContourSamplePtInfo> &outSampleInfo,
            bool sublevelset = true, bool lerp = true)
{
    if (size_t(f.size()) != m.numVertices())
        throw std::runtime_error("Expected piecewise linear scalar field `f`");

    outVertices.clear();
    outElements.clear();
    outSampleInfo.clear();

    constexpr size_t NONE = std::numeric_limits<size_t>::max();
    std::vector<size_t> outIdxForVtx(m.numVertices(), NONE);
    std::map<UnorderedPair, size_t> vtxForEdge;

    auto insertEdgeVertex = [&](size_t va, size_t vb) -> size_t {
        auto it = vtxForEdge.find(UnorderedPair(va, vb));
        if (it != vtxForEdge.end()) return it->second;
        size_t outIdx = outVertices.size();

        // f(t) = (1 - t) * f_a + t * f_b == 0
        //      ==> t = f_a / (f_a - f_b)
        double t = 0.5;
        if (lerp) t = f[va] / (f[va] - f[vb]);

        auto pa = m.node(va)->p;
        auto pb = m.node(vb)->p;
        outVertices.push_back(((1 - t) * pa + t * pb).eval());
        outSampleInfo.emplace_back(va, vb, t);

        return outIdx;
    };

    for (auto e : m.elements()) {
        std::vector<size_t> inside_idx, outside_idx;
        for (auto v : e.vertices()) {
            if (f[v.index()] <= 0) inside_idx.push_back(v.localIndex());
            else                  outside_idx.push_back(v.localIndex());
        }
        size_t numOutside = outside_idx.size();

        if ((numOutside == 0) || (numOutside == 4)) continue;

        // The case with 1 vertex inside generates the same single
        // triangle as the case with one vertex outside, but with
        // flipped orientation.
        bool flip = false;
        if (numOutside == 3) {
            std::swap(inside_idx, outside_idx);
            numOutside = 1;
            flip = true;
        }

        if (numOutside == 1) {
            //       v
            //      (+)
            //      / \`.
            //     /   \ `(-)
            //    / _.--\ /
            //  (-)-----(-)
            auto v  = e.vertex(outside_idx[0]);
            auto hf = e.halfFace(outside_idx[0]); // outside face opposite v
            auto &tri = outElements.emplace_back();
            for (auto v_opp : hf.vertices())
                tri.push_back(insertEdgeVertex(v.index(), v_opp.index()));
            if (flip) std::swap(tri[0], tri[1]);
        }
        else if (numOutside == 2) {
            //          v1
            //         (+)
            //         / \`d
            //        /   c `(-)  ^
            //       / _.a-\-/   / he = e.halfedge(v1, v2)
            //  v2 (+)--b--(-)
            auto v1 = e.vertex(outside_idx[0]);
            auto v2 = e.vertex(outside_idx[1]);

            auto he = e.halfEdge(outside_idx[0], outside_idx[1]);

            size_t va = insertEdgeVertex(he.tip() .index(), v2.index());
            size_t vb = insertEdgeVertex(he.tail().index(), v2.index());
            size_t vc = insertEdgeVertex(he.tail().index(), v1.index());
            size_t vd = insertEdgeVertex(he.tip() .index(), v1.index());

            outElements.emplace_back(va, vb, vc);
            outElements.emplace_back(vc, vd, va);
        }
        else {
            throw std::logic_error("Impossible numOutside: " + std::to_string(numOutside));
        }
    }

    size_t numContourTris = outElements.size();

    if (sublevelset) {
        auto insertOrigVertex = [&](size_t origIdx) -> size_t {
            size_t outIdx = outIdxForVtx[origIdx];
            if (outIdx == NONE) {
                outIdxForVtx[origIdx] = outIdx = outVertices.size();
                outVertices.push_back(m.node(origIdx)->p);
                outSampleInfo.emplace_back(origIdx, 0, 0.0);
            }
            return outIdx;
        };

        // Insert the triangle corresponding to half face `hf` if it lies on the
        // boundary. (Used in the `sublevelset` case.)
        auto insertBoundaryFace = [&](auto bhf) {
            auto &bdryTri = outElements.emplace_back();
            for (auto bv : bhf.vertices())
                bdryTri.push_back(insertOrigVertex(bv.volumeVertex().index()));
        };

        for (auto be : m.boundaryElements()) {
            std::vector<size_t> inside_idx, outside_idx;
            for (auto bv : be.vertices()) {
                auto v = bv.volumeVertex();
                if (f[v.index()] <= 0) inside_idx.push_back(bv.localIndex());
                else                  outside_idx.push_back(bv.localIndex());
            }
            const size_t numInside = inside_idx.size();

            if (numInside == 0) continue;
            if (numInside == 3) { insertBoundaryFace(be); continue; }

            if (numInside == 1) {
                //     (-)
                //     / `
                //    *---*
                //   /     `
                // (+)--he->(+)
                auto he = be.halfEdge(inside_idx[0]);
                size_t vi = be.vertex(inside_idx[0]).volumeVertex().index();
                outElements.emplace_back(
                        insertOrigVertex(vi),
                        insertEdgeVertex(he.tail().volumeVertex().index(), vi),
                        insertEdgeVertex(he.tip ().volumeVertex().index(), vi));
            }
            else if (numInside == 2) {
                //     a(-)
                //he /  / `
                //  /  / | * d
                // v  /  |/ `
                // b(-)--*---(+) v
                //       c
                auto he = be.halfEdge(outside_idx[0]);
                size_t va = insertOrigVertex(he.tail().volumeVertex().index());
                size_t vb = insertOrigVertex(he. tip().volumeVertex().index());

                size_t vi = be.vertex(outside_idx[0]).volumeVertex().index();
                size_t vc = insertEdgeVertex(he. tip().volumeVertex().index(), vi);
                size_t vd = insertEdgeVertex(he.tail().volumeVertex().index(), vi);

                outElements.emplace_back(va, vb, vc);
                outElements.emplace_back(va, vc, vd);
            }
            else {
                throw std::runtime_error("Impossible numInside: " + std::to_string(numInside));
            }
        }
    }

    return numContourTris;
}

#endif /* end of include guard: MARCHING_TETRAHEDRA_HH */
