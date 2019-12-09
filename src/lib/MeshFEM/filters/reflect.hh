#ifndef REFLECT_HH
#define REFLECT_HH
#include <ratio>
#include <vector>
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/ComponentMask.hh>

inline bool isEq(Real a, Real b, Real tol = 0) {
    return std::abs(a - b) < tol;
}

//     +------+------+
//    /  reflect z  /|
//   +------+------+ |
//  /      /      /| |
// +------+------+ | +
// |      |      | |/|
// | refx | base | + |
// |      | (0)  |/| |
// +------+------+ | |
// |      |      | |/ 
// |  reflect y  | |
// |      |      |/
// +------+------+
// Reflect 2/3D geometry around its bounding box's minimum faces to
// generate a mesh that is 2^dim x larger. Vertices on the minimum faces are not
// duplicated, so the resulting mesh is connected
template<typename Vertex, typename Element, typename TOL = std::ratio<1, long(1e10)>>
void reflect(const size_t Dim, // dimensions to reflect in (length of [x, y, z] prefix)
             const std::vector<Vertex> &inVertices,
             const std::vector<Element> &inElements,
             std::vector<Vertex>  &outVertices,
             std::vector<Element> &outElements,
             const ComponentMask &mask = ComponentMask("xyz"),
             std::vector<size_t> *origVertex  = nullptr, // optional tracking of origin vertex/elem
             std::vector<size_t> *origElement = nullptr,
             std::vector<ComponentMask> *vtxReflection = nullptr, // the reflection applied to generate each output vertex
             std::vector<ComponentMask> *elmReflection = nullptr  // the reflection applied to generate each output elem
             )
{
    static constexpr double tolerance = double(TOL::num) / double(TOL::den);
    BBox<decltype(inVertices[0].point)> bbox((inVertices));

    outVertices = inVertices;
    outElements = inElements;

    if (origVertex) {
        origVertex->resize(inVertices.size());
        for (size_t i = 0; i < inVertices.size(); ++i)
            (*origVertex)[i] = i;
        if (vtxReflection) vtxReflection->assign(inVertices.size(), ComponentMask());
    }
    if (origElement) {
        origElement->resize(inElements.size());
        for (size_t i = 0; i < inElements.size(); ++i)
            (*origElement)[i] = i;
        if (elmReflection) elmReflection->assign(inElements.size(), ComponentMask());
    }

    // Reflect geometry in the dth dimension
    for (size_t d = 0; d < Dim; ++d) {
        if (!mask.has(d)) continue;
        // We need a mapping from vertex indices of the new reflected geometry
        // we're about to create to global vertex indices.
        // All vertices except those on the reflection pane are copied.
        std::vector<size_t> globalVertexIndex(outVertices.size());

        // Generate reflected vertices for vertices not on the reflection plane
        size_t numVertices = outVertices.size();
        for (size_t vi = 0; vi < numVertices; ++vi) {
            bool onReflectionPlane = isEq(outVertices[vi][d], bbox.minCorner[d], tolerance);
            if (onReflectionPlane) globalVertexIndex[vi] = vi;
            else {
                globalVertexIndex[vi] = outVertices.size();
                auto reflV = outVertices[vi];
                reflV[d] *= -1; reflV[d] += 2 * bbox.minCorner[d];
                outVertices.push_back(reflV);
                if (origVertex) { origVertex->push_back(origVertex->at(vi)); }
                if (vtxReflection) {
                    ComponentMask refl = vtxReflection->at(vi);
                    refl.set(d);
                    vtxReflection->push_back(refl);
                }
            }
        }

        // Generate reflected elements
        size_t numElems = outElements.size();
        for (size_t ei = 0; ei < numElems; ++ei) {
            auto re = outElements[ei];
            // Reindex corner indices.
            // Note: reflection inverts the elements, so we must also permute
            // the corner indices to get positive orientation.
            // This actually matters! The inverted reflected elements cause a
            // cancellation during stiffness matrix assembly resulting in a
            // singular system.
            size_t tmp = re[0];
            re[0] = globalVertexIndex.at(re[1]);
            re[1] = globalVertexIndex.at(tmp);
            for (size_t d2 = 2; d2 < re.size(); ++d2) re[d2] = globalVertexIndex.at(re[d2]);
            outElements.push_back(re);
            if (origElement) { origElement->push_back(origElement->at(ei)); }
            if (elmReflection) {
                ComponentMask refl = elmReflection->at(ei);
                refl.set(d);
                elmReflection->push_back(refl);
            }
        }
    }
}


#endif /* end of include guard: REFLECT_HH */
