////////////////////////////////////////////////////////////////////////////////
// quad_subdiv_high_aspect.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Subdivide planar rectangular quads with high aspect ratio to make them
//		more square:
//		3---------2     3----m1---2
//		|         | ==> |    |    |
//		0---------1     0----m0---1
//		Any quad more than twice as long in one dimension is split in two.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/27/2014 18:45:58
////////////////////////////////////////////////////////////////////////////////
#ifndef QUAD_SUBDIV_HIGH_ASPECT_HH
#define QUAD_SUBDIV_HIGH_ASPECT_HH

// quadIdx: index of the quad from which each output element originated
//          This can be propagated across several subdivisions by passing the
//          same array for each call.
// return: true if a subdivision was performed (in case we want to iterate until
// all 
template<class Vertex, class Element>
bool quad_subdiv_high_aspect(
        const std::vector<Vertex>  &inVertices, const std::vector<Element> &inElements,
        std::vector<Vertex>  &outVertices, std::vector<Element> &outElements,
        std::vector<size_t> &quadIdx, bool ignoreNonQuads = false, Real aspectThreshold = 2)
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
        throw std::runtime_error("Subdivision aspect ratio threshold must be > sqrt(2) for convergence");

    Element newQuad(4);

    bool subdivided = false;
    bool hasNonQuads = false;
    // Use collision grid to merge new vertices with those from adjacent cells
    Real epsilon = 1e-8;
    CollisionGrid<Real, Point3D> cgrid(epsilon);
    for (size_t i = 0; i < inElements.size(); ++i) {
        auto e = inElements[i];
        if (e.size() != 4) {
            if (ignoreNonQuads) {
                hasNonQuads = true;
                quadIdx.push_back(oldQuadIdx[i]);
                outElements.push_back(e);
                continue;
            }
            throw std::runtime_error("Non-quad encountered.");
        }
        // TODO: check for non-planar and non-rectangular cases!!!

        // Determine which edge pair, if any, should be split
        // subdivPair is 0 if ==, 1 if ||
        Point3D e0 = Point3D(inVertices[e[1]]) - Point3D(inVertices[e[0]]);
        Point3D e1 = Point3D(inVertices[e[2]]) - Point3D(inVertices[e[1]]);
        int subdivPair = -1;
        if (e0.norm() > (aspectThreshold * e1.norm())) subdivPair = 0;
        if (e1.norm() > (aspectThreshold * e0.norm())) subdivPair = 1;
        if (subdivPair < 0) {
            quadIdx.push_back(oldQuadIdx[i]);
            outElements.push_back(e);
            continue;
        }

        subdivided = true;

        // subdivPair is an index offset that effectively rotates our picture
        // by 90 degress to always look like:
        //		3---------2     3----m1---2
        //		|         | ==> | q0 | q1 |
        //		0---------1     0----m0---1

        // Midpoint vertices
        Point3D m[2] = { (Point3D(inVertices[e[0 + subdivPair]]) + Point3D(inVertices[e[ 1 + subdivPair     ]])) / 2,
                         (Point3D(inVertices[e[2 + subdivPair]]) + Point3D(inVertices[e[(3 + subdivPair) % 4]])) / 2, };

        // Generate/merge new midpoint vertices.
        int midx[2];
        for (size_t c = 0; c < 2; ++c) {
            midx[c] = cgrid.getClosestPoint(m[c], epsilon).first;
            if (midx[c] < 0) {
                midx[c] = outVertices.size();
                outVertices.push_back(m[c]);
                cgrid.addPoint(m[c], midx[c]);
            }
        }

        // Generate both new quads in ccw order
        for (size_t q = 0; q < 2; ++q) {
            newQuad[0] = e[2 * q + subdivPair];
            newQuad[1] = midx[q];
            newQuad[2] = midx[(q + 1) % 2];
            newQuad[3] = e[(2 * q + 3 + subdivPair) % 4];
            outElements.push_back(newQuad);
            quadIdx.push_back(oldQuadIdx[i]);
        }
    }

    if (hasNonQuads) {
        std::cerr << "WARNING: subdivided quads in mesh with non-quads--"
                  << "a nonmanifold mesh may have resulted." << std::endl;
    }

    return subdivided;
}

#endif /* end of include guard: QUAD_SUBDIV_HIGH_ASPECT_HH */
