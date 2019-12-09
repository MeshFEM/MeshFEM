////////////////////////////////////////////////////////////////////////////////
// reorient_negative_elements.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Reorient triangles or tets so that volume is positive.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/30/2017 03:42:42
////////////////////////////////////////////////////////////////////////////////
#ifndef REORIENT_NEGATIVE_ELEMENTS_HH
#define REORIENT_NEGATIVE_ELEMENTS_HH

#include <vector>
#include <limits>
#include <algorithm>
#include <MeshFEM/Types.hh>

template<class Vertex, class Element>
size_t reorient_negative_elements(std::vector<Vertex>  &vertices,
                                  std::vector<Element> &elements) {
    size_t numFlipped = 0;

    for (Element &e : elements) {
        // Proper (positive) orientations:
        // |       3         |              |
        // |       *         |       2      |
        // |      / \`       |      / \     |
        // |     /   \ `* 2  |     /   \    |
        // |    / __--\ /    |    /     \   |
        // |  0*-------* 1   |   0-------1  |
        Real signedVolScaled = 0;
        if (e.size() == 3) {
            auto p0 = truncateFrom3D<Point2D>(vertices[e[0]].point),
                 p1 = truncateFrom3D<Point2D>(vertices[e[1]].point),
                 p2 = truncateFrom3D<Point2D>(vertices[e[2]].point);
            Vector2D e0 = p2 - p1,
                     e1 = p2 - p0;
            signedVolScaled = e0[0] * e1[1] - e0[1] * e1[0];
        }
        else if (e.size() == 4) {
            auto p0 = vertices[e[0]].point,
                 p1 = vertices[e[1]].point,
                 p2 = vertices[e[2]].point,
                 p3 = vertices[e[3]].point;
            signedVolScaled = ((p3 - p1).cross(p2 - p1)).dot(p0 - p1);
        }
        else throw std::runtime_error("Invalid element size: " + std::to_string(e.size()));

        if (signedVolScaled < 0) {
            ++numFlipped;
            std::swap(e[0], e[1]);
        }
    }

    return numFlipped;
}

#endif /* end of include guard: REORIENT_NEGATIVE_ELEMENTS_HH */
