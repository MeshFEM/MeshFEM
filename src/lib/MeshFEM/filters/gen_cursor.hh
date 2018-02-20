////////////////////////////////////////////////////////////////////////////////
// gen_cursor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Appends a 3D line mesh cursor (crosshairs) centered a particular point.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  09/04/2016 19:09:20
////////////////////////////////////////////////////////////////////////////////
#ifndef GEN_CURSOR_HH
#define GEN_CURSOR_HH
#include <vector>

template<class Pt, class Vertex, class Element>
void gen_cursor(double radius, const Pt &p,
                std::vector<Vertex> &vertices,
                std::vector<Element> &elements) {
    size_t offset = vertices.size();
    vertices.push_back(p);

    vertices.emplace_back(p[0] - radius, p[1], p[2]);
    vertices.emplace_back(p[0] + radius, p[1], p[2]);

    vertices.emplace_back(p[0], p[1] - radius, p[2]);
    vertices.emplace_back(p[0], p[1] + radius, p[2]);

    vertices.emplace_back(p[0], p[1], p[2] - radius);
    vertices.emplace_back(p[0], p[1], p[2] + radius);

    for (size_t i = 1; i <= 6; ++i)
        elements.emplace_back(offset, offset + i);
}

#endif /* end of include guard: GEN_CURSOR_HH */
