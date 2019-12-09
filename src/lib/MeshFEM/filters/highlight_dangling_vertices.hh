////////////////////////////////////////////////////////////////////////////////
// highlight_dangling_vertices.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Highlights dangling (unreferenced) vertices in a mesh by writing a line
//      mesh with a + shape centered around each.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  11/20/2015 12:52:40
////////////////////////////////////////////////////////////////////////////////
#ifndef HIGHLIGHT_DANGLING_VERTICES_HH
#define HIGHLIGHT_DANGLING_VERTICES_HH

#include <MeshFEM/Geometry.hh>
#include <MeshFEM/MeshIO.hh>

template<class Vertex, class Element>
void highlight_dangling_vertices(const std::vector<Vertex>  &vertices,
                                 const std::vector<Element> &elements,
                                 const std::string &path, bool pointsOnly = false) {
    std::vector<MeshIO::IOVertex > highlightVertices;
    std::vector<MeshIO::IOElement> highlightEdges;

    BBox<Vector3D> bbox(vertices);
    auto dim = bbox.dimensions();
    
    std::vector<bool> seen(vertices.size(), false);
    for (const auto &e : elements) {
        for (size_t c = 0; c < e.size(); ++c)
            seen.at(e[c]) = true;
    }

    for (size_t i = 0; i < vertices.size(); ++i) {
        if (!seen[i]) {
            auto p = vertices[i].point;
            size_t offset = highlightVertices.size(), numPlusvertices = 0;
            highlightVertices.emplace_back(vertices[i]);
            if (pointsOnly) continue;
            // Create plus geometry
            for (size_t d = 0; d < 3; ++d) {
                if (dim[d] > 0) {
                    Vector3D delta(Vector3D::Zero());
                    delta[d] = 0.025 * dim[d];
                    highlightVertices.emplace_back((p + delta).eval());
                    highlightVertices.emplace_back((p - delta).eval());
                    numPlusvertices += 2;
                }
            }
            for (size_t v = 1; v <= numPlusvertices; ++v)
                highlightEdges.emplace_back(offset, offset + v);
        }
    }

    if (highlightVertices.size() > 0)
        MeshIO::save(path, highlightVertices, highlightEdges);
    else 
        std::cerr << "WARNING: No dangling vertices detected; not creating highlight file." << std::endl;
}


#endif /* end of include guard: HIGHLIGHT_DANGLING_VERTICES_HH */
