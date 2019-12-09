////////////////////////////////////////////////////////////////////////////////
// gen_grid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Generates a 2D grid of quads or a 3D grid of hexes. The vertex positions
//      encode their row/column index.
//
//      MSH ordering is used in all cases.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/19/2015 13:34:03
////////////////////////////////////////////////////////////////////////////////
#ifndef GEN_GRID_HH
#define GEN_GRID_HH
#include <vector>
#include <stdexcept>

template<class Vertex, class Element>
void gen_grid(size_t sx, size_t sy,
              std::vector<Vertex> &vertices,
              std::vector<Element> &elements) {
    size_t nCols = sx, nRows = sy;
    vertices.clear(), elements.clear();
    //   ^ row (y)
    //   |
    //   |
    //   |
    //   +------> column (x)
    // index(Vertex(r, c)) = (nCols + 1) * r + c
    auto cornerVertexIdx = [=](size_t r, size_t c) { return (nCols + 1) * r + c; };

    // Generate corner vertices
    for (size_t r = 0; r <= nRows; ++r) {
        for (size_t c = 0; c <= nCols; ++c) {
            vertices.emplace_back(c, r, 0.0);
        }
    }

    // Generate quads in GMSH order
    for (size_t r = 0; r < nRows; ++r) {
        for (size_t c = 0; c < nCols; ++c) {
            elements.emplace_back(cornerVertexIdx(r    , c    ),
                                  cornerVertexIdx(r    , c + 1),
                                  cornerVertexIdx(r + 1, c + 1),
                                  cornerVertexIdx(r + 1, c    ));
        }
    }
}

template<class Vertex, class Element>
void gen_grid(size_t sx, size_t sy, size_t sz,
              std::vector<Vertex> &vertices,
              std::vector<Element> &elements) {
    size_t nCols = sx, nRows = sy, nSlices = sz;
    vertices.clear(), elements.clear();
    //   ^ row (y)
    //   |
    //   |
    //   |
    //   +------> column (x)
    //  /
    // v slice (z)
    // index(Vertex(s, r, c)) = (nCols + 1) * ((nRows + 1) * s + r) + c
    auto cornerVertexIdx = [=](size_t s, size_t r, size_t c)
            { return (nCols + 1) * ((nRows + 1) * s + r) + c; };

    // Generate corner vertices
    for (size_t s = 0; s <= nSlices; ++s) {
        for (size_t r = 0; r <= nRows; ++r) {
            for (size_t c = 0; c <= nCols; ++c) {
                vertices.emplace_back(c, r, s);
            }
        }
    }

    // Generate hexes in GMSH order
    for (size_t s = 0; s < nSlices; ++s) {
        for (size_t r = 0; r < nRows; ++r) {
            for (size_t c = 0; c < nCols; ++c) {
                elements.emplace_back(cornerVertexIdx(s    , r    , c    ),
                                      cornerVertexIdx(s    , r    , c + 1),
                                      cornerVertexIdx(s    , r + 1, c + 1),
                                      cornerVertexIdx(s    , r + 1, c    ),
                                      cornerVertexIdx(s + 1, r    , c    ),
                                      cornerVertexIdx(s + 1, r    , c + 1),
                                      cornerVertexIdx(s + 1, r + 1, c + 1),
                                      cornerVertexIdx(s + 1, r + 1, c    ));
            }
        }
    }
}

template<class Vertex, class Element>
void gen_grid(const std::vector<size_t> &sizes,
                    std::vector<Vertex> &vertices,
                    std::vector<Element> &elements)
{
    switch(sizes.size()) {
        case 2: gen_grid(sizes[0], sizes[1], vertices, elements); break;
        case 3: gen_grid(sizes[0], sizes[1], sizes[2], vertices, elements); break;
        default: throw std::runtime_error("Only 2D and 3D grids are supported.");
    }
}


#endif /* end of include guard: GEN_GRID_HH */
