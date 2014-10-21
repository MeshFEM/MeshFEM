////////////////////////////////////////////////////////////////////////////////
// Simplex.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Properties of K-simplices and the position of FEM nodes on them.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/14/2014 14:55:16
////////////////////////////////////////////////////////////////////////////////
#ifndef SIMPLEX_HH
#define SIMPLEX_HH

namespace Simplex {
    constexpr size_t numVertices(size_t K) { return K + 1; }
    constexpr size_t numEdges(size_t K)    { return (K * (K + 1)) / 2; }
    constexpr size_t numNodes(size_t K, size_t deg) {
        return deg == 0 ? 1 : (deg == 1 ? numVertices(K)
                                        : numVertices(K) + numEdges(K));
    }

    enum { Edge = 1, Triangle = 2, Tetrahedron = 3};

    // For interpolation of values at the edge nodes, we need to know the nodes
    // indices at the endpoints of the corresponding edges. For 1- 2- and
    // 3-simplices, these are found using (prefixes of) the following lookup tables.
    // To use these tables, edge nodes are re-indexed so that the first edge is index
    // 0 (i.e. edge index = node index - numVertices)
    constexpr size_t edgeStartNode(size_t i) { return (i < 3) ? i : (6 - i) % 3; }
    constexpr size_t edgeEndNode(size_t i)   { return (i < 3) ? (i + 1) % 3 : 3; }
    //   const size_t edgeStartNode[6] = { 0, 1, 2, 0, 2, 1 };
    //   const size_t edgeEndNode[6]   = { 1, 2, 0, 3, 3, 3 };

    //   // For gradients of edge shape functions, we need to know the "other nodes"
    //   // not incident each edge. Again, these are found using (prefixes of) the
    //   // following lookup table after re-indexing. For 1-simplices, no lookup is
    //   // needed. For 1-simplices, only the first sub-entry of the first three
    //   // entries are used. For 2-simplices, all entries are used.
    //   const size_t otherNodes[6][2] = { {2, 3}, {0, 3}, {1, 3},
    //                                     {1, 2}, {0, 1}, {0, 2} };
}

#endif /* end of include guard: SIMPLEX_HH */
