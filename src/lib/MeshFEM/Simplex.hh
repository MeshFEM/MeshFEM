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
#include <stdexcept>

namespace Simplex {
    constexpr size_t numVertices(size_t K) { return K + 1; }
    constexpr size_t numEdges(size_t K)    { return (K * (K + 1)) / 2; }
    // 1D polynomials require deg + 1 nodes
    // 2D polynomials require TriNumber(deg + 1) nodes (the "deg + 1"^st triangle number)
    // 3D polynomials require TetNumber(deg + 1) nodes (the "deg + 1"^st tetrahedral number)
    // ("throw" will cause a compilation error if deg > 2 since constexpr cannot throw)
    constexpr size_t numNodes(size_t K, size_t deg) {
        return K == 1 ? deg + 1 :
              (K == 2 ? ((deg + 1) * (deg + 2)) / 2 :
              (K == 3 ? ((deg + 1) * (deg + 2) * (deg + 3)) / 6 :
              throw std::logic_error("Simplex dimension must be 1, 2, or 3")));
    }

    enum { Edge = 1, Triangle = 2, Tetrahedron = 3};

    // Node ordering is consistent with GMSH:
    //       3
    //       *          |       0       |
    //      / \`8       |      / \      |
    //     7   9 `* 2   |     3   5     |
    //    / _6--\ /5    |    /     \    |
    //  0*---4---* 1    |   1---4---2   |
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

// Function evaluation point type (barycentric coordinate array) for a K-simplex
template<size_t _K>
using EvalPt = std::array<Real, Simplex::numVertices(_K)>; // typename NTuple<Real, Simplex::numVertices(_K)>::type;
// Evaluation point as an Eigen vector type
template<size_t _K>
using EigenEvalPt = VectorND<Simplex::numVertices(_K)>;

#endif /* end of include guard: SIMPLEX_HH */
