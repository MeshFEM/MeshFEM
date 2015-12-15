////////////////////////////////////////////////////////////////////////////////
// UniformLaplacian.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Constructs the (primal mesh) uniform laplacian system. This is the graph
//      laplacian of the mesh graph (i.e. vertices instead of FEM nodes).
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/12/2015 12:14:03
////////////////////////////////////////////////////////////////////////////////
#ifndef UNIFORMLAPLACIAN_HH
#define UNIFORMLAPLACIAN_HH

#include <algorithm> 
#include <vector>

#include "SparseMatrices.hh"

namespace UniformLaplacian {

template<class _Mesh>
// Assemble the rank-deficient nv x nv uniform graph Laplacian (rank nv - 1).
SPSDSystem<Real> assemble(_Mesh &mesh, SPSDSystem<Real> &system) {
    // We currently don't have vetex-vertex connectivity. It can be accessed
    // with circulators for TriMesh, but it would require a bit more code for
    // TetMesh--instead use elements to determine the connectivity
    // (inefficient).
    size_t nv = mesh.numVertices();
    std::vector<std::vector<size_t>> adj(nv);
    size_t numEdges = 0;
    for (auto e : mesh.elements()) {
        for (size_t i = 0; i < e.numVertices(); ++i) {
            size_t vi = e.vertex(i).index();
            auto &adj_i = adj.at(vi);
            for (size_t j = i; j < e.numVertices(); ++j) {
                size_t vj = e.vertex(j).index();
                // Check if undirected (i, j) has been inserted.
                auto ij = std::find(adj_i.begin(), adj_i.end(), vj);
                if (ij == adj_i.end()) {
                    adj_i.push_back(vj);
                    ++numEdges;
                    adj.at(vj).push_back(vi);
                }
            }
        }
    }

    TripletMatrix<> L(nv, nv);
    L.reserve(numEdges + nv);
    for (size_t vi = 0; vi < nv; ++vi) {
        const auto &adj_i = adj.at(vi);
        L.addNZ(vi, vi, (Real) adj_i.size());
        for (vj : adj_i)
            L.addNZ(vi, vj, -1.0);
    }

    return SPSDSystem<Real>(L);
}

#endif /* end of include guard: UNIFORMLAPLACIAN_HH */
