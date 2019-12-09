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
#include <set>
#include <stdexcept>
#include <iostream>

#include <MeshFEM/SparseMatrices.hh>

namespace UniformLaplacian {

// Assemble the rank-deficient nv x nv uniform graph Laplacian (rank nv - 1).
// If varForVertex is passed, vertices can share variables (e.g. to implement
// periodic constraints).
template<class _Mesh>
void assemble(_Mesh &mesh, SPSDSystem<Real> &system,
              const std::vector<size_t> &varForVertex = std::vector<size_t>()) {
    size_t numVertices = mesh.numVertices();
    bool hasVarForVertex = (varForVertex.size() != 0);
    if (hasVarForVertex && (varForVertex.size() != numVertices))
        throw std::runtime_error("Invalid varForVertex size.");
    size_t numVars = numVertices;
    if (hasVarForVertex) {
        size_t m = *std::max_element(varForVertex.begin(), varForVertex.end());
        numVars = m + 1;
    }
    if (numVars > numVertices) { std::cerr << "WARNING: more variables than vertices." << std::endl; }
    
    // We currently don't have vetex-vertex connectivity. It can be accessed
    // with circulators for TriMesh, but it would require a bit more code for
    // TetMesh--instead use elements to determine the connectivity
    // (inefficient).
    std::vector<std::set<size_t>> adj(numVars);
    size_t numEdges = 0;
    for (auto e : mesh.elements()) {
        for (size_t i = 0; i < e.numVertices(); ++i) {
            size_t vi = e.vertex(i).index();
            if (hasVarForVertex) vi = varForVertex.at(vi);
            auto &adj_i = adj.at(vi);
            for (size_t j = i; j < e.numVertices(); ++j) {
                size_t vj = e.vertex(j).index();
                if (hasVarForVertex) vj = varForVertex.at(vj);
                // Insert undirected (i, j) if it hasn't already been
                if (adj_i.count(vj) == 0) {
                    adj_i.insert(vj);
                    adj.at(vj).insert(vi);
                    ++numEdges;
                }
            }
        }
    }

    TripletMatrix<> L(numVars, numVars);
    L.reserve(2 * numEdges + numVars);
    for (size_t vi = 0; vi < numVars; ++vi) {
        const auto &adj_i = adj.at(vi);
        size_t numNeighbors = adj_i.size();
        // if (numNeighbors == 0)
        //     std::cerr << "WARNING: variable " << vi << " unreferenced" << std::endl;
        L.addNZ(vi, vi, (Real) numNeighbors);
        for (size_t vj : adj_i)
            L.addNZ(vi, vj, -1.0);
    }

    system.set(L);
}

}

#endif /* end of include guard: UNIFORMLAPLACIAN_HH */
