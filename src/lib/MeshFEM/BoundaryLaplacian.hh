////////////////////////////////////////////////////////////////////////////////
// BoundaryLaplacian.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Discrete Nb x Nb FEM Laplacian operator, L, acting on boundary values:
//          L_ij = int_bdry grad bphi_i . grad bphi_j dA.
//      Here, bphi_i and bphi_j are the ith and jth boundary shape functions
//      respectively. Nb is the number of boundary nodes.
//////////////////////////////////////////////////////////////////////////////////
// PERIODICITY:
//      We also provide rudimentary support for operating on the boundary of
//      periodic volume meshes. Conceptually, this case involves computing the
//      Laplacian of a periodic boundary scalar field over a periodically-tiled
//      boundary mesh.
//
//      Boundary elements on the period cell faces are not true boundary
//      elements (they "cancel out" when the mesh is tiled), and we must ignore
//      their contributions. Likewise, the variables of "false boundary
//      nodes" (nodes with only periodic boundary elements incident)
//      should not appear in the system.
//      
//      Further, we assume the input scalar field's values on identified
//      "periodic nodes" are equal, and we must produce a result with this
//      periodicity.
//
//      We could enforce this constraint by constructing a reduced set of
//      variables for only the true boundary nodes and have identified
//      nodes share variables. However, we take the following simpler
//      approach that works if we're not using the Laplacian with a direct
//      solver:
//          1) Still build a full Vb x Vb matrix. This matrix has in its
//             nullspace the components for "false boundary nodes"
//          2) The matrix assumes values on identified nodes equal, and (will
//             read the values from all such nodes at some point).
//          3) The matrix outputs the periodic Laplacian on EVERY identified
//             node.
//      Fact (1) means, unsurprisingly, that the resulting Laplacian is highly
//      rank deficient. Facts (2) and (3) mean that it is actually ASYMMETRIC.
//
//      To understand this, consider the more traditional reduced var approach.
//      It would involve constructing a symmetric "reduced" Laplacian, Lr. Using
//      Lr in our setting requires introducing a selection matrix, S, that reads
//      off the reduced variables from an arbitrary one of the corresponding
//      per-boundary-node variables. Then we can build the symmetric Nb by Nb
//      matrix S^T Lr S. This computes the Laplacian correctly, but then places
//      it on an arbitrary one of the identified boundary nodes. Since we
//      want to put the Laplacian on EVERY identified node, we need an
//      additional "distributing" matrix, D, leading to the asymmetry:
//          Lfull = D S^T Lr S.
//      It is possible to construct a symmetric Nb x Nb matrix with the property
//      that it takes a vector of boundary value and outputs the full Laplacian
//      value on each boundary value, but this would require replacing S with a
//      carefully weighted "identified node averaging" matrix and isn't worth
//      the trouble.
//      
//      Finally, we describe how to build Lfull without constructing the reduced
//      variable system. Call Lsum the Vb x Vb matrix obtained by summing the
//      per-element Laplacian matrices for every non-periodic boundary element.
//      In other words, this is the full Laplacian before applying boundary
//      conditions. When applied to a vector satisfying the periodicity input
//      constraints, this computes the correct Laplacian for all non-periodic
//      nodes, but puts only a part of the full result on each identified
//      periodic nodes. For each set of identified nodes, we must sum
//      these partial results and distribute the total to every node in the
//      set. In other words Lfull = DS * Lsum, where sum-distribute operator DS
//      has the vector v_ij = [1 if node j identified with i, 0 otherwise] in
//      each row i. We implement this by accumulating terms from the per-element
//      matrix to not only the rows corresponding to the element corners,
//      but also to the rows for all nodes identified with the element corners.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/29/2016 13:33:06
////////////////////////////////////////////////////////////////////////////////
#ifndef BOUNDARYLAPLACIAN_HH
#define BOUNDARYLAPLACIAN_HH

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/GaussQuadrature.hh>

#include <vector>

namespace BoundaryLaplacian {

// Assemble sparse Vb x Vb boundary Laplacian matrix. For non-periodic cases,
// only "mesh" should be passed, and a symmetric matrix (stored in upper
// triangle) is returned.
//
// If isPeriodicBE, and identifiedBdryNodes is passed, we build an asymmetric
// matrix for the implied periodic problem (see discussion above).
//
// Semantics of identifiedBdryNodes: a node *IS* identified with itself, so
//      identifiedBdryNodes[i] contains i
template<class _Mesh>
TripletMatrix<> assemble(const _Mesh &mesh,
        const std::vector<bool> &isPeriodicBE = std::vector<bool>(),
        const std::vector<std::vector<size_t>> &identifiedBdryNodes = std::vector<std::vector<size_t>>())
{
    constexpr size_t K   = _Mesh::K - 1;
    constexpr size_t Deg = _Mesh::Deg;

    bool periodic = isPeriodicBE.size() == mesh.numBoundaryElements();
    assert((isPeriodicBE.size() == 0)        != periodic);
    assert((identifiedBdryNodes.size() == 0) != periodic);
    assert(!periodic || identifiedBdryNodes.size() == mesh.numBoundaryNodes());

    TripletMatrix<> L(mesh.numBoundaryNodes(),
                      mesh.numBoundaryNodes());

    for (auto be : mesh.boundaryElements()) {
        if (periodic && isPeriodicBE.at(be.index())) continue;
        for (size_t i = 0; i < be.numNodes(); ++i) {
            for (size_t j = i; j < be.numNodes(); ++j) {
                size_t bni = be.node(i).index();
                size_t bnj = be.node(j).index();
                Real val = Quadrature<K, 2 * (Deg - 1)>::integrate(
                    [&](const VectorND<Simplex::numVertices(K)> &p) {
                        return be->gradPhi(i)(p).dot(be->gradPhi(j)(p));
                    }, be->volume());
                if (periodic) {
                    // Build the full, non-symmetric matrix: accumulate a
                    // contribution to not only (bni, bnj) and (bnj, bni), but
                    // also the other associated rows (pair_bni, bnj) and
                    // (pair_bnj, bni) for all nodes periodically identified
                    // with bni and bnj.
                    for (size_t pair_bni : identifiedBdryNodes.at(bni)) L.addNZ(pair_bni, bnj, val);

                    // Process lower triangle of the per-elem matrix too
                    if (bni != bnj)
                        for (size_t pair_bnj : identifiedBdryNodes.at(bnj)) L.addNZ(pair_bnj, bni, val);
                }
                else {
                    // Non-periodic case: build the upper triangle of the
                    // symmetric matrix.
                    if (bni <= bnj) L.addNZ(bni, bnj, val);
                    else            L.addNZ(bnj, bni, val);
                }
            }
        }
    }

    L.symmetry_mode = periodic ? TripletMatrix<>::SymmetryMode::NONE
                               : TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    return L;
}

} // End of namespace BoundaryLaplacian

#endif /* end of include guard: BOUNDARYLAPLACIAN_HH */
