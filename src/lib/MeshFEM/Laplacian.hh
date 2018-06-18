////////////////////////////////////////////////////////////////////////////////
// Laplacian.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Construct the sparse FEM Laplacian matrix in triplet form.
//      CONSTRUCTS UPPER TRIANGLE ONLY!!!
//
//      Note: this actually constructs the negative of the negative semidefinite
//      Laplacian operator. In other words, it's the positive semidefinite
//      matrix for the PDE:
//      - laplacian u = f
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/18/2016 12:40:54
////////////////////////////////////////////////////////////////////////////////
#ifndef LAPLACIAN_HH
#define LAPLACIAN_HH

#include <MeshFEM/SparseMatrices.hh>
#include <limits>
#include <MeshFEM/GaussQuadrature.hh>

namespace Laplacian {

// General degree version
template<size_t Deg>
struct Impl {
    template<class _FEMMesh>
    static void construct(const _FEMMesh &mesh, TripletMatrix<> &L) {
        static_assert(Deg == _FEMMesh::Deg, "Degree mismatch");
        constexpr size_t N = _FEMMesh::K;

        size_t nn = mesh.numNodes();
        size_t numElemNodes = mesh.element(0).numNodes();
        L.init(nn, nn);
        L.reserve(mesh.numElements() * (numElemNodes * (numElemNodes + 1)) / 2);
        for (auto e : mesh.elements()) {
            for (size_t i = 0; i < numElemNodes; ++i) {
                size_t ni = e.node(i).index();
                auto grad_phi_i = e->gradPhi(i);
                for (size_t j = i; j < numElemNodes; ++j) {
                    auto grad_phi_j = e->gradPhi(j);
                    size_t nj = e.node(j).index();
                    Real val = Quadrature<N, 2 * (Deg - 1)>::integrate(
                        [&](const EvalPt<N> &pt) { return
                            grad_phi_i(pt).dot(grad_phi_j(pt));
                        }, e->volume());
                    if (ni <= nj) L.addNZ(ni, nj, val);
                    else          L.addNZ(nj, ni, val);
                }
            }
        }
    }
};

// Degree-1 version (forced or otherwise)
template<>
struct Impl<1> {
    template<class _FEMMesh>
    static void construct(const _FEMMesh &mesh, TripletMatrix<> &L) {
        size_t nv = mesh.numVertices();
        size_t numCorners = mesh.element(0).numVertices();
        L.init(nv, nv);
        L.reserve(mesh.numElements() * (numCorners * (numCorners + 1)) / 2);
        for (auto e : mesh.elements()) {
            const auto &lambda = e->gradBarycentric();
            for (size_t i = 0; i < numCorners; ++i) {
                size_t vi = e.vertex(i).index();
                for (size_t j = i; j < numCorners; ++j) {
                    size_t vj = e.vertex(j).index();
                    Real val = lambda.col(i).dot(lambda.col(j)) * e->volume();
                    if (vi <= vj) L.addNZ(vi, vj, val);
                    else          L.addNZ(vj, vi, val);
                }
            }
        }
    }
};

// Degree deduction version
template<>
struct Impl<std::numeric_limits<size_t>::max()> {
    template<class _FEMMesh>
    static void construct(const _FEMMesh &mesh, TripletMatrix<> &L) {
        Impl<_FEMMesh::Deg>::construct(mesh, L);
    }
};

// Degree is deduced from _FEMMesh, unless specified.
// Construct upper triangle of FEM Laplacian matrix.
template<size_t Deg = std::numeric_limits<size_t>::max(), class _FEMMesh>
TripletMatrix<> construct(const _FEMMesh &mesh) {
    TripletMatrix<> L;
    Impl<Deg>::construct(mesh, L);
    L.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    return L;
}

}

#endif /* end of include guard: LAPLACIAN_HH */
