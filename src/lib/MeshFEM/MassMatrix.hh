////////////////////////////////////////////////////////////////////////////////
// MassMatrix.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Constructs the sparse FEM (scalar) shape function mass matrix in
//      triplet form.
//
//      Also supports construction of the lumped mass matrix, a diagonal matrix
//      whose entries are the sums of the original mass matrix rows.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/28/2016 17:23:55
////////////////////////////////////////////////////////////////////////////////
#ifndef MASSMATRIX_HH
#define MASSMATRIX_HH

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/GaussQuadrature.hh>
#include <limits>
#include <stdexcept>

namespace MassMatrix {

// Adapter to access node collections for different shape function degrees.
// Generic version: full degree
template<size_t Deg, class _FEMMesh>
struct NodeGetter {
    static_assert(Deg == _FEMMesh::Deg, "Only full-degree and degree 1 mass matrices are supported.");
    using EHandle = typename _FEMMesh::template EHandle<const _FEMMesh>;
    using NRT     = typename EHandle::NRangeTraits;
    static SubEntityHandleRange<NRT> nodes   (const  EHandle    &h) { return h.nodes(); }
    static size_t                    numNodes(const _FEMMesh &mesh) { return mesh.numNodes(); }
    static constexpr size_t          numElementNodes()              { return EHandle::numNodes(); }
};

// Degree-1 specialization: nodes always coincide with the vertices
template<class _FEMMesh>
struct NodeGetter<1, _FEMMesh> {
    using EHandle = typename _FEMMesh::template EHandle<const _FEMMesh>;
    using VRT     = typename EHandle::VRangeTraits;
    static SubEntityHandleRange<VRT> nodes   (const  EHandle    &h) { return h.vertices(); }
    static size_t                    numNodes(const _FEMMesh &mesh) { return mesh.numVertices(); }
    static constexpr size_t          numElementNodes()              { return EHandle::numVertices(); }
};

template<size_t Deg>
struct Impl {
    template<class _FEMMesh>
    static void construct(const _FEMMesh &mesh, 
                          const std::vector<bool> &skipElem,
                          TripletMatrix<> &M) {
        bool skipping = (skipElem.size() == mesh.numElements());
        if (!skipping && skipElem.size()) throw std::runtime_error("Invalid skipElem array size.");

        using NG = NodeGetter<Deg, _FEMMesh>;
        constexpr size_t K = _FEMMesh::K;

        size_t nn = NG::numNodes(mesh);
        size_t nen = NG::numElementNodes();
        M.init(nn, nn);
        M.reserve(mesh.numElements() * (nen * (nen + 1)) / 2);
        for (auto e : mesh.elements()) {
            if (skipping && skipElem[e.index()]) continue;
            for (auto ni : NG::nodes(e)) {
                for (auto nj : NG::nodes(e)) {
                    if (nj.index() < ni.index()) continue; // upper tri only
                    Real val = Quadrature<K, 2 * Deg>::integrate(
                            [&](const EvalPt<K> &pt) {
                                // Note: MSVC breaks if we use `K` instead of _FEMMesh::K :(
                                return shapeFunction<Deg, _FEMMesh::K>(ni.localIndex(), pt) *
                                       shapeFunction<Deg, _FEMMesh::K>(nj.localIndex(), pt);
                            }, e->volume());
                    M.addNZ(ni.index(), nj.index(), val);
                }
            }
        }
    }
};

// Degree deduction wrapper
template<>
struct Impl<std::numeric_limits<size_t>::max()> {
    template<class _FEMMesh>
    static void construct(const _FEMMesh &mesh,
                          const std::vector<bool> &skipElem,
                          TripletMatrix<> &M) {
        Impl<_FEMMesh::Deg>::construct(mesh, skipElem, M);
    }
};

// Degree is deduced from _FEMMesh, unless specified.
// Construct upper triangle of FEM mass matrix.
//
// If "lumped == true", a diagonal mass matrix is constructed by summing the
// entries in each row.
//
// If skipElem array is passed, contributions from elements e with
// "skipElem[e] == true" are ignored. In other words, functions are assumed to
// vanish on these elements.
template<size_t Deg = std::numeric_limits<size_t>::max(), class _FEMMesh>
TripletMatrix<> construct(const _FEMMesh &mesh, bool lumped = false,
                          const std::vector<bool> &skipElem = std::vector<bool>()) {
    TripletMatrix<> M;
    Impl<Deg>::construct(mesh, skipElem, M);

    if (lumped) {
        // Sum rows and place on the diagonal
        // Note, only the upper triangle is stored, so off-diagonal entries
        // contribute to two rows.

        std::vector<Real> diag(M.n, 0.0);
        for (const Triplet<Real> &t : M.nz) {
            diag.at(t.i) += t.v;
            if (t.j != t.i) diag.at(t.j) += t.v;
        }
        TripletMatrix<> M_lumped(M.n, M.n);
        M_lumped.reserve(diag.size());
        for (size_t i = 0; i < diag.size(); ++i)
            M_lumped.addNZ(i, i, diag[i]);

        // Diagonal matrix doesn't need special handling.
        return M_lumped;
    }

    M.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    return M;
}

// Construct the mass matrix for vector-valued shape functions
// (assumes interleaved ordering of the unknown components (x0, y0, ...))
template<size_t Deg = std::numeric_limits<size_t>::max(), class _FEMMesh, typename... Args>
TripletMatrix<> construct_vector_valued(const _FEMMesh &mesh, Args&&... args) {
    TripletMatrix<> Mscalar = construct<Deg>(mesh, std::forward<Args>(args)...);
    Mscalar.sumRepeated();

    constexpr size_t N = _FEMMesh::EmbeddingDimension;
    TripletMatrix<> M(Mscalar.n * N, Mscalar.m * N);
    M.reserve(N * Mscalar.nnz());

    for (const auto &t : Mscalar)
        for (size_t c = 0; c < N; ++c)
            M.addNZ(N * t.i + c, N * t.j + c, t.v);
    M.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    return M;
}

}

#endif /* end of include guard: MASSMATRIX_HH */
