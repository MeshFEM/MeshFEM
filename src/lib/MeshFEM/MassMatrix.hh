////////////////////////////////////////////////////////////////////////////////
// MassMatrix.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Constructs the sparse FEM shape function mass matrix.
//
// We also support construction of HRZ-lumped mass matrices. In the case of
// linear elements, this is equivalent to the popular ad-hoc method of
// collecting the sum of off-diagonal entries onto the diagonal, or using a
// quadrature rule whose points coincide with the FEM nodes. However, the
// latter two approaches generate singular or indefinite mass matrices
// in the quadratic case--as well as for more exotic elements.
// Unfortunately, an optimal lumping strategy producing a positive definite
// result for quadratic triangles/tetrahedra appears not to exist in the
// literature, and multiple references like Huges and Felippa recommend the HRZ
// strategy despite its lack of a rigorous mathematical justification.
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
    static void construct(const _FEMMesh &mesh, bool lumped,
                          const std::vector<bool> &skipElem,
                          TripletMatrix<> &M) {
        bool skipping = (skipElem.size() == mesh.numElements());
        if (!skipping && skipElem.size()) throw std::runtime_error("Invalid skipElem array size.");

        using NG = NodeGetter<Deg, _FEMMesh>;
        constexpr size_t K = _FEMMesh::K;

        size_t nn = NG::numNodes(mesh);
        constexpr size_t nen = NG::numElementNodes();
        M.init(nn, nn);
        M.reserve(mesh.numElements() * (nen * (nen + 1)) / 2);
        Eigen::Matrix<Real, nen, nen> M_e;
        if (lumped) M_e = lumpedElementMatrix<_FEMMesh>().asDiagonal();
        else        M_e = elementMatrix<_FEMMesh>();
        for (auto e : mesh.elements()) {
            if (skipping && skipElem[e.index()]) continue;
            for (auto ni : NG::nodes(e)) {
                for (auto nj : NG::nodes(e)) {
                    if (nj.index() < ni.index()) continue; // upper tri only
                    M.addNZ(ni.index(), nj.index(), e->volume() * M_e(ni.localIndex(), nj.localIndex()));
                }
            }
        }
    }

    // Per-element mass matrix, assuming an element mass of 1.
    template<class _FEMMesh>
    static auto elementMatrix() {
        using NG = NodeGetter<Deg, _FEMMesh>;
        constexpr size_t K = _FEMMesh::K;
        constexpr size_t nen = NG::numElementNodes();
        Eigen::Matrix<Real, nen, nen> result;
        for (size_t lni = 0; lni < nen; ++lni) {
            for (size_t lnj = 0; lnj < nen; ++lnj) {
                result(lni, lnj) = Quadrature<K, 2 * Deg>::integrate(
                        [&](const EvalPt<K> &pt) {
                            // Note: MSVC breaks if we use `K` instead of _FEMMesh::K :(
                            return shapeFunction<Deg, _FEMMesh::K>(lni, pt) *
                                   shapeFunction<Deg, _FEMMesh::K>(lnj, pt);
                        });
            }
        }
        return result;
    }

    // HRZ-lumped version of `elementMatrix`
    template<class _FEMMesh>
    static auto lumpedElementMatrix() {
        auto result = elementMatrix<_FEMMesh>().diagonal().eval();
        result /= result.sum();
        return result;
    }
};

// Degree deduction wrapper
template<>
struct Impl<std::numeric_limits<size_t>::max()> {
    template<class _FEMMesh>
    static void construct(const _FEMMesh &mesh, bool lumped,
                          const std::vector<bool> &skipElem,
                          TripletMatrix<> &M) {
        Impl<_FEMMesh::Deg>::construct(mesh, lumped, skipElem, M);
    }

    template<class _FEMMesh> static auto       elementMatrix() { return Impl<_FEMMesh::Deg>::template       elementMatrix<_FEMMesh>(); }
    template<class _FEMMesh> static auto lumpedElementMatrix() { return Impl<_FEMMesh::Deg>::template lumpedElementMatrix<_FEMMesh>(); }
};

// Degree is deduced from _FEMMesh, unless specified.
// Construct upper triangle of FEM mass matrix.
//
// If "lumped == true", a diagonal mass matrix is constructed using HRZ lumping.
//
// If skipElem array is passed, contributions from elements e with
// "skipElem[e] == true" are ignored. In other words, functions are assumed to
// vanish on these elements.
template<size_t Deg = std::numeric_limits<size_t>::max(), class _FEMMesh>
TripletMatrix<> construct(const _FEMMesh &mesh, bool lumped = false,
                          const std::vector<bool> &skipElem = std::vector<bool>()) {
    TripletMatrix<> M;
    Impl<Deg>::construct(mesh, lumped, skipElem, M);
    M.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    return M;
}

// Construct the mass matrix for vector-valued shape functions
// (assumes interleaved ordering of the coefficient components (x0, y0, ...))
template<size_t Deg = std::numeric_limits<size_t>::max(), class _FEMMesh, class _SPMat>
void accumulate_vector_valued(const _FEMMesh &mesh, _SPMat &M, bool lumped = false, const std::vector<bool> &skipElem = std::vector<bool>()) {
    constexpr size_t N = _FEMMesh::EmbeddingDimension;
    if ((size_t(M.m) != mesh.numNodes() * N) || (M.n != M.m))
        throw std::runtime_error("Unexpected output size");
    if (M.symmetry_mode != _SPMat::SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Unexpected symmetry mode (should be UPPER_TRIANGLE)");

    bool skipping = (skipElem.size() == mesh.numElements());
    if (!skipping && skipElem.size()) throw std::runtime_error("Invalid skipElem array size.");

    const size_t ne = mesh.numElements();
    if (lumped) {
        const auto lumpedM_e = Impl<Deg>::template lumpedElementMatrix<_FEMMesh>();
        for (size_t ei = 0; ei < ne; ++ei) {
            if (skipping && skipElem[ei]) continue;
            auto blockVars = mesh.elementNodeIndices(ei);
            assert(int(blockVars.size()) >= lumpedM_e.rows());
            auto vol = mesh.element(ei)->volume();
            for (int j = 0; j < lumpedM_e.rows(); ++j) {
                for (size_t c = 0; c < N; ++c)
                    M.addDiagEntry(N * blockVars[j] + c, vol * lumpedM_e[j]);
            }
        }
    }
    else {
        const auto M_e = Impl<Deg>::template elementMatrix<_FEMMesh>();
        for (size_t ei = 0; ei < ne; ++ei) {
            if (skipping && skipElem[ei]) continue;
            auto blockVars = mesh.elementNodeIndices(ei);
            assert(int(blockVars.size()) >= M_e.rows());
            auto vol = mesh.element(ei)->volume();
            for (int j = 0; j < M_e.cols(); ++j) {
                for (size_t c = 0; c < N; ++c) {
                    for (int i = 0; i < M_e.rows(); ++i) {
                        if (blockVars[i] <= blockVars[j])
                            M.addNZ(N * blockVars[i] + c, N * blockVars[j] + c, vol * M_e(i, j));
                    }
                }
            }
        }
    }
}

// Construct the mass matrix for vector-valued shape functions
// (assumes interleaved ordering of the unknown components (x0, y0, ...))
template<size_t Deg = std::numeric_limits<size_t>::max(), class _FEMMesh, typename... Args>
TripletMatrix<> construct_vector_valued(const _FEMMesh &mesh, Args&&... args) {
    constexpr size_t N = _FEMMesh::EmbeddingDimension;
    TripletMatrix<> M(mesh.numNodes() * N, mesh.numNodes() * N);
    M.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    accumulate_vector_valued(mesh, M, std::forward<Args>(args)...);
    return M;
}

}

#endif /* end of include guard: MASSMATRIX_HH */
