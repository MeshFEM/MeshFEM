////////////////////////////////////////////////////////////////////////////////
// SimplicialMeshInterface.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Provides a dimension-independent interface for Tri/Tet mesh. Meant to be
//  inherited from Tri/Tet mesh (CRTP).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  09/11/2016 03:02:21
////////////////////////////////////////////////////////////////////////////////
#ifndef SIMPLICIALMESHINTERFACE_HH
#define SIMPLICIALMESHINTERFACE_HH
#include <MeshFEM/Handles/Handle.hh>

template<class _Mesh>
struct SimplicialMeshTraits;

template<class... T>
struct SimplicialMeshTraits<TetMesh<T...>> {
    using _Mesh = TetMesh<T...>;
    static size_t         numSimplices(const _Mesh &m) { return m.numTets(); }
    static size_t numBoundarySimplices(const _Mesh &m) { return m.numBoundaryFaces(); }
    template<class _M> using        SHandle = typename HandleTraits<_Mesh>::template  THandle<_M>;
    template<class _M> using       BSHandle = typename HandleTraits<_Mesh>::template BFHandle<_M>;
};

template<class... T>
struct SimplicialMeshTraits<TriMesh<T...>> {
    using _Mesh = TriMesh<T...>;
    static size_t         numSimplices(const _Mesh &m) { return m.numTris(); }
    static size_t numBoundarySimplices(const _Mesh &m) { return m.numBoundaryEdges(); }
    template<class _M> using  SHandle = typename HandleTraits<_Mesh>::template  THandle<_M>;
    template<class _M> using BSHandle = typename HandleTraits<_Mesh>::template BEHandle<_M>;
};

////////////////////////////////////////////////////////////////////////////////
// Dimension-independent interface: ([K]simplices are tets,
// boundary [K-1]simplices are boundary faces.)
////////////////////////////////////////////////////////////////////////////////
template<class _Mesh>
struct SimplicialMeshInterface {
    using SMT = SimplicialMeshTraits<_Mesh>;

    size_t         numSimplices() const { return SMT::        numSimplices(*static_cast<const _Mesh *>(this)); }
    size_t numBoundarySimplices() const { return SMT::numBoundarySimplices(*static_cast<const _Mesh *>(this)); }

    template<class _M> using  SHandle = typename SMT::template  SHandle<_M>;
    template<class _M> using BSHandle = typename SMT::template BSHandle<_M>;

     SHandle<      _Mesh>          simplex(size_t i)       { return  SHandle<      _Mesh>(i, *static_cast<      _Mesh *>(this)); }
     SHandle<const _Mesh>          simplex(size_t i) const { return  SHandle<const _Mesh>(i, *static_cast<const _Mesh *>(this)); }
    BSHandle<      _Mesh>  boundarySimplex(size_t i)       { return BSHandle<      _Mesh>(i, *static_cast<      _Mesh *>(this)); }
    BSHandle<const _Mesh>  boundarySimplex(size_t i) const { return BSHandle<const _Mesh>(i, *static_cast<const _Mesh *>(this)); }

    template<template<class> class _Handle> using  HR = HandleRange<      _Mesh, _Handle>;
    template<template<class> class _Handle> using CHR = HandleRange<const _Mesh, _Handle>;

     HR< SHandle>              simplices()       { return  HR< SHandle>(*static_cast<      _Mesh *>(this)); }
    CHR< SHandle>              simplices() const { return CHR< SHandle>(*static_cast<const _Mesh *>(this)); }
    CHR< SHandle>         constSimplices() const { return CHR< SHandle>(*static_cast<const _Mesh *>(this)); }
     HR<BSHandle>      boundarySimplices()       { return  HR<BSHandle>(*static_cast<      _Mesh *>(this)); }
    CHR<BSHandle>      boundarySimplices() const { return CHR<BSHandle>(*static_cast<const _Mesh *>(this)); }
    CHR<BSHandle> constBoundarySimplices() const { return CHR<BSHandle>(*static_cast<const _Mesh *>(this)); }
};

#endif /* end of include guard: SIMPLICIALMESHINTERFACE_HH */
