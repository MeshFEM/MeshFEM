////////////////////////////////////////////////////////////////////////////////
// SimplicialMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Wrapper for Tri/Tet mesh to support dimension-independent code.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/09/2016 10:25:32
////////////////////////////////////////////////////////////////////////////////
#ifndef SIMPLICIALMESH_HH
#define SIMPLICIALMESH_HH

#include "TriMesh.hh"
#include "TetMesh.hh"

// Simplicial mesh with optional data to store on:
//       VData: vertices
//       SData: simplices
//      BVData: boundary vertices
//      BSData: boundary simplices
template<size_t _K, class VData = TMEmptyData, class SData = TMEmptyData, class BVData = TMEmptyData, class BSData = TMEmptyData>
struct SimplicialMesh;

////////////////////////////////////////////////////////////////////////////////
// 2D case: TriMesh
////////////////////////////////////////////////////////////////////////////////
template<class VData, class SData, class BVData, class BSData>
struct SimplicialMesh<2, VData, SData, BVData, BSData> : public TriMesh<VData, TMEmptyData, SData, BVData, BSData> {
    using Base = TriMesh<VData, TMEmptyData, SData, BVData, BSData>;
    using Base::Base;

    // Simplex/vertex ranges
private:
    // Specialization for nested class templates isn't allowed, so we can't
    // implement a true traits design pattern...
    struct  VRangeTraits { using HType = typename Base::        VertexHandle; using CHType = typename Base::        ConstVertexHandle; static constexpr size_t (Base::*entityCount)() const = &Base::numVertices; };
    struct  SRangeTraits { using HType = typename Base::           TriHandle; using CHType = typename Base::           ConstTriHandle; static constexpr size_t (Base::*entityCount)() const = &Base::numTris; };
    struct BVRangeTraits { using HType = typename Base::BoundaryVertexHandle; using CHType = typename Base::ConstBoundaryVertexHandle; static constexpr size_t (Base::*entityCount)() const = &Base::numBoundaryVertices; };
    struct BSRangeTraits { using HType = typename Base::  BoundaryEdgeHandle; using CHType = typename Base::  ConstBoundaryEdgeHandle; static constexpr size_t (Base::*entityCount)() const = &Base::numBoundaryEdges; };
public:
         HandleRange< VRangeTraits>          vertices()       { return      HandleRange< VRangeTraits>(*this); }
    ConstHandleRange< VRangeTraits>          vertices() const { return ConstHandleRange< VRangeTraits>(*this); }
         HandleRange< SRangeTraits>         simplices()       { return      HandleRange< SRangeTraits>(*this); }
    ConstHandleRange< SRangeTraits>         simplices() const { return ConstHandleRange< SRangeTraits>(*this); }
         HandleRange<BVRangeTraits>  boundaryVertices()       { return      HandleRange<BVRangeTraits>(*this); }
    ConstHandleRange<BVRangeTraits>  boundaryVertices() const { return ConstHandleRange<BVRangeTraits>(*this); }
         HandleRange<BSRangeTraits> boundarySimplices()       { return      HandleRange<BSRangeTraits>(*this); }
    ConstHandleRange<BSRangeTraits> boundarySimplices() const { return ConstHandleRange<BSRangeTraits>(*this); }
};

////////////////////////////////////////////////////////////////////////////////
// 3D case: TetMesh
////////////////////////////////////////////////////////////////////////////////
template<class VData, class SData, class BVData, class BSData>
struct SimplicialMesh<3, VData, SData, BVData, BSData> : public TetMesh<VData, TMEmptyData, SData, BVData, TMEmptyData, BSData> {
    using Base = TetMesh<VData, TMEmptyData, SData, BVData, TMEmptyData, BSData>;
    using Base::Base;

    // Simplex/vertex ranges
private:
    // Specialization for nested class templates isn't allowed, so we can't
    // implement a true traits design pattern...
    struct  VRangeTraits { using HType = typename Base::        VertexHandle; using CHType = typename Base::        ConstVertexHandle; static constexpr size_t (Base::*entityCount)() const = &Base::numVertices; };
    struct  SRangeTraits { using HType = typename Base::           TetHandle; using CHType = typename Base::           ConstTetHandle; static constexpr size_t (Base::*entityCount)() const = &Base::numTets; };
    struct BVRangeTraits { using HType = typename Base::BoundaryVertexHandle; using CHType = typename Base::ConstBoundaryVertexHandle; static constexpr size_t (Base::*entityCount)() const = &Base::numBoundaryVertices; };
    struct BSRangeTraits { using HType = typename Base::  BoundaryFaceHandle; using CHType = typename Base::  ConstBoundaryFaceHandle; static constexpr size_t (Base::*entityCount)() const = &Base::numBoundaryFaces; };
public:
         HandleRange< VRangeTraits>          vertices()       { return      HandleRange< VRangeTraits>(*this); }
    ConstHandleRange< VRangeTraits>          vertices() const { return ConstHandleRange< VRangeTraits>(*this); }
         HandleRange< SRangeTraits>         simplices()       { return      HandleRange< SRangeTraits>(*this); }
    ConstHandleRange< SRangeTraits>         simplices() const { return ConstHandleRange< SRangeTraits>(*this); }
         HandleRange<BVRangeTraits>  boundaryVertices()       { return      HandleRange<BVRangeTraits>(*this); }
    ConstHandleRange<BVRangeTraits>  boundaryVertices() const { return ConstHandleRange<BVRangeTraits>(*this); }
         HandleRange<BSRangeTraits> boundarySimplices()       { return      HandleRange<BSRangeTraits>(*this); }
    ConstHandleRange<BSRangeTraits> boundarySimplices() const { return ConstHandleRange<BSRangeTraits>(*this); }
};

#endif /* end of include guard: SIMPLICIALMESH_HH */
