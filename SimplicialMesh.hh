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

// Mesh with optional data to store on
//       VData: vertices
//       SData: simplices
//      BVData: boundary vertices
//      BSData: boundary simplices
template<size_t _K, class VData = TMEmptyData, class SData = TMEmptyData, class BVData = TMEmptyData, class BSData = TMEmptyData>
struct SimplicialMesh;

template<class VData, class SData, class BVData, class BSData>
struct SimplicialMesh<2, VData, SData, BVData, BSData> : public TriMesh<VData, TMEmptyData, SData, BVData, BSData> {
    using Base = TriMesh<VData, TMEmptyData, SData, BVData, BSData>;
    using Base::Base;
};

template<class VData, class SData, class BVData, class BSData>
struct SimplicialMesh<3, VData, SData, BVData, BSData> : public TetMesh<VData, TMEmptyData, SData, BVData, TMEmptyData, BSData> {
    using Base = TetMesh<VData, TMEmptyData, SData, BVData, TMEmptyData, BSData>;
    using Base::Base;
};

#endif /* end of include guard: SIMPLICIALMESH_HH */
