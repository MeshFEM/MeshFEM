////////////////////////////////////////////////////////////////////////////////
// SimplicialMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Metafunction to select between Tet/Tri mesh with optional data on the
//      (vol, boundary) (vertices, simplices) in dimension-independent code.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/09/2016 10:25:32
////////////////////////////////////////////////////////////////////////////////
#ifndef SIMPLICIALMESH_HH
#define SIMPLICIALMESH_HH

#include <MeshFEM/TriMesh.hh>
#include <MeshFEM/TetMesh.hh>

// Metafunction to choose simplicial mesh (Tri/Tet) based on dimension.
template<size_t _K, class  VData = TMEmptyData, class  SData = TMEmptyData,
                    class BVData = TMEmptyData, class BSData = TMEmptyData>
struct SimplicialMeshSelector;

// Simplicial mesh type with optional data to store on:
//       VData: vertices
//       SData: simplices
//      BVData: boundary vertices
//      BSData: boundary simplices
template<size_t _K, class  VData = TMEmptyData, class  SData = TMEmptyData,
                    class BVData = TMEmptyData, class BSData = TMEmptyData>
using SimplicialMesh = typename SimplicialMeshSelector<_K, VData, SData, BVData, BSData>::type;

// 2D case: TriMesh
template<class VData, class SData, class BVData, class BSData>
struct SimplicialMeshSelector<2, VData, SData, BVData, BSData> {
    using type = TriMesh<VData, TMEmptyData, SData, BVData, BSData>;
};

// 3D case: TetMesh
template<class VData, class SData, class BVData, class BSData>
struct SimplicialMeshSelector<3, VData, SData, BVData, BSData> {
    using type = TetMesh<VData, TMEmptyData, TMEmptyData, SData, BVData, TMEmptyData, BSData>;
};

#endif /* end of include guard: SIMPLICIALMESH_HH */
