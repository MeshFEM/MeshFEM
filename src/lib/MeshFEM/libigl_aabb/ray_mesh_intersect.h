// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2016 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGLAABB_RAY_MESH_INTERSECT_H
#define IGLAABB_RAY_MESH_INTERSECT_H
#include "igl_inline.h"
#include "Hit.h"
#include <Eigen/Core>
#include <vector>
namespace iglaabb
{
  // Shoot a ray against a mesh (V,F) and collect all hits.
  //
  // Inputs:
  //   source  3-vector origin of ray
  //   dir  3-vector direction of ray
  //   V  #V by 3 list of mesh vertex positions
  //   F  #F by 3 list of mesh face indices into V
  // Outputs:
  //    hits  **sorted** list of hits
  // Returns true if there were any hits (hits.size() > 0)
  //
  template <
    typename Derivedsource,
    typename Deriveddir,
    typename DerivedV, 
    typename DerivedF> 
  IGLAABB_INLINE bool ray_mesh_intersect(
    const Eigen::MatrixBase<Derivedsource> & source,
    const Eigen::MatrixBase<Deriveddir> & dir,
    const Eigen::MatrixBase<DerivedV> & V,
    const Eigen::MatrixBase<DerivedF> & F,
    std::vector<iglaabb::Hit> & hits);
  // Outputs:
  //   hit  first hit, set only if it exists
  // Returns true if there was a hit
  template <
    typename Derivedsource,
    typename Deriveddir,
    typename DerivedV, 
    typename DerivedF> 
  IGLAABB_INLINE bool ray_mesh_intersect(
    const Eigen::MatrixBase<Derivedsource> & source,
    const Eigen::MatrixBase<Deriveddir> & dir,
    const Eigen::MatrixBase<DerivedV> & V,
    const Eigen::MatrixBase<DerivedF> & F,
    iglaabb::Hit & hit);
}
#ifndef IGLAABB_STATIC_LIBRARY
#  include "ray_mesh_intersect.cpp"
#endif
#endif
