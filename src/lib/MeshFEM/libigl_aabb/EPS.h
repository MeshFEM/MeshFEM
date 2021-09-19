// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGLAABB_EPS_H
#define IGLAABB_EPS_H
#include "igl_inline.h"
namespace iglaabb
{
  // Define a standard value for double epsilon
  const double DOUBLE_EPS    = 1.0e-14;
  const double DOUBLE_EPS_SQ = 1.0e-28;
  const float FLOAT_EPS    = 1.0e-7f;
  const float FLOAT_EPS_SQ = 1.0e-14f;
  // Function returning EPS for corresponding type
  template <typename S_type> IGLAABB_INLINE S_type EPS();
  template <typename S_type> IGLAABB_INLINE S_type EPS_SQ();
  // Template specializations for float and double
  template <> IGLAABB_INLINE float EPS<float>();
  template <> IGLAABB_INLINE double EPS<double>();
  template <> IGLAABB_INLINE float EPS_SQ<float>();
  template <> IGLAABB_INLINE double EPS_SQ<double>();
}

#ifndef IGLAABB_STATIC_LIBRARY
#  include "EPS.cpp"
#endif

#endif
