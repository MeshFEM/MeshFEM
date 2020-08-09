// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "EPS.h"

template <> IGLAABB_INLINE float iglaabb::EPS()
{
  return iglaabb::FLOAT_EPS;
}
template <> IGLAABB_INLINE double iglaabb::EPS()
{
  return iglaabb::DOUBLE_EPS;
}

template <> IGLAABB_INLINE float iglaabb::EPS_SQ()
{
  return iglaabb::FLOAT_EPS_SQ;
}
template <> IGLAABB_INLINE double iglaabb::EPS_SQ()
{
  return iglaabb::DOUBLE_EPS_SQ;
}

#ifdef IGLAABB_STATIC_LIBRARY
// Explicit template instantiation
#endif
