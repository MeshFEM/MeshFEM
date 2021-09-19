// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
// This should *NOT* be contained in a IGLAABB_*_H ifdef, since it may be defined
// differently based on when it is included
#ifdef IGLAABB_INLINE
#undef IGLAABB_INLINE
#endif

#ifndef IGLAABB_STATIC_LIBRARY
#  define IGLAABB_INLINE inline
#else
#  define IGLAABB_INLINE
#endif
