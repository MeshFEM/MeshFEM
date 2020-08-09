////////////////////////////////////////////////////////////////////////////////
// parallel_for.h
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Reimplementation of libigl's parallel_for as a wrapper for tbb
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/08/2020 15:17:14
////////////////////////////////////////////////////////////////////////////////
#ifndef IGLAABB_PARALLEL_FOR_H
#define IGLAABB_PARALLEL_FOR_H

#include <MeshFEM/Parallelism.hh>

namespace iglaabb {

template<typename Index, typename FunctionType>
bool parallel_for(const Index loop_size, 
                  const FunctionType &func,
                  const size_t min_parallel=0)
{
    if (loop_size < Index(min_parallel)) {
        for (Index i = 0; i < loop_size; ++i)
            func(i);
        return false;
    }
    else {
        ::parallel_for_range(loop_size, func);
        return true;
    }
}

} // namespace igl

#endif /* end of include guard: IGLAABB_PARALLEL_FOR_H */
