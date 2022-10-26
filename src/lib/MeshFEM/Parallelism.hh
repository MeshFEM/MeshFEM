#ifndef PARALLELISM_HH
#define PARALLELISM_HH

#ifdef MESHFEM_WITH_TBB
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/combinable.h>
#pragma GCC diagnostic pop

#include <memory>

#include <MeshFEM_export.h>

MESHFEM_EXPORT void   set_max_num_tbb_threads(int num_threads);
MESHFEM_EXPORT void unset_max_num_tbb_threads();

// We may want to use different numbers of threads to assemble the Hessian/gradient because of the
// overhead of the reduction operation used to combine the results.
MESHFEM_EXPORT void set_hessian_assembly_num_threads(int num_threads);
MESHFEM_EXPORT void set_gradient_assembly_num_threads(int num_threads);

MESHFEM_EXPORT tbb::task_arena &get_hessian_assembly_arena();
MESHFEM_EXPORT tbb::task_arena &get_gradient_assembly_arena();

#if 1
template<typename Partitioner = tbb::auto_partitioner, typename F>
void parallel_for_range(size_t start, size_t end, F &&f) {
    tbb::parallel_for(tbb::blocked_range<size_t>(start, end),
                      [&f](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); ++i)
            f(i);
    }, Partitioner());
}

template<typename Partitioner = tbb::auto_partitioner, typename F>
void parallel_for_range(size_t n, F &&f) {
    parallel_for_range<Partitioner>(0, n, f);
}

#else
template<typename F>
void parallel_for_range(size_t n, F &&f) {
    for (size_t i = 0; i < n; ++i)
        f(i);
}
#endif

#endif
#endif /* end of include guard: PARALLELISM_HH */
