#ifndef PARALLELISM_HH
#define PARALLELISM_HH

#include <stddef.h>

#ifdef MESHFEM_WITH_TBB
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#include <cstdint> // Must be brought in before `info.h` to work around OneAPI 2021.4.0 TBB issue.
#include <tbb/info.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/combinable.h>
#pragma GCC diagnostic pop

#include <memory>

#include <MeshFEM_export.h>

MESHFEM_EXPORT void   set_max_num_tbb_threads(int num_threads);
MESHFEM_EXPORT int    get_max_num_tbb_threads();
MESHFEM_EXPORT void unset_max_num_tbb_threads();

// We may want to use different numbers of threads to assemble the Hessian/gradient because of the
// overhead of the reduction operation used to combine the results.
MESHFEM_EXPORT void set_hessian_assembly_num_threads(int num_threads);
MESHFEM_EXPORT void set_gradient_assembly_num_threads(int num_threads);

MESHFEM_EXPORT tbb::task_arena &get_hessian_assembly_arena();
MESHFEM_EXPORT tbb::task_arena &get_gradient_assembly_arena();

template<typename Partitioner = tbb::auto_partitioner, typename F>
void parallel_for_range(size_t start, size_t end, F &&f, size_t grain_size = 1) {
    if (get_max_num_tbb_threads() == 1) {
        for (size_t i = start; i < end; ++i)
            f(i);
        return;
    }
    else {
        tbb::parallel_for(tbb::blocked_range<size_t>(start, end, grain_size),
                          [&f](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i < r.end(); ++i)
                f(i);
        }, Partitioner());
    }
}

template<typename Partitioner = tbb::auto_partitioner, typename F>
void parallel_for_range(size_t n, F &&f) {
    parallel_for_range<Partitioner>(0, n, f);
}

template<typename Partitioner = tbb::auto_partitioner, typename F>
void parallel_for_range(size_t n, F &&f, size_t grain_size, size_t parallelism_threshold) {
    if (n >= parallelism_threshold)
        parallel_for_range<Partitioner>(0, n, f, grain_size);
    else {
        for (size_t i = 0; i < n; ++i)
            f(i);
    }
}


#include <map>
MESHFEM_EXPORT std::map<int, std::vector<int>> parse_cpu_topology();

#ifdef __linux__
#include <tbb/task_scheduler_observer.h>
#include <iostream>
struct PinningObserver : public tbb::task_scheduler_observer {
    // If `spread` is `true`, we ecmulate `OMP_PROC_BIND=spread`.
    PinningObserver(bool spread = true) : m_core_to_logical_map(parse_cpu_topology()), m_spread(spread) {
        observe(true); // Activate the observer
    }

    void on_scheduler_entry(bool) override {
        int thread_index = tbb::this_task_arena::current_thread_index();
        if (thread_index < 0) return; // Not a worker thread

        int num_physical_cores = m_core_to_logical_map.size();
        int num_logical_per_core = m_core_to_logical_map.begin()->second.size();
        int num_threads = get_max_num_tbb_threads();

        int stride = 1;
        int logical_processor;
        if (m_spread && (num_physical_cores > num_threads)) {
            // Emulate OMP_PROC_BIND=spread `https://www.openmp.org/spec-html/5.0/openmpsu36.html#x56-900002.6.2`
            // (Divide the available cores into `num_threads` partitions, assigning one thread to each partition.
            int smallest_partition_size = num_physical_cores / num_threads;
            int remainder = num_physical_cores % num_threads;
            int assigned_core = smallest_partition_size * thread_index + std::min(thread_index, remainder);
            // std::cout << "num_threads: " << num_threads;
            // std::cout << "num_physical_cores: " << num_physical_cores;
            // std::cout << "smallest_partition_size: " << smallest_partition_size;
            // std::cout << "assigned_core: " << assigned_core;
            logical_processor = m_core_to_logical_map[assigned_core][0];
            // std::cout << "logical_processor: " << logical_processor;
        }
        else {
            int assigned_core = (thread_index % num_physical_cores);   // Assign to consecutive physical cores
            int assigned_sibling = (thread_index / num_physical_cores) % num_logical_per_core; // Use hyperthreads last
            logical_processor = m_core_to_logical_map[assigned_core][assigned_sibling];
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(logical_processor, &cpuset);

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    }
private:
    std::map<int, std::vector<int>> m_core_to_logical_map;
    bool m_spread = false;
};
#else
struct PinningObserver {
    PinningObserver(bool /* spread */ = true) {
        // Setting thread affinities is not supported on Apple Silicon: https://developer.apple.com/forums/thread/703361?answerId=709279022#709279022
        std::cout << "WARNING: pinning threads to cores is only supported on Linux." << std::endl;
    }
};
#endif

#else // !MESHFEM_WITH_TBB

// Dummy serial implementation
template<typename F>
void parallel_for_range(size_t n, F &&f) {
    for (size_t i = 0; i < n; ++i)
        f(i);
}

#endif
#endif /* end of include guard: PARALLELISM_HH */
