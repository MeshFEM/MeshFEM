#ifndef PARALLELISM_HH
#define PARALLELISM_HH

#if HAS_TBB
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#endif

#define USE_TBB HAS_TBB

#endif /* end of include guard: PARALLELISM_HH */
