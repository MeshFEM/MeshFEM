#ifndef PARALLELISM_HH
#define PARALLELISM_HH

#ifdef MESHFEM_WITH_TBB
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#endif

#endif /* end of include guard: PARALLELISM_HH */
