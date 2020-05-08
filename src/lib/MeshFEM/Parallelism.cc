#include <MeshFEM/Parallelism.hh>

#ifdef MESHFEM_WITH_TBB

std::unique_ptr<tbb::task_scheduler_init> g_task_scheduler_init;
std::unique_ptr<tbb::task_arena> g_hessian_assembly_arena,
                                 g_gradient_assembly_arena;

////////////////////////////////////////////////////////////////////////////////

void set_max_num_tbb_threads(int num_threads) {
    g_task_scheduler_init = std::make_unique<tbb::task_scheduler_init>(num_threads);
}

void set_hessian_assembly_num_threads(int num_threads) {
    if (!g_hessian_assembly_arena || (g_hessian_assembly_arena->max_concurrency() != num_threads))
        g_hessian_assembly_arena = std::make_unique<tbb::task_arena>(num_threads);
}

void set_gradient_assembly_num_threads(int num_threads) {
    if (!g_gradient_assembly_arena || (g_gradient_assembly_arena->max_concurrency() != num_threads))
        g_gradient_assembly_arena = std::make_unique<tbb::task_arena>(num_threads);
}

////////////////////////////////////////////////////////////////////////////////

tbb::task_arena &get_hessian_assembly_arena() {
    if (!g_hessian_assembly_arena) set_hessian_assembly_num_threads(tbb::task_arena::automatic);
    return *g_hessian_assembly_arena;
}

tbb::task_arena &get_gradient_assembly_arena() {
    if (!g_gradient_assembly_arena) set_gradient_assembly_num_threads(tbb::task_arena::automatic);
    return *g_gradient_assembly_arena;
}

#else // !MESHFEM_WITH_TBB

void set_max_num_tbb_threads(int num_threads) {
    throw std::runtime_error("TBB Disabled");
}

void set_hessian_assembly_num_threads(int num_threads) {
    throw std::runtime_error("TBB Disabled");
}

void set_gradient_assembly_num_threads(int num_threads) {
    throw std::runtime_error("TBB Disabled");
}

#endif // MESHFEM_WITH_TBB
