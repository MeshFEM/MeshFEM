#include <MeshFEM/Parallelism.hh>
#include <iostream>

#ifdef MESHFEM_WITH_TBB

std::unique_ptr<tbb::global_control> g_global_control;
std::unique_ptr<tbb::task_arena> g_hessian_assembly_arena,
                                 g_gradient_assembly_arena;

////////////////////////////////////////////////////////////////////////////////

void set_max_num_tbb_threads(int num_threads) {
    if (num_threads < 1) throw std::runtime_error("num_threads must be >= 1");
    g_global_control = std::make_unique<tbb::global_control>(tbb::global_control::parameter::max_allowed_parallelism, num_threads);
}

int get_max_num_tbb_threads() {
    if (!g_global_control) return tbb::this_task_arena::max_concurrency();
    return std::min<int>(tbb::this_task_arena::max_concurrency(), g_global_control->active_value(tbb::global_control::parameter::max_allowed_parallelism));
}

void unset_max_num_tbb_threads() {
    g_global_control.reset();
}

static void validateNumThreads(int num_threads) {
    if ((num_threads < 1) && (num_threads != tbb::task_arena::automatic))
        throw std::runtime_error("num_threads must be >= 1");
}

void set_hessian_assembly_num_threads(int num_threads) {
    validateNumThreads(num_threads);
    if (!g_hessian_assembly_arena || (g_hessian_assembly_arena->max_concurrency() != num_threads))
        g_hessian_assembly_arena = std::make_unique<tbb::task_arena>(num_threads);
}

void set_gradient_assembly_num_threads(int num_threads) {
    validateNumThreads(num_threads);
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

#include <map>
#include <fstream>
std::map<int, std::vector<int>> parse_cpu_topology() {
#ifdef __linux__
    std::map<int, std::vector<int>> core_to_logical_map; // The logical cores corresponding to each physical core.
    std::ifstream infile("/proc/cpuinfo");
    std::string line;
    int cpu_id = -1, core_id = -1;

    while (std::getline(infile, line)) {
        if (line.find("processor") == 0) {
            cpu_id = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("core id") == 0) {
            core_id = std::stoi(line.substr(line.find(":") + 1));
            core_to_logical_map[core_id].push_back(cpu_id);
        }
    }
    return core_to_logical_map;
#else
    throw std::runtime_error("parse_cpu_topology is only supported on Linux!");
#endif
}

#else // !MESHFEM_WITH_TBB

void set_max_num_tbb_threads(int num_threads) {
    throw std::runtime_error("TBB Disabled");
}

int get_max_num_tbb_threads() {
    throw std::runtime_error("TBB Disabled");
}

void set_hessian_assembly_num_threads(int num_threads) {
    throw std::runtime_error("TBB Disabled");
}

void set_gradient_assembly_num_threads(int num_threads) {
    throw std::runtime_error("TBB Disabled");
}

#endif // MESHFEM_WITH_TBB
