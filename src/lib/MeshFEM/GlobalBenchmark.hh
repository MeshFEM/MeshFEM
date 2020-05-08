#ifndef GLOBALBENCHMARK_HH
#define GLOBALBENCHMARK_HH
#include <vector>
#include <string>
#include <MeshFEM_export.h>

#ifdef BENCHMARK
#include <MeshFEM/Timer.hh>
MESHFEM_EXPORT void BENCHMARK_START_TIMER_SECTION(const std::string &name);
MESHFEM_EXPORT void  BENCHMARK_STOP_TIMER_SECTION(const std::string &name);
MESHFEM_EXPORT void         BENCHMARK_START_TIMER(const std::string &name);
MESHFEM_EXPORT void          BENCHMARK_STOP_TIMER(const std::string &name);
MESHFEM_EXPORT void               BENCHMARK_RESET();

MESHFEM_EXPORT void BENCHMARK_ADD_MESSAGE(const std::string &msg);
MESHFEM_EXPORT void BENCHMARK_CLEAR_MESSAGES();
MESHFEM_EXPORT void BENCHMARK_REPORT();
MESHFEM_EXPORT void BENCHMARK_REPORT_NO_MESSAGES();
#else
inline void BENCHMARK_START_TIMER_SECTION(const std::string &/* name */) { }
inline void  BENCHMARK_STOP_TIMER_SECTION(const std::string &/* name */) { }
inline void         BENCHMARK_START_TIMER(const std::string &/* name */) { }
inline void          BENCHMARK_STOP_TIMER(const std::string &/* name */) { }
inline void               BENCHMARK_RESET() { }
inline void BENCHMARK_ADD_MESSAGE(const std::string &/* msg */) { }
inline void BENCHMARK_CLEAR_MESSAGES() { }
inline void BENCHMARK_REPORT() { }
inline void BENCHMARK_REPORT_NO_MESSAGES() { }
#endif

struct BENCHMARK_SCOPED_TIMER_SECTION {
    BENCHMARK_SCOPED_TIMER_SECTION(const std::string &name) : m_name(name) {
        BENCHMARK_START_TIMER_SECTION(name);
    }

    ~BENCHMARK_SCOPED_TIMER_SECTION() {
        BENCHMARK_STOP_TIMER_SECTION(m_name);
    }
private:
    std::string m_name;
};


#endif /* end of include guard: GLOBALBENCHMARK_HH */
