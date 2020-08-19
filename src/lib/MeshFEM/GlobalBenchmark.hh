#ifndef GLOBALBENCHMARK_HH
#define GLOBALBENCHMARK_HH
#include <vector>
#include <string>
#include <iostream>
#include <MeshFEM_export.h>

#ifdef BENCHMARK
#include <MeshFEM/Timer.hh>

MESHFEM_EXPORT extern Timer g_timer;
MESHFEM_EXPORT extern std::vector<std::string> g_benchmarkMessages;

inline void BENCHMARK_START_TIMER_SECTION(const std::string &name) { g_timer.startSection(name); }
inline void  BENCHMARK_STOP_TIMER_SECTION(const std::string &name) { g_timer.stopSection(name); }
inline void         BENCHMARK_START_TIMER(const std::string &name) { g_timer.start(name); }
inline void          BENCHMARK_STOP_TIMER(const std::string &name) { g_timer.stop(name); }
inline void               BENCHMARK_RESET()                   { g_timer.reset(); }

inline void BENCHMARK_ADD_MESSAGE(const std::string &msg) {
    g_benchmarkMessages.push_back(msg);
}

inline void BENCHMARK_CLEAR_MESSAGES() { g_benchmarkMessages.clear(); }

inline void BENCHMARK_REPORT() {
    for (const auto &message : g_benchmarkMessages)
        std::cout << message << std::endl;
    g_timer.report(std::cout);
}

inline void BENCHMARK_REPORT_NO_MESSAGES() {
    g_timer.report(std::cout);
}
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
