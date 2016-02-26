#ifndef GLOBALBENCHMARK_HH
#define GLOBALBENCHMARK_HH
#include <vector>
#include <string>

#ifdef BENCHMARK
#include <Timer.hh>
void BENCHMARK_START_TIMER_SECTION(const std::string &name);
void  BENCHMARK_STOP_TIMER_SECTION(const std::string &name);
void         BENCHMARK_START_TIMER(const std::string &name);
void          BENCHMARK_STOP_TIMER(const std::string &name);
void BENCHMARK_ADD_MESSAGE(const std::string &msg);
void BENCHMARK_REPORT();
#else
inline void BENCHMARK_START_TIMER_SECTION(const std::string &/* name */) { }
inline void  BENCHMARK_STOP_TIMER_SECTION(const std::string &/* name */) { }
inline void         BENCHMARK_START_TIMER(const std::string &/* name */) { }
inline void          BENCHMARK_STOP_TIMER(const std::string &/* name */) { }
inline void BENCHMARK_ADD_MESSAGE(const std::string &/* msg */) { }
inline void BENCHMARK_REPORT() { }
#endif

#endif /* end of include guard: GLOBALBENCHMARK_HH */
