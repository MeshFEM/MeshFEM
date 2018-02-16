// TODO: decide how to include MeshFEM's GlobalBenchmark.hh
// (Should probably refactor all the shared code...)
#ifndef GLOBALBENCHMARK_HH
#define GLOBALBENCHMARK_HH

inline void BENCHMARK_START_TIMER_SECTION(const std::string &/* name */) { }
inline void  BENCHMARK_STOP_TIMER_SECTION(const std::string &/* name */) { }
inline void         BENCHMARK_START_TIMER(const std::string &/* name */) { }
inline void          BENCHMARK_STOP_TIMER(const std::string &/* name */) { }
inline void BENCHMARK_ADD_MESSAGE(const std::string &/* msg */) { }
inline void BENCHMARK_REPORT() { }

#endif /* end of include guard: GLOBALBENCHMARK_HH */
