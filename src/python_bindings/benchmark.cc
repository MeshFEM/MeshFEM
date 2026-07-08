#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <MeshFEMCore/GlobalBenchmark.hh>

using namespace MeshFEM;
namespace py = pybind11;

PYBIND11_MODULE(_benchmark, m) {
    m.def("reset", &BENCHMARK_RESET);
    m.def("start_timer_section", &BENCHMARK_START_TIMER_SECTION, py::arg("name"));
    m.def("stop_timer_section",  &BENCHMARK_STOP_TIMER_SECTION,  py::arg("name"));
    m.def("start_timer",         &BENCHMARK_START_TIMER,         py::arg("name"));
    m.def("stop_timer",          &BENCHMARK_STOP_TIMER,          py::arg("name"));
    m.def("report", [](bool includeMessages) {
            py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            if (includeMessages) BENCHMARK_REPORT(); else BENCHMARK_REPORT_NO_MESSAGES();
        },
        py::arg("include_messages") = false);
#ifdef BENCHMARK
    struct BenchmarkRecord {
        BenchmarkRecord(double t, int i) : time(t), invocations(i) { }
        double time;
        int invocations;
        double averageTime() const { return time / invocations; }
    };

    py::class_<BenchmarkRecord>(m, "BenchmarkRecord")
        .def_readonly("time", &BenchmarkRecord::time)
        .def_readonly("invocations", &BenchmarkRecord::invocations)
        .def_property_readonly("averageTime", &BenchmarkRecord::averageTime)
        .def("__repr__", [](const BenchmarkRecord &r) { return std::to_string(r.time) + "s over " + std::to_string(r.invocations) + " invocations (" + std::to_string(r.averageTime()) + "s per invocation)"; })
        .def(py::pickle([](const BenchmarkRecord &r)  { return std::tuple<double, int>(r.time, r.invocations); },
                        [](const std::tuple<double, int> t) { return BenchmarkRecord(std::get<0>(t), std::get<1>(t)); }))
        ;

    m.def("to_dict", []() {
            std::map<std::string, BenchmarkRecord> result;
            for (const auto &sec : g_timer.sections()) {
                result.emplace(sec.first, BenchmarkRecord(sec.second.elapsed(), sec.second.invocations));
                for (const auto &t : sec.second.timers)
                    result.emplace(t.first, BenchmarkRecord(t.second.elapsed(), t.second.invocations));
            }
            return result;
        });
#else
    m.def("to_dict", []() { });
#endif
}
