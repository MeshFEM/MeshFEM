#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <MeshFEM/GlobalBenchmark.hh>
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
    m.def("to_dict", []() {
            std::map<std::string, std::pair<double, std::map<std::string, double>>> result;
            for (const auto &sec : g_timer.sections()) {
                result[sec.first] = std::make_pair(sec.second.elapsed(), std::map<std::string, double>());
                auto &secResult = result[sec.first].second;
                for (const auto &t : sec.second.timers)
                    secResult[t.first] = t.second.time;
            }
            return result;
        });
#else
    m.def("to_dict", []() { });
#endif
}
