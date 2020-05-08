#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <MeshFEM/GlobalBenchmark.hh>
namespace py = pybind11;

PYBIND11_MODULE(benchmark, m) {
    m.def("reset", &BENCHMARK_RESET);
    m.def("start_timer_section", &BENCHMARK_START_TIMER_SECTION, py::arg("name"));
    m.def("stop_timer_section",  &BENCHMARK_STOP_TIMER_SECTION,  py::arg("name"));
    m.def("start_timer",         &BENCHMARK_START_TIMER,         py::arg("name"));
    m.def("stop_timer",          &BENCHMARK_STOP_TIMER,          py::arg("name"));
    m.def("report", [](bool includeMessages) {
            py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            if (includeMessages) BENCHMARK_REPORT(); else BENCHMARK_REPORT_NO_MESSAGES();
        },
        py::arg("include_messages") = false)
        ;
}
