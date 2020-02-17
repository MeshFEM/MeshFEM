#ifndef MSHFIELDPARSER_BINDINGS_HH
#define MSHFIELDPARSER_BINDINGS_HH

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void bindMSHFieldParser(py::module &m);

#endif /* end of include guard: MSHFIELDPARSER_BINDINGS_HH */
