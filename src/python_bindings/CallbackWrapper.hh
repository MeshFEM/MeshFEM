////////////////////////////////////////////////////////////////////////////////
// CallbackWrapper.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Pybind11 wrappers for C++ callbacks of the form:
//          cb(SenderType &sender, size_t i)
//  where `sender` is the object sending the callback. These wrappers prevent
//  `sender` from being copied upon each invocation, which is inefficient and
//  can lead to `RuntimeErrors` as described below.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
*///////////////////////////////////////////////////////////////////////////////
#ifndef CALLBACKWRAPPER_HH
#define CALLBACKWRAPPER_HH

#include <pybind11/pybind11.h>
namespace py = pybind11; // NOLINT (workaround clang-tidy bug)

// Hack around a limitation of pybind11 where we cannot specify argument passing policies and
// pybind11 tries to make a copy if the passed instance is not already registered:
//      https://github.com/pybind/pybind11/issues/1200
// We therefore make our Python callback interface use a raw pointer to forbid this copy (which
// causes an error since NewtonProblem is not copyable).
template<class SenderType, typename ReturnType=bool>
using PyCallbackFunction = std::function<ReturnType(SenderType *, size_t)>;

template<class SenderType, typename ReturnType=bool>
std::function<ReturnType(SenderType &, size_t)> callbackWrapper(const PyCallbackFunction<SenderType, ReturnType> &pcb) {
    return [pcb](SenderType &p, size_t i) -> ReturnType { if (pcb) return pcb(&p, i); return ReturnType(); }; // Note that the default `ReturnType()` used here is value-initialized and so will be zero for primitives like `bool`.
}

// Fully generic version taking arbitrary args after the sender (though
// the policies of these additional arguments are not controlled).
template<class SenderType, typename ReturnType, typename... Args>
using GenericPyCallbackFunction = std::function<ReturnType(SenderType *, Args...)>;

template<class SenderType, typename ReturnType, typename... Args>
std::function<ReturnType(SenderType &, Args...)> callbackWrapper(const GenericPyCallbackFunction<SenderType, ReturnType, Args...> &pcb) {
    return [pcb](SenderType &p, Args&&... args) -> ReturnType { if (pcb) return pcb(&p, std::forward<Args>(args)...); return ReturnType(); };
}

#endif /* end of include guard: CALLBACKWRAPPER_HH */
