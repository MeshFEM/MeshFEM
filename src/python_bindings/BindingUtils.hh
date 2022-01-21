#ifndef BINDINGUTILS_HH
#define BINDINGUTILS_HH

#include <pybind11/pybind11.h>
namespace py = pybind11;

// Recursive metafunction that tries casting the python unpickled type (py::tuple)
// to each C++ state class in a list of supported state classes (proceeding
// from left to right). If all casts fail, the cast failure exception for the
// last attempt is thrown.
// This functionality is to provide  backwards compatibility when new state
// fields are added after pickle files have already been generated.
// See, e.g., the bindings for NewtonOptimizerOptions.
template<class C, class PyC, class... States>
struct DeserializeBackwardsCompatible;

template<class C, class PyC, class State>
struct DeserializeBackwardsCompatible<C, PyC, State> {
    static typename PyC::holder_type run(const py::object &t) {
        return C::deserialize(t.cast<State>());
    }
};

template<class C, class PyC, class State, class BCState0, class... BCStates>
struct DeserializeBackwardsCompatible<C, PyC, State, BCState0, BCStates...> {
    static typename PyC::holder_type run(const py::object &t) {
        try { return DeserializeBackwardsCompatible<C, PyC, State>::run(t); }
        catch (...) {
            return DeserializeBackwardsCompatible<C, PyC, BCState0, BCStates...>::run(t);
        }
    }
};

// Add bindings for pickling and serialization-based cloning via the
// `serialize` and `deserialize` static methods (and `State` type).
// Note: the deserialization calls are always copied safely to the holder type,
// so `C::deserialize` can return a `unique_ptr` even if the holder type is `shared_ptr`
// (and returning a `shared_ptr` should cause a compilation error if the holder
// type is `unique_ptr`).
template<class C, class PyC, class... BackwardsCompatibilityStates>
void addSerializationBindings(PyC &pyClass) {
    pyClass.def(py::pickle([](const C &obj) { return py::cast(C::serialize(obj)); },
                           &DeserializeBackwardsCompatible<C, PyC, typename C::State, BackwardsCompatibilityStates...>::run))
           // This clone operation is actually quite dangerous since it does *not* necessarily perform a deep copy
           // unlike pickling/unpickling: if the serialized state contains pointers, the pointed-to objects will
           // be shared across clones.
           // .def("clone", [](const C &obj) { return typename PyC::holder_type(C::deserialize(C::serialize(obj))); })
           ;
}

#endif /* end of include guard: BINDINGUTILS_HH */
