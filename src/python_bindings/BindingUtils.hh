#ifndef BINDINGUTILS_HH
#define BINDINGUTILS_HH

// Add bindings for pickling and serialization-based cloning via the
// `serialize` and `deserialize` static methods (and `State` type).
// Note: the deserialization calls are always copied safely to the holder type,
// so `C::deserialize` can return a `unique_ptr` even if the holder type is `shared_ptr`
// (and returning a `shared_ptr` should cause a compilation error if the holder
// type is `unique_ptr`).
template<class C, class PyC>
void addSerializationBindings(PyC &pyClass) {
    pyClass.def(py::pickle(&C::serialize, [](const typename C::State &s) { return typename PyC::holder_type(C::deserialize(s)); }))
           .def("clone", [](const C &obj) { return typename PyC::holder_type(C::deserialize(C::serialize(obj))); })
           ;
}

#endif /* end of include guard: BINDINGUTILS_HH */
