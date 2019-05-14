////////////////////////////////////////////////////////////////////////////////
// Handle.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Handle classes for mesh data structures (e.g. TetMesh and TriMesh).
//		These are index-based bidirectional iterators with some added features.
//		Subclasses will implement entity-dependent traversal.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/26/2014 01:37:40
////////////////////////////////////////////////////////////////////////////////
#ifndef HANDLE_HH
#define HANDLE_HH

#include <type_traits>
#include <cassert>
#include <vector>

// Collection of the handle types for a particular mesh (to be specialized).
// Should provide, e.g:
//      template<class _Mesh> class VHandle;
//      template<class _Mesh> class VRangeTraits;
template<class _Mesh>
struct HandleTraits;

// Information needed to construct a handle range for Handle<_Mesh>.
// (To be specialized).
// Currently this is just:
//      static size_t entityCount(const _Mesh &m) ; // range size
template<class _Handle> struct HandleRangeTraits;

// Special data type that causes no per-entity storage for each entity it is
// assigned to.
// This exact type must be used because empty structs actually have nonzero
// size. By comparing against this type, we explicitly avoid allocating
// instances of it.
struct TMEmptyData { };

template<class Data>
struct DataStorage : public std::vector<Data> {
    using Base = std::vector<Data>;
    using Base::Base;
          Data *getPtr(size_t i)       { return &((*this)[i]); }
    const Data *getPtr(size_t i) const { return &((*this)[i]); }
};

template<>
struct DataStorage<TMEmptyData> {
    void resize(size_t /* i */) { /* Empty! */ }
    TMEmptyData *getPtr(size_t /* i */) const { return NULL; }
};

// DataAccessPolicy:
// Only provide data dereferencing operators if there is actual data (i.e. not
// with TMEmptyData)
// This way, accessing a mesh with TMEmptyData is a compile-time error.
template<class _Mesh, template<class> class Subtype, class Data>
struct DataAccessPolicy {
    // Data is const if mesh is (then this is a const handle).
    static constexpr bool IsConstHandle = std::is_const<_Mesh>::value;
    using value_type = typename std::conditional<IsConstHandle, const Data, Data>::type;
    using value_ptr  = value_type *;

    value_type &operator* () const { return *m_guardedGetPtr(); }
    value_type *operator->() const { return  m_guardedGetPtr(); }
private:
    value_ptr m_guardedGetPtr() const {
        const Subtype<_Mesh> *self = static_cast<const Subtype<_Mesh> *>(this);
        assert(self->valid());
        return self->dataPtr();
    }
};
template<class _Mesh, template<class> class Subtype>
struct DataAccessPolicy<_Mesh, Subtype, TMEmptyData> {
    static constexpr bool IsConstHandle = std::is_const<_Mesh>::value;
    using value_type = typename std::conditional<IsConstHandle, const TMEmptyData, TMEmptyData>::type;
    using value_ptr  = value_type *;
};

// This a regular handle if _Mesh is non-const
// and a const handle if _Mesh is const.
template<class _Mesh, template<class> class Subtype, class Data>
class Handle : public DataAccessPolicy<_Mesh, Subtype, Data> {
public:
    using Mesh = _Mesh;
    using CMesh = typename std::add_const<Mesh>::type;
    using SubHandle =  Subtype<Mesh>;
    using CSubHandle = Subtype<CMesh>;
    using value_ptr = typename DataAccessPolicy<_Mesh, Subtype, Data>::value_ptr;

    Handle(int idx, Mesh &mesh) : m_idx(idx), m_mesh(mesh) { }

    explicit operator bool() const { return static_cast<const SubHandle *>(this)->valid(); }

    // Implement all comparisons/assignments against CSubHandle as we can always
    // cast non-const handles to those.
    bool   sameMesh(const CSubHandle &h) const { return &m_mesh == &(h.mesh()); }
    bool operator==(const CSubHandle &h) const { return sameMesh(h) && index() == h.index(); }
    bool operator!=(const CSubHandle &h) const { return !(*this == h); }

    // Allow assignment between handles on the same mesh
    Handle &operator=(const CSubHandle &h) { assert(sameMesh(h)); m_idx = h.index(); return *this; }
    // The following is required because the default assignment operator is
    // implicitly deleted but *still* participates in overload resolution,
    // hiding the one we want above...
    Handle &operator=(const Handle &h)     { assert(sameMesh(h)); m_idx = h.index(); return *this; }

    Handle &operator++() { ++m_idx; return *this; }
    Handle &operator--() { ++m_idx; return *this; }
    Handle  operator++(int) { Handle old(*this); ++(*this); return old; }
    Handle  operator--(int) { Handle old(*this); --(*this); return old; }

          Mesh &mesh()       { return m_mesh; }
    const Mesh &mesh() const { return m_mesh; }

    int index() const { return m_idx; }

    // Allow casting to const version if non-const
    operator CSubHandle() const { return CSubHandle(m_idx, m_mesh); }
protected:
    int m_idx;
    Mesh &m_mesh;
};

////////////////////////////////////////////////////////////////////////////////
// Handle Ranges: range-based for support
////////////////////////////////////////////////////////////////////////////////

// Create an iterator wrapper for handles to be used in range-based for.
// The problem we're solving is that, if mesh.vertices()
// is a range of handles,
//      for (auto v : mesh.vertices()) {
//          ...
//      }
// actually iterates over the vertex *data* not the vertex handles themselves
// since range-based for loops apply the "*" operator.
template<class DerivedHandle>
class HandleIteratorWrapper : public DerivedHandle {
public:
    HandleIteratorWrapper(const DerivedHandle &h) : DerivedHandle(h) { }
    // Dereference operator just strips away this wrapper
    DerivedHandle operator*() const { return DerivedHandle(this->m_idx, this->m_mesh); }
};

// Class representing a range of handles [0..entityCount) to be used in a
// range-based for.
template<class _Mesh, template<class> class _HType>
struct HandleRange {
    using HType = _HType<_Mesh>;
    using Iterator = HandleIteratorWrapper<HType>;

    HandleRange(_Mesh &mesh) : m_mesh(mesh) { }
    Iterator begin() const { return Iterator(HType(0,      m_mesh)); }
    Iterator   end() const { return Iterator(HType(size(), m_mesh)); }
    size_t    size() const { return HandleRangeTraits<HType>::entityCount(m_mesh); }
private:
    _Mesh &m_mesh;
};

////////////////////////////////////////////////////////////////////////////////
// Sub-Entity Handle Ranges: support range-based for over, e.g. nodes within
// elements
// The sub-entity handle is augmented with a "localIndex", which is the
// sub-entity's index within the collection over which we are iterating.
////////////////////////////////////////////////////////////////////////////////
template<class SEHType>
class SubEntityHandle : public SEHType {
public:
    SubEntityHandle(const SEHType &h, size_t localIndex)
        : SEHType(h), m_localIndex(localIndex) { }

    size_t localIndex() const { return m_localIndex; }
private:
    size_t m_localIndex;
};

template<class _RangeTraits>
class SubEntityHandleIterator {
public:
    using SEH = SubEntityHandle<typename _RangeTraits::SEHType>;
    using EH  = typename _RangeTraits::EHType;
    SubEntityHandleIterator(const EH &h, size_t i) : m_h(h), m_i(i) { }
    SubEntityHandleIterator(const SubEntityHandleIterator &sh) : m_h(sh.m_h), m_i(sh.m_i) { }

    bool operator==(const SubEntityHandleIterator &hi) const { return (m_h == hi.m_h) && (m_i == hi.m_i); }
    bool operator!=(const SubEntityHandleIterator &hi) const { return !(*this == hi); }

    SubEntityHandleIterator &operator++() { ++m_i; return *this; }
    SubEntityHandleIterator &operator--() { ++m_i; return *this; }
    SubEntityHandleIterator  operator++(int) { SubEntityHandleIterator old(*this); ++(*this); return old; }
    SubEntityHandleIterator  operator--(int) { SubEntityHandleIterator old(*this); --(*this); return old; }

    SEH operator*() const { return SEH((m_h .* _RangeTraits::get)(m_i), m_i); }
private:
    EH m_h;
    size_t m_i;
};

// Class representing a range of sub-entity handles [0..count) to be used in a
// range-based for.
// Template param RangeTraits should be a struct with the following
// types/memebers:
//      EHType:      type of entity Handle
//      SEHType:     type of Sub-entity Handle
//      count:       sub-handle collection size (static const! we only support
//                   fixed-sized sub-entity collections for now)
//      getter:      pointer to handle's member function getting the ith sub-entity handle.
template<class _RangeTraits>
struct SubEntityHandleRange {
    using        EH = typename _RangeTraits::EHType;
    using       SEH = typename _RangeTraits::SEHType;
    static_assert(std::is_same<typename  EH::Mesh,
                               typename SEH::Mesh>::value,
        "Entity and sub-entity handles must have same underlying mesh type!");

    SubEntityHandleRange(const EH &h) : m_h(h) { }

    using Iterator = SubEntityHandleIterator<_RangeTraits>;

    Iterator begin() const { return Iterator(m_h, 0); }
    Iterator end()   const { return Iterator(m_h, size()); }
    static constexpr size_t size() { return _RangeTraits::count; }
private:
    EH m_h;
};

#endif /* end of include guard: HANDLE_HH */
