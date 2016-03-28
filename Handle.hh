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

// Special data type that causes no per-entity storage for each entity it is
// assigned to.
// This exact type must be used because empty structs actually have nonzero
// size. By comparing against this type, we explicitly avoid allocating
// instances of it.
class TMEmptyData { };

// Forward declare ConstHandle since Handle must referece it.
template<class Mesh, class Subtype, class ConstSubtype, class Data>
class ConstHandle;

template<class Mesh, class Subtype, class ConstSubtype, class Data>
class Handle {
public:
    typedef Mesh  mesh_type;
    typedef Data  value_type;
    typedef Data *value_ptr;
    typedef ConstHandle<Mesh, Subtype, ConstSubtype, Data> _ConstHandle;

    Handle(int idx, Mesh &mesh) : m_idx(idx), m_mesh(mesh) { }
    explicit operator bool() const { return static_cast<const Subtype *>(this)->valid(); }
    bool sameMesh(const Handle &h)         const { return &m_mesh == &(h.mesh()); }
    bool sameMesh(const _ConstHandle &h)   const { return &m_mesh == &(h.mesh()); }
    bool operator==(const Handle &h)       const { return sameMesh(h) && index() == h.index(); }
    bool operator==(const _ConstHandle &h) const { return sameMesh(h) && index() == h.index(); }
    bool operator!=(const Handle &h)       const { return !(*this == h); }
    bool operator!=(const _ConstHandle &h) const { return !(*this == h); }

    // Allow assignment between handles on the same mesh
    Handle &operator=(const Handle &h)       { assert(sameMesh(h)); m_idx = h.index(); return *this; }
    Handle &operator=(const _ConstHandle &h) { assert(sameMesh(h)); m_idx = h.index(); return *this; }
    Handle &operator++() { ++m_idx; return *this; }
    Handle &operator--() { ++m_idx; return *this; }
    Handle &operator++(int) { Handle old(*this); ++(*this); return old; }
    Handle &operator--(int) { Handle old(*this); --(*this); return old; }

    value_type &operator*()  const { return *m_guardedGetPtr(); }
    value_type *operator->() const { return  m_guardedGetPtr(); }

          Mesh &mesh()       { return m_mesh; }
    const Mesh &mesh() const { return m_mesh; }

    int index() const { return m_idx; }
    operator ConstSubtype() const { return ConstSubtype(m_idx, m_mesh); }
protected:
    Data *m_guardedGetPtr() const {
        const Subtype *self = static_cast<const Subtype *>(this);
        assert(self->valid());
        if (std::is_same<Data, TMEmptyData>::value)
            return reinterpret_cast<Data *>(&m_mesh.m_emptyDataDummy);
        return self->dataPtr();
    }

    int m_idx;
    Mesh &m_mesh;
};

template<class Mesh, class Subtype, class ConstSubtype, class Data>
class ConstHandle {
public:
    typedef Mesh  mesh_type;
    typedef Data value_type;
    typedef const Data *value_ptr;
    typedef Handle<Mesh, Subtype, ConstSubtype, Data> _Handle;

    ConstHandle(int idx, const Mesh &mesh) : m_idx(idx), m_mesh(mesh) { }
    explicit operator bool() const { return static_cast<const ConstSubtype *>(this)->valid(); }
    bool sameMesh(const _Handle &h)       const { return &m_mesh == &(h.mesh()); }
    bool sameMesh(const ConstHandle &h)   const { return &m_mesh == &(h.mesh()); }
    bool operator==(const _Handle &h)     const { return sameMesh(h) && index() == h.index(); }
    bool operator==(const ConstHandle &h) const { return sameMesh(h) && index() == h.index(); }
    bool operator!=(const _Handle &h)     const { return !(*this == h); }
    bool operator!=(const ConstHandle &h) const { return !(*this == h); }

    // Allow assignment between handles on the same mesh
    ConstHandle &operator=(const _Handle &h)     { assert(sameMesh(h)); m_idx = h.index(); return *this; }
    ConstHandle &operator=(const ConstHandle &h) { assert(sameMesh(h)); m_idx = h.index(); return *this; }
    ConstHandle &operator++() { assert(bool(*this)); ++m_idx; return *this; }
    ConstHandle &operator--() { assert(bool(*this)); ++m_idx; return *this; }
    ConstHandle &operator++(int) { assert(bool(*this)); ConstHandle old(*this); ++(*this); return old; }
    ConstHandle &operator--(int) { assert(bool(*this)); ConstHandle old(*this); --(*this); return old; }

    const value_type &operator*()  const { return *m_guardedGetPtr(); }
    const value_type *operator->() const { return  m_guardedGetPtr(); }

    const Mesh &mesh() const { return m_mesh; }

    int index() const { return m_idx; }
protected:
    const Data *m_guardedGetPtr() const {
        const ConstSubtype *self = static_cast<const ConstSubtype *>(this);
        assert(self->valid());
        if (std::is_same<Data, TMEmptyData>::value)
            return reinterpret_cast<const Data *>(&m_mesh.m_emptyDataDummy);
        return self->dataPtr();
    }

    int m_idx;
    const Mesh &m_mesh;
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
template<class Handle>
class HandleIteratorWrapper : public Handle {
public:
    HandleIteratorWrapper(const Handle &h) : Handle(h) { }
    // Dereference operator just strips away this wrapper
    Handle operator*() const { return Handle(this->m_idx, this->m_mesh); }
};

// Class representing a range of handles [0..entityCount) to be used in a
// range-based for.
// Template param RangeTraits should be a struct with the following
// types/memebers:
//      HType:       typedef of Handle type
//      CHType:      typedef of ConstHandle type (with same mesh_type as HType)
//      entityCount: pointer to memember function getting the size of the handle
//                   collection
template<typename RangeTraits>
struct HandleRange {
    typedef typename RangeTraits::HType::mesh_type mesh_type;
    static_assert(std::is_same<mesh_type, typename RangeTraits::CHType::mesh_type>::value,
            "Handles must have same underlying mesh type!");

    HandleRange(mesh_type &mesh) : m_mesh(mesh) { }
    typedef typename RangeTraits::HType HType;
    typedef HandleIteratorWrapper<HType> Iterator;
    Iterator begin() const { return Iterator(HType(0, m_mesh)); }
    Iterator end()   const { return Iterator(HType((m_mesh .* RangeTraits::entityCount)(), m_mesh)); }
    size_t   size()  const { return (m_mesh .* RangeTraits::entityCount)(); }
private:
    mesh_type &m_mesh;
};
template<typename RangeTraits>
struct ConstHandleRange {
    typedef typename RangeTraits::HType::mesh_type mesh_type;
    static_assert(std::is_same<mesh_type, typename RangeTraits::CHType::mesh_type>::value,
            "Handles must have same underlying mesh type!");

    ConstHandleRange(const mesh_type &mesh) : m_mesh(mesh) { }
    typedef typename RangeTraits::CHType CHType;
    typedef HandleIteratorWrapper<CHType> Iterator;
    Iterator begin() const { return Iterator(CHType(0, m_mesh)); }
    Iterator end()   const { return Iterator(CHType((m_mesh .* RangeTraits::entityCount)(), m_mesh)); }
    size_t   size()  const { return (m_mesh .* RangeTraits::entityCount)(); }
private:
    const mesh_type &m_mesh;
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
    SubEntityHandleIterator &operator++(int) { SubEntityHandleIterator old(*this); ++(*this); return old; }
    SubEntityHandleIterator &operator--(int) { SubEntityHandleIterator old(*this); --(*this); return old; }

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
    static_assert(std::is_same<typename  EH::mesh_type,
                               typename SEH::mesh_type>::value,
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
