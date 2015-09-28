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
    operator bool() const { return static_cast<const Subtype *>(this)->valid(); }
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
    operator bool() const { return static_cast<const ConstSubtype *>(this)->valid(); }
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
// actually iterates over the vertex *data* not the vertices themselves
// since range-based for loops apply the "*" operator.
template<class Handle>
class HandleIteratorWrapper : public Handle {
public:
    HandleIteratorWrapper(const Handle &h) : Handle(h) { }
    // Dereference operator just strips away this wrapper
    Handle operator*() const { return Handle(this->m_idx, this->m_mesh); }
};

// Class representing a range of handles [0..entityCount) to be used in a
// ranage-based for.
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
private:
    const mesh_type &m_mesh;
};


#endif /* end of include guard: HANDLE_HH */
