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
        if (typeid(Data) == typeid(TMEmptyData))
            return reinterpret_cast<Data *>(&m_mesh.m_emptyDataDummy);
        return self->dataPtr();
    }

    int m_idx;
    Mesh &m_mesh;
};

template<class Mesh, class Subtype, class ConstSubtype, class Data>
class ConstHandle {
public:
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
        if (typeid(Data) == typeid(TMEmptyData))
            return reinterpret_cast<const Data *>(&m_mesh.m_emptyDataDummy);
        return self->dataPtr();
    }

    int m_idx;
    const Mesh &m_mesh;
};

#endif /* end of include guard: HANDLE_HH */
