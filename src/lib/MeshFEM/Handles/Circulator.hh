////////////////////////////////////////////////////////////////////////////////
// Circulator.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Provide range-based for support for circulating around vertices in triangle
//  meshes.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/21/2019 15:51:29
////////////////////////////////////////////////////////////////////////////////
#ifndef CIRCULATOR_HH
#define CIRCULATOR_HH

#include "Handle.hh"

// Iterator-like type for circulating counter clockwise (++) or clockwise (--)
// around the tip of a halfedge "h".
template<class HEHType>
struct Circulator {
    Circulator(const HEHType    &h, int wn = 0) : m_h(    h), m_begin(        h), m_windingNumber(               wn) { }
    Circulator(const Circulator &b            ) : m_h(b.m_h), m_begin(b.m_begin), m_windingNumber(b.m_windingNumber) { }

    bool operator==(const Circulator &b) const { return (m_h == b.m_h) && (m_windingNumber == b.m_windingNumber); }
    bool operator!=(const Circulator &b) const { return !(*this == b); }

    Circulator &operator++() { m_h = m_h.ccw(); if (m_h == m_begin) ++m_windingNumber; return *this; }
    Circulator &operator--() { if (m_h == m_begin) --m_windingNumber; m_h = m_h.cw (); return *this; }
    Circulator  operator++(int) { Circulator old(*this); ++(*this); return old; }
    Circulator  operator--(int) { Circulator old(*this); --(*this); return old; }

    // Dereference operator just strips away this wrapper
    HEHType operator*() const { return m_h; }

private:
    HEHType m_h, m_begin;
    int m_windingNumber = 0; // how many times we've passed the beginning in the ccw direction
};

// A range for circulating counter clockwise around the tip of a given half-edge.
template<typename HEHType>
struct CirculatorRange {
    using Circ  = Circulator<HEHType>;

    CirculatorRange(const HEHType &h) : m_h(h) { }

    Circ begin() const { return Circ(m_h, 0); }
    Circ end()   const { return Circ(m_h, 1); }

private:
    HEHType m_h;
};

#endif /* end of include guard: CIRCULATOR_HH */
