////////////////////////////////////////////////////////////////////////////////
// LevelSet.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Provides a generic level set interface that can act as a "Model" for
//        MeshlessFEM[23]D as well as some simple examples.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  04/30/2014 17:33:35
////////////////////////////////////////////////////////////////////////////////
#ifndef LEVEL_SET_HH
#define LEVEL_SET_HH
#include <cmath>

template<typename _Vector>
class LevelSet {
public:
    typedef BBox<_Vector>           BBox;
    typedef _Vector                 Vector;
    typedef typename Vector::Scalar Real;

    LevelSet(const BBox &domain)
        : m_domain(domain) { }

    virtual bool isInside(const Vector &p) const = 0;
    Real value(const Vector &p) const = 0;

    void setDomain(const BBox &b) { m_domain = b; }
    BBox    domain() const        { return m_domain; }

    ~LevelSet() { }
private:
    BBox m_domain;
};

template<typename _Vector>
class Sphere : public LevelSet<_Vector>
{
public:
    using typename LevelSet<_Vector>::BBox;
    using typename LevelSet<_Vector>::Vector;
    using typename LevelSet<_Vector>::Real;

    Sphere(const BBox &domain, const Vector &center, Real radius)
        : LevelSet(domain), m_center(center), m_radius(radius) { }
    bool isInside(const Vector &p) const {
        return (p - m_center).norm() <= m_radius;
    }
private:
    Vector m_center;
    Real m_radius;
};

#endif // LEVEL_SET_HH
