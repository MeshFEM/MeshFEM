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
    typedef BBox<_Vector>           _BBox;
    typedef _Vector                 Vector;
    typedef typename Vector::Scalar Real;

    LevelSet() { } // Only should be used by subclass
    LevelSet(const _BBox &domain)
        : m_domain(domain) { }

    virtual bool isInside(const Vector &p) const {
        return signedDistance(p) < 0;
    }

    virtual Real signedDistance(const Vector &p) const = 0;

    virtual void setDomain(const _BBox &b) { m_domain = b; }
    virtual _BBox      domain() const { return m_domain; }
    virtual _BBox boundingBox() const { return domain(); }

    virtual ~LevelSet() { }
protected:
    _BBox m_domain;
};

template<typename _Vector>
class Sphere : public LevelSet<_Vector>
{
public:
    typedef LevelSet<_Vector> super;
    using typename super::_BBox;
    using typename super::Vector;
    using typename super::Real;

    Sphere(const _BBox &domain, const Vector &center, Real radius)
        : super(domain), m_center(center), m_radius(radius) { }
    Real signedDistance(const Vector &p) const {
        return (p - m_center).norm() - m_radius;
    }
private:
    Vector m_center;
    Real m_radius;
};

template<typename _Vector>
class SchwarzP : public LevelSet<_Vector>
{
public:
    typedef LevelSet<_Vector> super;
    using typename super::_BBox;
    using typename super::Vector;
    using typename super::Real;

    SchwarzP(const _BBox &domain) : super(domain)  { }

    Real signedDistance(const Vector &p) const {
        return cos(p[0]) + cos(p[1]) + cos(p[2]);
    }
};

#endif // LEVEL_SET_HH
