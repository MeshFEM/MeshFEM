////////////////////////////////////////////////////////////////////////////////
// Quadrature.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides quadrature sample locations and weights.
//      Only supports double precision currently because of partial template
//      specialization woes (only classes in c++ can be partially specialized,
//      tno their memebers :().
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/02/2013 15:45:35
////////////////////////////////////////////////////////////////////////////////
#ifndef QUADRATURE_HH
#define QUADRATURE_HH
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>

#include "Geometry.hh"
#include "GlobalTypes.hh"

#include <Eigen/Dense>

template<int _Dim>
class Quadrature {
    typedef Scalar                       Real;
    typedef Eigen::Matrix<Real, _Dim, 1> _Vector;
public:
    Quadrature(int numPoints = -1, QuadratureMethod method =
            UNIFORM_QUADRATURE);

    void setNumPoints(int numPoints);
    size_t numPoints() const { return m_referenceQuadraturePoints.size(); }

    void setQuadratureMethod(QuadratureMethod method) { m_method = method; }
    QuadratureMethod getQuadratureMethod() const { return m_method; }

    void setUsingGaussQuadrature(bool b) {
        setQuadratureMethod(b ? GAUSS_QUADRATURE : UNIFORM_QUADRATURE);
    }

    void quadraturePoints(const BBox<_Vector> &b,
                          std::vector<_Vector> &qp) const;
    std::vector<_Vector> quadraturePoints(const BBox<_Vector> &b) const;

    // For each quadrature point, calls f.accumulate() passing in the sample
    // point along with the corresponding reference point (in the canonical
    // element) and weight (scaled by the ref->element jacobian determinant)
    template<typename Func>
    void integrate(Func &f, const BBox<_Vector> &b) const {
        int n = numPoints();
        Real volume = b.volume();
        for (int i = 0; i < n; ++i) {
            const _Vector &p = m_referenceQuadraturePoints[i];
            Real weight = m_referenceQuadratureWeights[i];
            _Vector sample = b.interpolatePoint(p);
            f.accumulate(sample, p, weight * volume);
        }
    }

private:
    QuadratureMethod m_method;
    std::vector<_Vector> m_referenceQuadraturePoints;
    std::vector<Real>    m_referenceQuadratureWeights;

    void m_generateReferenceQuadratureNodes(int numPoints);
};

typedef Quadrature<2> Quadrature2D;
typedef Quadrature<3> Quadrature3D;

#endif // QUADRATURE_HH
