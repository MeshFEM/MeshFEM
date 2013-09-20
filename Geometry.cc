#include <cmath>
// Switched to boost for elliptic integrals since they didn't make it into C++11
// from tr1 :(
#include <boost/math/special_functions/ellint_2.hpp>

#include <algorithm>
#include "Geometry.hh"
#include <iostream>
#include <iomanip>

size_t NUM_NEWTON_ITERATIONS = 5;

////////////////////////////////////////////////////////////////////////////////
/*! Returns parameter values 't' generating N evenly (arc-length) spaced points
//  around the ellipse:
//      t |--> (a * sin(t), b * cos(t))
//  @param[in]  s   target spacing of points to distribute (this is never
//                  exceeded)
//  @param[in]  a   ellipse major axis
//  @param[in]  b   ellipse minor axis
//  @param[out] paramPoints     Vector of parameter values.
//  @param[out] pointAreas      length of the arc segment centered on each point
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
void ellipseParameterPoints(Real s, Real a, Real b,
                            std::vector<Real> &paramPoints,
                            Real &pointAreas)
{
    // Make sure a is the major radius, so (b / a) < 1
    if (a < b)
        std::swap(a, b);

    // full arc length:
    // l    = 4 * int_0^Pi/2 sqrt(a^2 cos^2(t) + b^2 sin^2(t))
    //      = 4 * int_0^Pi/2 sqrt(a^2 (1 - sin^2(t)) + b^2 sin^2(t))
    //      = 4 * int_0^Pi/2 sqrt(a^2 + (b^2 - a^2) sin^2(t))
    //      = 4 * a * int_0^Pi/2 sqrt(1 - (1 - b^2/a^2) sin^2(t))
    // ellint_2 calls compute:
    //      int sqrt(1 - k^2 sin^2 x) dx
    //  ==> k = sqrt(1 - b^2 / a^2), which is always real
    Real aSq = a * a;
    Real bSq = b * b;
    Real kSq = 1 - bSq / aSq;

    Real k = sqrt(kSq);
    Real perimeter = 4 * a * boost::math::ellint_2(k);
    int N = ceil(perimeter / s);
    Real segmentArcLen = perimeter / N;
    pointAreas = segmentArcLen;

    paramPoints.clear();
    paramPoints.reserve(N);
    // Parameter value:
    // From above derivation, we have:
    //      s(t) = a * int_0^t sqrt(1 - (1 - b^2/a^2) sin^2(t))
    // By the Fundamental Theorem of Calculus:
    //      s'(t) = a * sqrt(1 - (1 - b^2/a^2) sin^2(t))
    // So we can run Newton's method to solve for t corresponding to a given s.
    // Assuming reasonably large N, we start out close to the correct answer so
    // only a few steps need to be run.
    Real t = 0;
    paramPoints.push_back(t);

    for (int i = 1; i < N; ++i) {
        Real s_target = i * segmentArcLen;

        for (size_t j = 0; j < NUM_NEWTON_ITERATIONS; ++j) {
            Real s_t = a * boost::math::ellint_2(k, t);
            Real sin_t = sin(t);
            Real s_prime = a * sqrt(1 - kSq * sin_t * sin_t);
            t += (s_target - s_t) / s_prime;
        }

        paramPoints.push_back(t);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Template instantiations
////////////////////////////////////////////////////////////////////////////////
template
void ellipseParameterPoints(float s, float a, float b,
                            std::vector<float> &paramPoints, float &ptAreas);
template
void ellipseParameterPoints(double s, double a, double b,
                            std::vector<double> &paramPoints, double &ptAreas);
