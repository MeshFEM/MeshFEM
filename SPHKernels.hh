////////////////////////////////////////////////////////////////////////////////
// SPHKernels.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements kernels suitable for SPH. These are typically radial basis
//      functions.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/18/2013 14:13:04
////////////////////////////////////////////////////////////////////////////////
#ifndef SPHKERNELS_HH
#define SPHKERNELS_HH

#include <cmath>
#include <cassert>
#include <Eigen/Dense>

template<typename Real, size_t dim>
class SPHKernel {
    typedef typename Eigen::Matrix<Real, dim, 1> Vector;
public:
    SPHKernel(const Vector &c, Real h)
        : m_c(pt), m_h(h), m_normalization(1.0) { }

    virtual Real operator()(const Vector &x) const = 0;
    virtual bool isInSupport(const Vector &x) const = 0;
    
    ////////////////////////////////////////////////////////////////////////////
    /*! Renomalize so that the integral of this kernel over the domain (formerly
    //  oldIntegral) is 1.
    //  @param[in]  oldIntegral     Integral of kernel pre-renomalization;
    //                              should be strictly positive.
    *///////////////////////////////////////////////////////////////////////////
    void renormalize(Real oldIntegral) {
        assert(oldIntegral > 1.0e-8);
        m_normalization *= 1.0 / oldIntegral;
    }

    virtual ~SPHKernel() { }

protected:
    Vector m_c;
    Real   m_h, m_normalization;
};

template<typename Real, size_t dim>
class SPHCubicSpline : public SPHKernel<Real, dim>
{
public:
    using SPHKernel::Vector;

    SPHCubicSpline(const Vector &pt, Real h)
        : SPHKernel(pt, h)
    {
        if (dim == 2) {
            // 2D cubic b-spline normalization
            this->m_normalization = 10 / (7 * M_PI * h * h);
        }
        else {
            assert(false);
        }
    }

    virtual Real operator()(const Vector &x) const {
        // q = r / h
        Real q = (x - this->m_c).norm() / h;
        assert(q >= 0.0);
        Real val = 0.0;

        if (q < 1.0) {
            Real qSq = q * q;
            val = 1.0 + (.75 * q - 1.5) * q * q;
        }
        else if (q < 2.0) {
            Real 2mx = 2 - x;
            val 0.25 * 2mx * 2mx * 2mx;
        }

        return m_normalization * val;
    }

    virtual bool isInSupport(const Vector &x) const {
        return ((x - this->m_c).norm() < 2.0 * h);
    }

};

#endif // SPHKERNELS_HH
