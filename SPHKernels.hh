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
public:
    typedef typename Eigen::Matrix<Real, dim, 1> Vector;
    SPHKernel(const Vector &c, Real h)
        : m_c(c), m_h(h), m_normalization(1.0) { }

    virtual Real operator()(const Vector &x) const = 0;
    virtual bool isInSupport(const Vector &x) const = 0;

    const Vector &center() const { return m_c; }
    
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

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the scale factor that makes kernel's maximum value 1
    //  @return     scaling factor
    *///////////////////////////////////////////////////////////////////////////
    Real maxNormalizationFactor() const {
        return 1.0 / (*this)(m_c);
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
    using typename SPHKernel<Real, dim>::Vector;

    SPHCubicSpline(const Vector &c, Real h)
        : SPHKernel<Real, dim>(c, h)
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
        Real q = (x - this->m_c).norm() / this->m_h;
        assert(q >= 0.0);
        Real val = 0.0;

        if (q < 1.0) {
            val = 1.0 + (.75 * q - 1.5) * q * q;
        }
        else if (q < 2.0) {
            Real twomq = 2 - q;
            val = 0.25 * twomq * twomq * twomq;
        }

        return this->m_normalization * val;
    }

    virtual bool isInSupport(const Vector &x) const {
        return ((x - this->m_c).norm() < 2.0 * this->m_h);
    }

};

#endif // SPHKERNELS_HH
