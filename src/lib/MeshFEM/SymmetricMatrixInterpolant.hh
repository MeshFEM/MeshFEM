////////////////////////////////////////////////////////////////////////////////
// SymmetricMatrixInterpolant.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides a symmetric matrix-valued interpolant.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/20/2014 16:52:35
////////////////////////////////////////////////////////////////////////////////
#ifndef SYMMETRICMATRIXINTERPOLANT_HH
#define SYMMETRICMATRIXINTERPOLANT_HH

#include <MeshFEM/SymmetricMatrix.hh>
#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/Functions.hh>

template<class SMat, size_t _K, size_t _Deg>
class SymmetricMatrixInterpolant : public Interpolant<SMat, _K, _Deg>
{
    typedef Interpolant<SMat, _K, _Deg> Base;
public:
    using Base::Base; using Base::numNodalValues;

    // Zero out this symmetric matrix function
    void clear() {
        for (size_t inode = 0; inode < numNodalValues; ++inode)
            (*this)[inode].clear();
    }
    SymmetricMatrixInterpolant doubleContract(const ElasticityTensor<Real, SMat::N> &E) const {
        SymmetricMatrixInterpolant result;
        for (size_t inode = 0; inode < numNodalValues; ++inode)
            result[inode] = E.doubleContract((*this)[inode]);
        return result;
    }
};

#endif /* end of include guard: SYMMETRICMATRIXINTERPOLANT_HH */
