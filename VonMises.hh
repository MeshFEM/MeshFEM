////////////////////////////////////////////////////////////////////////////////
// VonMises.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Computes the deviatoric extractor and von Mises stress extractor
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/11/2016 23:10:44
////////////////////////////////////////////////////////////////////////////////
#ifndef VONMISES_HH
#define VONMISES_HH

#include <stdexcept>

#include "ElasticityTensor.hh"
#include "SymmetricMatrix.hh"

template<size_t N>
ElasticityTensor<Real, N> deviatoricExtractor() {
    ElasticityTensor<Real, N> result;
    result.setIdentity();

    // Subtact off deviatoric part
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i; j < N; ++j) {
            result.D(i, j) -= 1.0 / N;
        }
    }

    return result;
}

template<size_t N>
ElasticityTensor<Real, N> vonMisesExtractor() {
    auto result = deviatoricExtractor<N>();
    result *= sqrt(3.0 / 2.0);
    return result;
}

template<size_t N>
SymmetricMatrixField<Real, N> vonMises(const SymmetricMatrixField<Real, N> &smf) {
    SymmetricMatrixField<Real, N> result(smf.domainSize());
    auto V = vonMisesExtractor<N>();
    for (size_t i = 0; i < smf.domainSize(); ++i)
        result(i) = V.doubleContract(smf(i));
    return result;
}

template<class _SymMat>
typename std::enable_if<is_symmetric_matrix<_SymMat>::value, _SymMat>::type
vonMises(const _SymMat &sm) {
    constexpr size_t N = _SymMat::N;
    return vonMisesExtractor<N>().doubleContract(sm);
}

// DynamicSymmetricMatrix must be handled separately (needed for msh_processor)
DynamicSymmetricMatrix<Real> vonMises(const DynamicSymmetricMatrix<Real> &dsm) {
    // Note: double contraction doesn't yet work for Dynamic-sized matrices--it
    // treats DynamicSymmetricMatrix as 3x3.
    if (dsm.size() == 2) {
        // Do a wasteful conversion for the 2D case...
        SymmetricMatrixValue<Real, 2> sm;
        sm[0] = dsm[0];
        sm[1] = dsm[1];
        sm[2] = dsm[2];
        return vonMisesExtractor<2>().doubleContract(sm);
    }
    if (dsm.size() == 3)
        return vonMisesExtractor<3>().doubleContract(dsm);
    throw std::runtime_error("Invalid matrix dimension");
}

#endif /* end of include guard: VONMISES_HH */
