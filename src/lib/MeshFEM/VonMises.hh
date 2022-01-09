////////////////////////////////////////////////////////////////////////////////
// VonMises.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Computes the deviatoric extractor and von Mises stress extractor
//      In 3D, the "von Mises stress tensor" is sqrt(3/2) * stress deviator.
//      This is a tensor such that the squared Frobenius norm is the scalar
//      stress level used in the von Mises yield criterion.
//
//      In 2D (plane stress), we have a problem, since the stress deviator
//      tensor actually cannot be represented as a 2x2 matrix: in general, it
//      will have a nonzero sigma_zz component! To allow 2D code to work with
//      all 2D quantities, but still get the correct scalar von Mises stress
//      by computing a squared Frobenius norm, we must derive a special tensor.
//      For plane stress the von Mises scalar value is:
//          s_00^2 + s_11^2 - s_00 * s_11 + 3 s_01^2
//      We can simply weight the shear entry by sqrt(3/2) to ensure it
//      contributes the 3 s_01^2 term, but to find two diagonal entries to
//      contribute the rest is more difficult. Specifically, we need these
//      entries to be in the form:
//          [v_00] = [a b][s_00]
//          [v_11] = [b c][s_11]
//      so that the extractor tensor is a major symmetric rank 4 tensor.
//      These entries will contribute:
//          v_00^2 + v_11^2 = (a s_00 + b s_11)^2 + (b s_00 + c s_11)^2
//              = (a^2 + b^2) s_00 + (b^2 + c^2) s_11 + 2 (a b + b c) s_00 s_11
//      For this contribution to equal the desired von Mises scalar value terms:
//          a^2 + b^2 = 1
//          b^2 + c^2 = 1
//          a * b + b * c = -1/2
//      Variables b and c can be written in terms of a (which lies in [-1, 1])
//      by enforcing the first two equations:
//          b = +/- sqrt(1 - a^2)   (either sign choice works, we choose +)
//          c = a (if c = -a, the LHS of the third equation vanishes)
//      Now we are left with a single equation in one variable:
//          2 (a * sqrt(1 - a^2)) = -1/2    (a must be negative)
//          ==> a^2 (1 - a^2) = 1/16
//          ==> n^2 - n - 1/16 = 0          (n = a^2)
//          n = (1 +/- sqrt(1 - 1/4)) / 2
//            = (2 +/- sqrt(3)) / 4
//          ==> a = -sqrt(n) = -sqrt(2 +/- sqrt(3)) / 2
//      Whichever solution we pick for a, the other becomes b (up to sign):
//          a = -sqrt(2 + sqrt(3)) / 2, b = + sqrt(2 - sqrt(3)) / 2
//      These coefficients define a major symmetric rank 4 tensor mapping plane
//      stress tensor to a 2D tensor whose Frobenius norm equals the von Mises
//      stress scalar value, as desired.
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
#include "Fields.hh"


// WARNING: in 2D, this does not actually extract the stress deviator tensor for
// plane stress! That would require a tensor mapping a 2D stress to a 3D stress.
template<size_t N>
ElasticityTensor<Real, N> deviatoricExtractor() {
    ElasticityTensor<Real, N> result;
    result.setIdentity();

    for (size_t i = 0; i < N; ++i)
        for (size_t j = i; j < N; ++j)
            result.D(i, j) -= 1.0 / N;

    return result;
}

template<size_t N> ElasticityTensor<Real, N> vonMisesExtractor();
template<> ElasticityTensor<Real, 3> vonMisesExtractor<3>() {
    auto result = deviatoricExtractor<3>();
    result *= sqrt(3.0 / 2.0);
    return result;
}

// WARNING: this may not work as expected in 2D.
// This tensor does not extract a multiple of the stress deviator tensor as in
// the 3D case, but rather it is the unique (up to permutations and sign flips)
// tensor mapping to a 2x2 tensor whose Frobenius norm equals the von Mises
// stress scalar value.
// See derivation in file header.
template<> ElasticityTensor<Real, 2> vonMisesExtractor<2>() {
    ElasticityTensor<Real, 2> result; // zero-inits
    Real a = -sqrt(2.0 - sqrt(3)) / 2.0,
         b =  sqrt(2.0 + sqrt(3)) / 2.0;
    result.D(0, 0) = a;
    result.D(0, 1) = b;
    result.D(1, 1) = a;
    result.D(2, 2) = 0.5 * sqrt(3.0 / 2.0); // half to counter shear doubler.

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
