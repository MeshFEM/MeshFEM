////////////////////////////////////////////////////////////////////////////////
// TensorProjection.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Finds the closest elasticity/compliance tensors with various symmetries
//      using various metrics (eventually)...
//      For now, we just find the closest isotropic elasticity tensor in the
//      Frobenius/Euclidean metric.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/19/2016 16:30:10
////////////////////////////////////////////////////////////////////////////////
#ifndef TENSORPROJECTION_HH
#define TENSORPROJECTION_HH

#include "ElasticityTensor.hh"

// Closest isotropic elasticity tensor in the Frobenius/Euclidean metric:
// argmin_(E, nu) ||C_given - C(E, nu)||_F^2
template<typename Real, size_t N>
ElasticityTensor<Real, N> closestIsotropicTensor(const ElasticityTensor<Real, N> &C) {
    // Deriving the closest isotropic tensor is simplified by choosing a nice
    // basis: we choose the "Hydrostatic Extractor" and "Deviatoric Extractor"
    //      J = 1/n delta_ij delta_kl,       K = I - J,
    // where I is the rank 4 symmetric identity.
    // Now any isotropic tensor can be written in this basis:
    //      Ciso = lambda delta_ij delta_kl + 2 mu I
    //           =  n * lambda J + 2 mu (J + K)
    //           = (n * lambda + 2 mu) J + 2 mu K
    //           = n * kappa * J + 2 mu K,
    //          := alpha * J + beta * K
    // where kappa is the bulk modulus and mu is the shear modulus.
    // This particular basis is nice because one can show it's orthogonal.
    // <J, K> = 0
    // Thus coefficients [alpha, beta] of the closest isotropic tensor to C can
    // be found directly by taking inner products (here the inner product is
    // quadruple contraction):
    // [<J, J>  <J, K>][alpha] = [<C, J>] = [<J, J>       0][alpha]
    // [<J, K>  <K, K>][beta ]   [<C, K>]   [     0  <K, K>][beta ]
    // ==> alpha = <C, J> / <J, J>
    //     beta  = <C, K> / <K, K>
    // It's also straightforward to compute:
    // <C, I> = C_ijij
    // <C, J> = 1/n C_iijj
    // <C, K> = <C, I> - <C, J>
    // <J, J> = 1
    // <K, K> = <I, K> = <I, I> - <I, J> = 1/2 (n^2 + n) - 1
    Real C_ijij = 0.0, C_iijj = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C_ijij += C(i, j, i, j);
            C_iijj += C(i, i, j, j);
        }
    }
    Real n = N;

    Real CdotI = C_ijij;
    Real CdotJ = (C_iijj / n);
    Real CdotK = CdotI - CdotJ;

    Real JdotJ = 1.0;
    Real KdotK = 0.5 * (n * n + n) - 1.0;

    Real alpha = CdotJ / JdotJ;
    Real  beta = CdotK / KdotK;

    // n * lambda + 2 mu = alpha
    // beta = 2mu
    Real lambda = (alpha - beta) / n;
    Real mu = beta / 2.0;

    ElasticityTensor<Real, N> result;
    result.setIsotropicLame(lambda, mu);
    return result;
}

#endif /* end of include guard: TENSORPROJECTION_HH */
