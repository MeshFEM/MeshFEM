////////////////////////////////////////////////////////////////////////////////
// MonteCarloIntegration.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Simple Monte Carlo integration over simplices using uniformly distributed
//  random points (useful for integrating non-smooth functions when low-degree
//  Gaussian quadrature rules don't perform well).
*/
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Created:  11/05/2022 18:41:05
////////////////////////////////////////////////////////////////////////////////
#ifndef MONTECARLOINTEGRATION_H
#define MONTECARLOINTEGRATION_H

#include <MeshFEM/Simplex.hh>
#include <MeshFEM/function_traits.hh>
#include <stdexcept>

namespace detail {
    template<size_t K>
    using EigenPtMap = Eigen::Map<Eigen::Array<Real, K + 1, 1>>;

    template<size_t K>
    struct MonteCarloIntegration;

    template<>
    struct MonteCarloIntegration<1> {
        static constexpr size_t K = 1;
        static EvalPt<K> randomBarycoords() {
            EvalPt<K> x;
            EigenPtMap<K> x_eig(x.data());
            x_eig.head<K>() = 0.5 * (Eigen::Array<Real, K, 1>::Random() + 1.0);
            x_eig[K] = 1.0 - x_eig.head<K>().sum();
            return x;
        }
    };

    template<>
    struct MonteCarloIntegration<2> {
        static constexpr size_t K = 2;
        static EvalPt<K> randomBarycoords() {
            EvalPt<K> x;
            EigenPtMap<K> x_eig(x.data());
            x_eig.head<K>() = 0.5 * (Eigen::Array<Real, K, 1>::Random() + 1.0);
            if (x_eig.head<K>().sum() > 1) {
                // Reflect points in the "upper" triangle into the lower triangle
                x_eig = 1.0 - x_eig;
            }
            x_eig[K] = 1.0 - x_eig.head<K>().sum();
            return x;
        }
    };

    template<>
    struct MonteCarloIntegration<3> {
        static constexpr size_t K = 3;
        static EvalPt<K> randomBarycoords() {
            // Approach from http://vcg.isti.cnr.it/jgt/tetra.htm
            // We generate a random point in a cube tesselated into 6 reflected copies of
            // the canonical tetrahedron. Then we reflect the point into the canonical
            // tetrahedron.
            EvalPt<K> x;
            EigenPtMap<K> x_eig(x.data());
            x_eig.head<K>() = 0.5 * (Eigen::Array<Real, K, 1>::Random() + 1.0);

            if (x_eig.head<2>().sum() > 1) {
                x_eig.head<2>() = 1 - x_eig.head<2>();
            }

            if (x_eig.segment<2>(1).sum() > 1) {
                x_eig.segment<2>(1) = Eigen::Array<Real, 2, 1>(1 - x_eig[2], 1 - x_eig[0] - x_eig[1]);
            }
            else if (x_eig.head<K>().sum() > 1) {
                x_eig.head<K>() = Eigen::Array<Real, K, 1>(x_eig.head<K>().sum() - 1.0, x_eig[1], 1 - x_eig[0] - x_eig[1]);
            }

            x_eig[K] = 1.0 - x_eig.head<K>().sum();
            return x;
        }
    };
}

template<size_t K, class F>
void foreachMonteCarloSample(const F &f, size_t ns, Real vol = 1.0) {
    Real weight = vol / ns;
    for (size_t i = 0; i < ns; ++i)
        f(detail::MonteCarloIntegration<K>::randomBarycoords(), weight);
}

template<size_t K, class F>
auto monteCarloIntegration(const F &f, size_t ns, Real vol = 1.0) {
    if (ns == 0) throw std::runtime_error("At least one sample point must be used.");

    typename function_traits<F>::result_type result = f(detail::MonteCarloIntegration<K>::randomBarycoords());
    for (size_t i = 1; i < ns; ++i)
        result += f(detail::MonteCarloIntegration<K>::randomBarycoords());
    return result * (vol / ns);
}

#endif /* end of include guard: MONTECARLOINTEGRATION_H */
