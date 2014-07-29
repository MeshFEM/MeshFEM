////////////////////////////////////////////////////////////////////////////////
// Materials.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Parametrized materials that can be used with MaterialField for purposes
//      of material optimization.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/11/2014 15:48:34
////////////////////////////////////////////////////////////////////////////////
#ifndef MATERIAL_HH
#define MATERIAL_HH

#include <Flattening.hh>
#include <ElasticityTensor.hh>

// Material parameter bounds
struct Bounds {
    Bounds(size_t _var, Real _val) : var(_var), value(_val) { }
    size_t var; Real value;
};

namespace Materials {
// Var 0: Young's modulus, var 1: Poisson ratio
template<size_t _N>
struct Isotropic {
    static constexpr size_t N = _N;
    static constexpr size_t numVars = 2;
    typedef ElasticityTensor<Real, _N> ETensor;
    typedef Eigen::Matrix<Real, flatLen(_N), 1> FlattenedSymmetricMatrix;

    Isotropic() { vars[0] = 1.0; vars[1] = 0.3; }

    // Used for adjoint method gradient-based optimization
    void getETensorDerivative(size_t p, ETensor &d) const {
        assert(p == 0 || p == 1);
        d.clear();
        Real E = vars[0], nu = vars[1];
        Real dL, dmu;
        if (_N == 2) {
            // 2D Lambda = (nu * E) / (1.0 - nu * nu);
            //    mu = E / (2.0 + 2.0 * nu);
            dL = (p == 0) ? nu / (1 - nu * nu)
                          : E * (1 + nu * nu) / ((1 - nu * nu) * (1 - nu * nu));
        }
        if (_N == 3) {
            // 3D Lambda = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu));
            Real denSqrt = 1 - nu - 2 * nu * nu;
            dL = (p == 0) ? nu / ((1.0 + nu) * (1.0 - 2 * nu))
                          : E * (1 + 2 * nu * nu) / (denSqrt * denSqrt);
        }

        // 2D and 3D mu: E / (2 (1 + nu))
        dmu = (p == 0) ? 1 / (2 * (1 + nu))
                       : -E / (2 * (1 + nu) * (1 + nu));
        for (size_t i = 0; i < flatLen(_N); ++i) {
            for (size_t j = i; j < _N; ++j)
                d.D(i, j) = dL;
            d.D(i, i) += (i < _N) ? 2 * dmu : dmu;
        }
    }

    void getTensor(ETensor &tensor) const {
        tensor.setIsotropic(vars[0], vars[1]);
    }

    // Ceres-compatible cost function to fit Young's modulus, Y, and Poisson
    // ratio, nu, to closely achieve:
    //      s ~= E(Y, nu) : e
    // Where Y and nu are Young's modulus and Poisson ratio. In 2D, the
    // condition s = E(Y, nu) : e can be written in a linear form:
    //  [s_00]   [e_00,  s_11][Y ]
    //  [s_11] = [e_11,  s_00][nu]
    //  [s_01]   [e_01, -s_01]
    // We solve this in a least squares sense to get ``optimal'' Y and nu. This
    // is a slightly strange formulation in which the residual is harder to
    // interpret, but it's nice because the optimization variables are Y and nu
    // directly.
    //
    // Other options are to use Lame coefficients, which appear linearly in the
    // stress-strain relationship, or variables 1/Y and nu/Y (which also appear
    // linearly), but then E and nu are nonlinear functions of the variables
    // (and bounds/penalties/etc. on them must be transformed accordingly).
    // In 3D the corresponding condition is:
    //  [s_00]   [e_00,  s_11 + s_22]
    //  [s_11]   [e_11,  s_00 + s_22]
    //  [s_22] = [e_22,  s_00 + s_11][Y ]
    //  [s_12]   [e_12,        -s_12][nu]
    //  [s_02]   [e_02,        -s_02]
    //  [s_01]   [e_01,        -s_01]
    template<class SMatrix>
    struct stressStrainFitCostFunction {
        stressStrainFitCostFunction(const SMatrix &e, const SMatrix &s)
            : strain(e), stress(s) { }

        template<typename T>
        bool operator()(const T *x, T *e) const {
            // Nonlinear version
            if (_N == 3) {
                e[0] = T(stress[0]) - x[1] * T(stress[1] + stress[2]);
                e[1] = T(stress[1]) - x[1] * T(stress[0] + stress[2]);
                e[2] = T(stress[2]) - x[1] * T(stress[0] + stress[1]);
                e[3] = (T(1) + x[1]) * T(stress[3]);
                e[4] = (T(1) + x[1]) * T(stress[4]);
                e[5] = (T(1) + x[1]) * T(stress[5]);
            }
            else {
                e[0] = T(stress[0]) - x[1] * T(stress[1]);
                e[1] = T(stress[1]) - x[1] * T(stress[0]);
                e[2] = (T(1) + x[1]) * T(stress[2]);
            }
            for (size_t i = 0; i < flatLen(_N); ++i) {
                e[i] /= x[0];
                e[i] -= T(strain[i]);
            }

            // // Linear version
            // if (_N == 3) {
            //     e[0] = T(strain[0]) * x[0] + T(stress[1] + stress[2]) * x[1];
            //     e[1] = T(strain[1]) * x[0] + T(stress[0] + stress[2]) * x[1];
            //     e[2] = T(strain[2]) * x[0] + T(stress[0] + stress[1]) * x[1];
            //     e[3] = T(strain[3]) * x[0] - T(stress[3]) * x[1];
            //     e[4] = T(strain[4]) * x[0] - T(stress[4]) * x[1];
            //     e[5] = T(strain[5]) * x[0] - T(stress[5]) * x[1];
            // }
            // else {
            //     e[0] = T(strain[0]) * x[0] + T(stress[1]) * x[1];
            //     e[1] = T(strain[1]) * x[0] + T(stress[0]) * x[1];
            //     e[2] = T(strain[2]) * x[0] - T(stress[2]) * x[1];
            // }
            // for (size_t i = 0; i < flatLen(_N); ++i) {
            //     e[i] -= T(stress[i]);
            // }
            
            return true;
        }

        SMatrix strain, stress;
    };

    struct Bounds {
        Bounds(size_t _var, Real _val) : var(_var), value(_val) { }
        size_t var; Real value;
    };

    std::vector<Bounds> upperBounds() const { return {                  Bounds(1,  0.5) }; }
    std::vector<Bounds> lowerBounds() const { return { Bounds(0, 0.01), Bounds(1, -1.0) }; }

    Real vars[numVars];
};

// Var 0: 1/Y, var 1: nu/Y
template<size_t _N>
struct IsotropicInvParam {
    static constexpr size_t N = _N;
    static constexpr size_t numVars = 2;
    typedef ElasticityTensor<Real, _N> ETensor;
    typedef Eigen::Matrix<Real, flatLen(_N), 1> FlattenedSymmetricMatrix;

    IsotropicInvParam() { vars[0] = 1.0; vars[1] = 0.3; }

    // Used for adjoint method gradient-based optimization
    void getETensorDerivative(size_t p, ETensor &d) const { assert(false); }

    void getTensor(ETensor &tensor) const {
        tensor.setIsotropic(1.0 / vars[0], vars[1] / vars[0]);
    }

    template<class SMatrix>
    struct stressStrainFitCostFunction {
        stressStrainFitCostFunction(const SMatrix &e, const SMatrix &s)
            : strain(e), stress(s) { }

        template<typename T>
        bool operator()(const T *x, T *e) const {
            if (_N == 3) {
                e[0] = T(stress[0]) * x[0] - T(stress[1] + stress[2]) * x[1];
                e[1] = T(stress[1]) * x[0] - T(stress[0] + stress[2]) * x[1];
                e[2] = T(stress[2]) * x[0] - T(stress[0] + stress[1]) * x[1];
                e[3] = (x[0] + x[1]) * T(stress[3]);
                e[4] = (x[0] + x[1]) * T(stress[4]);
                e[5] = (x[0] + x[1]) * T(stress[5]);
            }
            else {
                e[0] = T(stress[0]) * x[0] - T(stress[1]) * x[1];
                e[1] = T(stress[1]) * x[0] - T(stress[0]) * x[1];
                e[2] = (x[0] + x[1]) * T(stress[2]);
            }
            for (size_t i = 0; i < flatLen(_N); ++i) {
                e[i] -= T(strain[i]);
            }
            
            return true;
        }

        SMatrix strain, stress;
    };

    std::vector<Bounds> upperBounds() const { return {                 }; }
    std::vector<Bounds> lowerBounds() const { return { Bounds(0, 1e-5) }; }

    Real vars[numVars];
};

}

#endif /* end of include guard: MATERIAL_HH */
