////////////////////////////////////////////////////////////////////////////////
// Materials.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Parametrized materials that can be used with MaterialField for purposes
//      of material optimization. Each material provides getETensorDerivative,
//      which gives the derivative of the elasticity tensor with respect to one
//      material parameter.
//
//      The exception is ConstantMaterial, which is intended to be read from a
//      file and which doesn't support material optimization.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/11/2014 15:48:34
////////////////////////////////////////////////////////////////////////////////
#ifndef MATERIAL_HH
#define MATERIAL_HH

#include "Types.hh"
#include <Flattening.hh>
#include <ElasticityTensor.hh>
#include <stdexcept>
#include <vector>
#include <string>

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

    static const std::string &variableName(size_t i) {
        static const std::vector<std::string> names = { "E", "nu" };
        return names.at(i);
    }

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
            dL = (p == 0) ? nu / ((1.0 + nu) * (1.0 - 2 * nu)) : E * (1 + 2 * nu * nu) / (denSqrt * denSqrt);
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
    // ratio, nu, to best achieve:
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
    struct StressStrainFitCostFunction {
        StressStrainFitCostFunction(const SMatrix &e, const SMatrix &s)
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

    // Upper bounds: Upper bounds should be based on base material's moduli.
    //               Poisson ratio can't be greater than or equal 0.5
    //               (at 0.5, 3D lambda becomes Inf)
    // Lower bounds: Young's modulus must be positive and is hard to make
    //               small--this minimum should be set based on homogenization results
    //               Poisson ratio can't be less than -1, and for robustness we
    //               limit it to -0.75
    constexpr std::vector<Bounds> upperBounds() const { return {                  Bounds(1,  0.45) }; }
    constexpr std::vector<Bounds> lowerBounds() const { return { Bounds(0, 0.01), Bounds(1, -0.75) }; }

    Real vars[numVars];
};

// Axis-aligned orthotropic material.
// 2D: 4 variables
// Vars 0..1: Young's moduli,
// Var     2: Poisson ratio (YX)
// Var     3: Shear modulus
// 3D: 9 variables
// Vars 0..2: Young's moduli,
// Vars 3..5: Poisson ratios (YX, ZX, ZY)
// Vars 6..8: Shear ratios   (YZ, ZX, XY)
template<size_t _N>
struct Orthotropic {
    static constexpr size_t N = _N;
    static constexpr size_t nvarsForDim(size_t n) { return (_N == 3) ? 9 : 4; }
    static constexpr size_t numVars = nvarsForDim(_N);
    typedef ElasticityTensor<Real, _N> ETensor;
    typedef Eigen::Matrix<Real, flatLen(_N), 1> FlattenedSymmetricMatrix;

    Orthotropic() {
        if (_N == 3) {
            vars[0] = vars[1] = vars[2] = 1.0;
            vars[3] = vars[4] = vars[5] = 0.3;
            vars[6] = vars[7] = vars[8] = 1 / (2.0 * (1 + 0.3));
        }
        else {
            vars[0] = vars[1] = 1.0;
            vars[2] = 0.3;
            vars[3] = 1 / (2.0 * (1 + 0.3));
        }
    }

    static const std::string &variableName(size_t i) {
        if (_N == 3) {
            static const std::vector<std::string> names3D = {
                "E_x", "E_y", "E_z",
                "nu_yx", "nu_zx", "nu_zy",
                "mu_yz", "mu_zx", "mu_xy" };
            return names3D.at(i);
        }
        else {
            static const std::vector<std::string> names2D = {
                "E_x", "E_y", "nu_yx", "mu" };
            return names2D.at(i);
        }
    }

    // Used for adjoint method gradient-based optimization
    void getETensorDerivative(size_t p, ETensor &d) const;

    void getTensor(ETensor &tensor) const {
        if (_N == 3) {
            tensor.setOrthotropic3D(vars[0], vars[1], vars[2],
                                    vars[3], vars[4], vars[5],
                                    vars[6], vars[7], vars[8]);
        }
        else {
            tensor.setOrthotropic2D(vars[0], vars[1], vars[2], vars[3]);
        }
    }

    // Ceres-compatible cost function to fit orthotropic material parameters to
    // best achieve:
    //      e ~= E^(-1)(Y_x, Y_y, ...) : s
    template<class SMatrix>
    struct StressStrainFitCostFunction {
        StressStrainFitCostFunction(const SMatrix &e, const SMatrix &s)
            : strain(e), stress(s) { }

        template<typename T>
        bool operator()(const T *x, T *e) const {
            if (_N == 3) {
                T D01 =  -x[3] / x[1], // -nu_yx / E_y
                  D02 =  -x[4] / x[2], // -nu_zx / E_z
                  D12 =  -x[5] / x[2]; // -nu_zy / E_z
                e[0] = T(stress[0]) / x[0] + T(stress[1]) *  D01 + T(stress[2]) *  D02;
                e[1] = T(stress[0]) *  D01 + T(stress[1]) / x[1] + T(stress[2]) *  D12;
                e[2] = T(stress[0]) *  D02 + T(stress[1]) *  D12 + T(stress[2]) / x[2];
                e[3] = T(0.5 * stress[3]) / x[6];
                e[4] = T(0.5 * stress[4]) / x[7];
                e[5] = T(0.5 * stress[5]) / x[8];
            }
            else {
                T D01 = -x[2] / x[1]; // -nu_yx / E_y
                e[0] = T(stress[0]) / x[0] + T(stress[1]) *  D01;
                e[1] = T(stress[0]) *  D01 + T(stress[1]) / x[1];
                e[2] = T(0.5 * stress[2]) / x[3];
            }

            for (size_t i = 0; i < flatLen(_N); ++i) {
                e[i] -= T(strain[i]);
            }

            return true;
        }

        SMatrix strain, stress;
    };

    struct Bounds {
        Bounds(size_t _var, Real _val) : var(_var), value(_val) { }
        size_t var; Real value;
    };

    // Upper bounds: Upper bounds should be based on base material's moduli.
    //               Poisson ratios can't be greater than 0.5
    //               (at 0.5, 3D isotropic lambda becomes Inf, so we avoid it
    //               here too)
    // Lower bounds: Young's and sheer moduli must be positive and are hard to make
    //               small--this minimum should be set based on homogenization results
    //               Poisson ratios can't be less than -1, and for robustness we
    //               limit them to -0.75
    std::vector<Bounds> upperBounds() const {
        if (_N == 3) return { Bounds(3,  0.45), Bounds(4, 0.45), Bounds(5, 0.45) };
        else         return { Bounds(2,  0.45) };
    }
    std::vector<Bounds> lowerBounds() const {
        if (_N == 3)
             return { Bounds(0,  0.01), Bounds(1,  0.01), Bounds(2,  0.01),
                      Bounds(3, -0.75), Bounds(4, -0.75), Bounds(5, -0.75),
                      Bounds(6,  0.01), Bounds(7,  0.01), Bounds(8,  0.01) };
        else return { Bounds(0,  0.01), Bounds(1,  0.01),
                      Bounds(2, -0.75), Bounds(3,  0.01) };
    }

    Real vars[numVars];
};

template<size_t _N>
struct Constant {
    static constexpr size_t N = _N;
    static constexpr size_t numVars = 0;
    typedef ElasticityTensor<Real, _N> ETensor;

    Constant() { m_E.setIsotropic(1.0, 0.3); }
    Constant(const std::string &materialFile) { setFromFile(materialFile); }

    void setFromFile(const std::string &materialFile);

    // Used for adjoint method gradient-based optimization
    void getETensorDerivative(size_t p, ETensor &d) const {
        throw std::runtime_error("Constant material can't be optimized\n");
    }

    const ETensor &getTensor()      const { return m_E; }
    void getTensor(ETensor &tensor) const { tensor = m_E; }

private:
    ETensor m_E;
};

}

#endif /* end of include guard: MATERIAL_HH */
