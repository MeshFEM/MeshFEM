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

#include <MeshFEM/Types.hh>
#include <MeshFEM/Flattening.hh>
#include <MeshFEM/ElasticityTensor.hh>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define BOOST_PARSER 1
#if BOOST_PARSER
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#endif

#include <MeshFEM_export.h>

namespace Materials {

// Material parameter bounds
struct MESHFEM_EXPORT Bounds {
    struct Bound {
        Bound(size_t _var, Real _val) : var(_var), value(_val) { }
        size_t var; Real value;
    };

    Bounds() { }
    Bounds(const std::vector<Bound> &l, const std::vector<Bound> &u)
        : m_lower(l), m_upper(u) { }

    void setFromJson(const nlohmann::json &entry);

    const std::vector<Bound> &lower() const { return m_lower; }
    const std::vector<Bound> &upper() const { return m_upper; }

    void setLower(const std::vector<Bound> &l) { m_lower = l; }
    void setUpper(const std::vector<Bound> &u) { m_upper = u; }
private:
    std::vector<Bound> m_lower, m_upper;
};

// Base class for variable materials
template<size_t _N, template<size_t> class _Mat, size_t _NVars>
struct MESHFEM_EXPORT VariableMaterial {
    typedef ElasticityTensor<Real, _N> ETensor;
    static constexpr size_t numVars = _NVars;

    virtual void getETensorDerivative(size_t p, ETensor &d) const = 0;
    virtual void getTensor(ETensor &tensor) const = 0;

    static const std::vector<Bounds::Bound> &upperBounds() { return _Mat<_N>::g_bounds.upper(); }
    static const std::vector<Bounds::Bound> &lowerBounds() { return _Mat<_N>::g_bounds.lower(); }

    static void setUpperBounds(const std::vector<Bounds::Bound> &u) { _Mat<_N>::g_bounds.setUpper(u); }
    static void setLowerBounds(const std::vector<Bounds::Bound> &l) { _Mat<_N>::g_bounds.setLower(l); }

    static void setBoundsFromFile(const std::string &path) {
        std::ifstream is(path);
        if (!is.is_open()) {
            throw std::runtime_error("Couldn't open bounds " + path);
        }
        nlohmann::json config;
        is >> config;
        setBoundsFromJson(config);
    }

    static void setBoundsFromJson(const nlohmann::json &config) {
        _Mat<_N>::g_bounds.setFromJson(config);
        // Validate bounds.
        std::runtime_error indexError("Bounds variable index out-of-bounds.");
        for (const auto &b : _Mat<_N>::g_bounds.lower()) if (b.var >= numVars) throw indexError;
        for (const auto &b : _Mat<_N>::g_bounds.upper()) if (b.var >= numVars) throw indexError;
    }

    virtual ~VariableMaterial() { }

    Real vars[numVars];
};

// Var 0: Young's modulus, var 1: Poisson ratio
template<size_t _N>
struct MESHFEM_EXPORT Isotropic : public VariableMaterial<_N, Isotropic, 2> {
    static constexpr size_t N = _N;
    typedef ElasticityTensor<Real, _N> ETensor;
    typedef Eigen::Matrix<Real, flatLen(_N), 1> FlattenedSymmetricMatrix;
    typedef VariableMaterial<N, Materials::Isotropic, 2> Base;
    using Base::vars;

    // WARNING: bounds are shared by all isotropic materials! (static)
    struct IsotropicBounds : Bounds {
        IsotropicBounds() {
            // Default Bounds
            // Upper: Upper bounds should be based on base material's moduli.
            //        Poisson ratio can't be greater than or equal 0.5
            //        (at 0.5, 3D lambda becomes Inf)
            // Lower: Young's modulus must be positive and is hard to make
            //        small--this minimum should be set based on homogenization results
            //        Poisson ratio can't be less than -1, and for robustness we
            //        limit it to -0.75
            Bounds::setUpper({ Bounds::Bound(0, 292), Bounds::Bound(1, 0.6) });
            Bounds::setLower({ Bounds::Bound(0, 25),  Bounds::Bound(1, 0.1) });
        }
    };

    Isotropic() {
        // Default Parameters: midway between bounds (if they exist)
        const auto &lb = this->lowerBounds();
        const auto &ub = this->upperBounds();
        if ((lb.size() == 2) && (ub.size() == 2)) {
            for (auto &bd : lb) vars[bd.var]  = 0.5 * bd.value;
            for (auto &bd : ub) vars[bd.var] += 0.5 * bd.value;
        }
        else {
            vars[0] = 50.0;
            vars[1] = 0.3;
        }
    }

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
        StressStrainFitCostFunction(const SMatrix &e, const SMatrix &s, Real vol)
            : strain(e), stress(s) {
            if (vol <= 0) throw std::runtime_error("Volume must be positive");
            volSqrt = sqrt(vol);
        }

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
                if (i >= _N) e[i] *= T(sqrt(2.0));
                e[i] *= T(volSqrt);
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
        Real volSqrt;
    };
private:
    friend Base;
    static IsotropicBounds g_bounds;
};

// Static variable needs to be explicitly defined...
template<size_t _N>
typename Isotropic<_N>::IsotropicBounds Isotropic<_N>::g_bounds;

// Axis-aligned orthotropic material.
// 2D: 4 variables
// Vars 0..1: Young's moduli,
// Var     2: Poisson ratio (YX)
// Var     3: Shear modulus
// 3D: 9 variables
// Vars 0..2: Young's moduli,
// Vars 3..5: Poisson ratios (YX, ZX, ZY)
// Vars 6..8: Shear ratios   (YZ, ZX, XY)
size_t constexpr nOrthotropicVars(size_t n) { return (n == 3) ? 9 : 4; }
template<size_t _N>
struct MESHFEM_EXPORT Orthotropic : public VariableMaterial<_N, Orthotropic, nOrthotropicVars(_N)> {
    static constexpr size_t N = _N;
    typedef ElasticityTensor<Real, _N> ETensor;
    typedef Eigen::Matrix<Real, flatLen(_N), 1> FlattenedSymmetricMatrix;

    typedef VariableMaterial<N, Materials::Orthotropic, nOrthotropicVars(N)> Base;
    using Base::vars;

    // WARNING: bounds are shared by all orthotropic materials! (static)
    struct OrthotropicBounds : Bounds {
        OrthotropicBounds() {
            // Default Bounds
            // Upper: Upper bounds should be based on base material's moduli.
            //        Poisson ratios can't be greater than 0.5
            //        (at 0.5, 3D isotropic lambda becomes Inf, so we avoid it
            //        here too)
            // Lower: Young's and shear moduli must be positive and are hard to make
            //        small--this minimum should be set based on homogenization results
            //        Poisson ratios can't be less than -1, and for robustness we
            //        limit them to -0.75
            if (_N == 3) Base::setUpperBounds({ Bound(3, 0.45), Bound(4, 0.45), Bound(5, 0.45) });
            else         Base::setUpperBounds({ Bound(0,  384), Bound(1,  384), Bound(2, 0.45), Bound(3, 102) });
            if (_N == 3) {
                Base::setLowerBounds({ Bound(0,  0.01), Bound(1,  0.01), Bound(2,  0.01),
                                       Bound(3, -0.75), Bound(4, -0.75), Bound(5, -0.75),
                                       Bound(6,  0.01), Bound(7,  0.01), Bound(8,  0.01) });
            }
            else Base::setLowerBounds({ Bound(0,  18), Bound(1,  18),
                                        Bound(2, 0.0), Bound(3,  2) });
        }
    };

    Orthotropic() {
        // Default Parameters
        if (_N == 3) {
            vars[0] = vars[1] = vars[2] = 1.0;
            vars[3] = vars[4] = vars[5] = 0.3;
            vars[6] = vars[7] = vars[8] = 1 / (2.0 * (1 + 0.3));
        }
        else {
            vars[0] = vars[1] = 200.0;
            vars[2] = 0.3;
            vars[3] = 200.0 / (2.0 * (1 + 0.3));
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
        StressStrainFitCostFunction(const SMatrix &e, const SMatrix &s, Real vol)
            : strain(e), stress(s) {
            if (vol <= 0) throw std::runtime_error("Volume must be positive");
            volSqrt = sqrt(vol);
        }

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
                if (i >= _N) e[i] *= T(sqrt(2.0));
                e[i] *= T(volSqrt);
            }

            return true;
        }

        Real volSqrt;
        SMatrix strain, stress;
    };
private:
    friend Base;
    static OrthotropicBounds g_bounds;
};

// Static variable needs to be explicitly defined...
template<size_t _N>
typename Orthotropic<_N>::OrthotropicBounds Orthotropic<_N>::g_bounds;

template<size_t _N>
struct MESHFEM_EXPORT Constant {
    static constexpr size_t N = _N;
    static constexpr size_t numVars = 0;
    typedef ElasticityTensor<Real, _N> ETensor;

    Constant() { m_E.setIsotropic(1.0, 0.3); }
    Constant(const std::string &materialFile) { setFromFile(materialFile); }


    void setFromFile(const std::string &materialFile);
    void setFromJson(const nlohmann::json &config);
#if BOOST_PARSER
    void setFromPTree(const boost::property_tree::ptree &pt);
#endif

    // Used for adjoint method gradient-based optimization
    void getETensorDerivative(size_t /* p */, ETensor &/* d */) const {
        throw std::runtime_error("Constant material can't be optimized\n");
    }

    const ETensor &getTensor()      const { return m_E; }
    void getTensor(ETensor &tensor) const { tensor = m_E; }
    void setTensor(const ETensor &tensor) { m_E = tensor; }

    void setIsotropic(Real E, Real nu) { m_E.setIsotropic(E, nu); }

    nlohmann::json getJson() const;

    // "type": "anisotropic",
    // "dim": 3,
    // "material_matrix": [[C_00, C_01, C02, C03, C04, C05],
    //                     [C_10, C_11, C12, C13, C14, C15],
    //                     [C_20, C_21, C22, C23, C24, C25],
    //                     [C_30, C_31, C32, C33, C34, C35],
    //                     [C_40, C_41, C42, C43, C44, C45],
    //                     [C_50, C_51, C52, C53, C54, C55]]
    friend std::ostream &operator<<(std::ostream &os, const Constant &cmat) {
        os << cmat.getJson().dump(4);
        return os;
        // os << "{ \"type\": \"anisotropic\"," << std::endl;
        // os << "\"material_matrix\": [";
        // for (size_t i = 0; i < flatLen(N); ++i) {
        //     for (size_t j = 0; j < flatLen(N); ++j) {
        //         os << (j ? ", " : "[") << cmat.m_E.D(i, j);
        //     }
        //     os << (i == flatLen(N) - 1 ? "]]" : "],") << std::endl;
        // }
        // os << "}";
        // return os;
    }

private:
    ETensor m_E;
};

} // Materials

#endif /* end of include guard: MATERIAL_HH */
