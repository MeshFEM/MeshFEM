#include "Materials.hh"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <map>

#if BOOST_PARSER
using boost::property_tree::ptree;
#endif

namespace Materials {

// Derivatives of the elasticity tensor with respect to the material properties:
// These are derived in Derivations/OrthotropicTensorDerivatives.nb
// 3D:
// Vars 0..2: Young's moduli,
// Vars 3..5: Poisson ratios (YX, ZX, ZY)
// Vars 6..8: Shear ratios   (YZ, ZX, XY)
// 2D:
// Vars 0..1: Young's moduli,
// Var     2: Poisson ratio (YX)
// Var     3: Shear modulus
template<size_t _N>
void Orthotropic<_N>::getETensorDerivative(size_t p, Orthotropic<_N>::ETensor &d) const {
    d.clear();
    if (_N == 3) {
        Real Ex = vars[0], Ey = vars[1], Ez = vars[2];
        Real vyx = vars[3], vzx = vars[4], vzy = vars[5];
        if (p == 0) {
            d.D(0, 0) = pow(Ey,2)*pow(Ez - Ey*pow(vzy,2),2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = -((Ez*vyx + Ey*vzx*vzy)*pow(Ey,2)*(-Ez + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(0, 2) = -(Ez*(vzx + vyx*vzy)*pow(Ey,2)*(-Ez + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 1) = pow(Ey,2)*pow(Ez*vyx + Ey*vzx*vzy,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = Ez*(vzx + vyx*vzy)*(Ez*vyx + Ey*vzx*vzy)*pow(Ey,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(2, 2) = pow(Ey,2)*pow(Ez,2)*pow(vzx + vyx*vzy,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 1) {
            d.D(0, 0) = pow(Ex,2)*(-(pow(Ez,2)*pow(vyx,2)) + vzx*(vzx + 2*vyx*vzy)*pow(Ey,2)*pow(vzy,2) + 2*Ey*Ez*pow(vyx,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = -(Ex*(-(Ez*vzy*(vzx + vyx*vzy)*pow(Ey,2)) + Ex*(2*Ey*Ez*vzx*vzy*pow(vyx,2) + pow(Ez,2)*pow(vyx,3) + vzy*(vzx + 2*vyx*vzy)*pow(Ey,2)*pow(vzx,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(0, 2) = -(Ex*Ez*(vzx + vyx*vzy)*(Ex*Ez*pow(vyx,2) - pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 1) = Ey*(Ey*(Ez - Ex*vzx*(vzx + 2*vyx*vzy)) - 2*Ex*Ez*pow(vyx,2))*(Ez - Ex*pow(vzx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = -(Ez*(-(Ez*vzy*pow(Ey,2)) + Ex*Ey*vzy*(Ey*vzx*(vzx + vyx*vzy) + 2*Ez*pow(vyx,2)) + Ez*vzx*pow(Ex,2)*pow(vyx,3))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(2, 2) = -(pow(Ez,2)*(vzx*(vzx + 2*vyx*vzy)*pow(Ex,2)*pow(vyx,2) - pow(Ey,2)*pow(vzy,2) + 2*Ex*Ey*pow(vyx,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
        };
        if (p == 2) {
            d.D(0, 0) = -(pow(Ex,2)*pow(Ey,2)*pow(vzx + vyx*vzy,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(0, 1) = -(Ex*(Ex*vyx*vzx + Ey*vzy)*(vzx + vyx*vzy)*pow(Ey,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(0, 2) = -(Ex*(vzx + vyx*vzy)*pow(Ey,2)*(Ex*vzx*(vzx + 2*vyx*vzy) + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 1) = -(pow(Ey,2)*pow(Ex*vyx*vzx + Ey*vzy,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 2) = -((Ex*vyx*vzx + Ey*vzy)*pow(Ey,2)*(Ex*vzx*(vzx + 2*vyx*vzy) + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(2, 2) = Ez*(Ey - Ex*pow(vyx,2))*(Ey*(Ez - 2*Ex*vzx*(vzx + 2*vyx*vzy)) - Ex*Ez*pow(vyx,2) - 2*pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 3) {
            d.D(0, 0) = -2*Ey*(Ez*vyx + Ey*vzx*vzy)*pow(Ex,2)*(-Ez + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = Ex*Ey*(Ey*Ez*(Ez - Ex*vzx*(vzx - 2*vyx*vzy)) + Ex*pow(Ez,2)*pow(vyx,2) - pow(Ey,2)*(Ez - 2*Ex*pow(vzx,2))*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 2) = Ex*Ey*Ez*(Ex*(Ez*vyx*(2*vzx + vyx*vzy) + Ey*vzy*pow(vzx,2)) + Ey*vzy*(Ez - Ey*pow(vzy,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 1) = 2*Ex*(Ez*vyx + Ey*vzx*vzy)*pow(Ey,2)*(Ez - Ex*pow(vzx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = Ex*Ey*Ez*(Ex*Ez*vzx*pow(vyx,2) + Ey*(Ez*(vzx + 2*vyx*vzy) - Ex*pow(vzx,3)) + vzx*pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(2, 2) = 2*Ex*Ey*(Ex*vyx*vzx + Ey*vzy)*(vzx + vyx*vzy)*pow(Ez,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 4) {
            d.D(0, 0) = -2*(vzx + vyx*vzy)*pow(Ex,2)*pow(Ey,2)*(-Ez + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = Ex*pow(Ey,2)*(Ex*(Ez*vyx*(2*vzx + vyx*vzy) + Ey*vzy*pow(vzx,2)) + Ey*vzy*(Ez - Ey*pow(vzy,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 2) = -(Ex*Ey*Ez*(Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2) - Ey*(Ez + Ex*(2*vyx*vzx*vzy + pow(vzx,2) + 2*pow(vyx,2)*pow(vzy,2))))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 1) = 2*Ex*(Ex*vyx*vzx + Ey*vzy)*(Ez*vyx + Ey*vzx*vzy)*pow(Ey,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = Ex*Ey*Ez*(vzy*(2*vzx + vyx*vzy)*pow(Ey,2) - Ex*Ez*pow(vyx,3) + Ey*vyx*(Ez + Ex*pow(vzx,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(2, 2) = 2*Ex*Ey*(vzx + vyx*vzy)*pow(Ez,2)*(Ey - Ex*pow(vyx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 5) {
            d.D(0, 0) = 2*(vzx + vyx*vzy)*(Ez*vyx + Ey*vzx*vzy)*pow(Ex,2)*pow(Ey,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = Ex*pow(Ey,2)*(Ex*Ez*vzx*pow(vyx,2) + Ey*(Ez*(vzx + 2*vyx*vzy) - Ex*pow(vzx,3)) + vzx*pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 2) = Ex*Ey*Ez*(vzy*(2*vzx + vyx*vzy)*pow(Ey,2) - Ex*Ez*pow(vyx,3) + Ey*vyx*(Ez + Ex*pow(vzx,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 1) = 2*(Ex*vyx*vzx + Ey*vzy)*pow(Ey,3)*(Ez - Ex*pow(vzx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = Ez*pow(Ey,2)*(Ey*(Ez - Ex*vzx*(vzx - 2*vyx*vzy)) + Ex*pow(vyx,2)*(-Ez + 2*Ex*pow(vzx,2)) + pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(2, 2) = 2*Ey*(Ex*vyx*vzx + Ey*vzy)*pow(Ez,2)*(Ey - Ex*pow(vyx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 6) { d.D(3, 3) = 1; };
        if (p == 7) { d.D(4, 4) = 1; };
        if (p == 8) { d.D(5, 5) = 1; };
    }
    else if (_N == 2) {
        Real Ex = vars[0], Ey = vars[1];
        Real vyx = vars[2];
        if (p == 0) {
            d.D(0, 0) = pow(Ey,2)*pow(Ey - Ex*pow(vyx,2),-2);
            d.D(0, 1) = vyx*pow(Ey,2)*pow(Ey - Ex*pow(vyx,2),-2);
            d.D(1, 1) = pow(Ey,2)*pow(vyx,2)*pow(Ey - Ex*pow(vyx,2),-2);
        };
        if (p == 1) {
            d.D(0, 0) = -(pow(Ex,2)*pow(vyx,2)*pow(Ey - Ex*pow(vyx,2),-2));
            d.D(0, 1) = -(pow(Ex,2)*pow(vyx,3)*pow(Ey - Ex*pow(vyx,2),-2));
            d.D(1, 1) = Ey*(Ey - 2*Ex*pow(vyx,2))*pow(Ey - Ex*pow(vyx,2),-2);
        };
        if (p == 2) {
            d.D(0, 0) = 2*Ey*vyx*pow(Ex,2)*pow(Ey - Ex*pow(vyx,2),-2);
            d.D(0, 1) = Ex*Ey*(Ey + Ex*pow(vyx,2))*pow(Ey - Ex*pow(vyx,2),-2);
            d.D(1, 1) = 2*Ex*vyx*pow(Ey,2)*pow(Ey - Ex*pow(vyx,2),-2);
        };
        if (p == 3) { d.D(2, 2) = 1; };
    }
}

////////////////////////////////////////////////////////////////////////////////

namespace {

#if BOOST_PARSER
    void parseNVector_ptree(size_t N, const ptree &pt, std::vector<Real> &v) {
        v.clear();
        std::runtime_error err("Failed to parse vector of size " + std::to_string(N));
        for (const auto &val : pt) {
            if (!val.first.empty()) throw err;
            v.push_back(val.second.get_value<Real>());
        }
        if (v.size() != N) throw err;
    }

    template<size_t _N>
    void parseIsotropic_ptree(const ptree &pt, ElasticityTensor<Real, _N> &E) {
        Real young = pt.get<Real>("young");
        Real poisson = pt.get<Real>("poisson");
        E.setIsotropic(young, poisson);
    }

    template<size_t _N>
    void parseOrthotropic_ptree(const ptree &pt, ElasticityTensor<Real, _N> &E) {
        std::vector<Real> poisson, young, shear;
        if (_N == 2) {
            parseNVector_ptree(2, pt.get_child("young")  , young);
            parseNVector_ptree(2, pt.get_child("poisson"), poisson);
            parseNVector_ptree(1, pt.get_child("shear")  , shear);
            // Ex, Ey, nuYX, muXY
            Real   E_x(  young[0]),   E_y(  young[1]),
                 nu_xy(poisson[0]), nu_yx(poisson[1]), mu(shear[0]);
            E.setOrthotropic2D(E_x, E_y, nu_yx, mu);

            if (std::abs(nu_yx / E_y - nu_xy / E_x) > 1e-10)
                throw std::runtime_error("Orthotopic parameters violate symmetry");
        }
        else {
            parseNVector_ptree(3, pt.get_child("young")  , young);
            parseNVector_ptree(6, pt.get_child("poisson"), poisson);
            parseNVector_ptree(3, pt.get_child("shear")  , shear);
            Real   E_x(  young[0]),   E_y(  young[1]), E_z(young[2]),
                 nu_yz(poisson[0]), nu_zy(poisson[1]),
                 nu_zx(poisson[2]), nu_xz(poisson[3]),
                 nu_xy(poisson[4]), nu_yx(poisson[5]),
                 mu_yz(  shear[0]), mu_zx(  shear[1]), mu_xy(shear[2]);

            E.setOrthotropic3D(E_x, E_y, E_z, nu_yx, nu_zx, nu_zy,
                               mu_yz, mu_zx, mu_xy);

            if ((std::abs(nu_yx / E_y - nu_xy / E_x) > 1e-10) ||
                (std::abs(nu_yz / E_y - nu_zy / E_z) > 1e-10) ||
                (std::abs(nu_zx / E_z - nu_xz / E_x) > 1e-10)) {
                throw std::runtime_error("Orthotopic parameters violate symmetry");
            }
        }
    }

    template<size_t _N>
    void parseAnisotropic_ptree(const ptree &pt, ElasticityTensor<Real, _N> &E) {
        std::runtime_error err("Failed to parse material_matrix");
        size_t row = 0;
        for (const auto &rpt : pt.get_child("material_matrix")) {
            if (!rpt.first.empty()) throw err;
            size_t col = 0;
            for (const auto &cpt : rpt.second) {
                if (!cpt.first.empty()) throw err;
                Real val = cpt.second.get_value<Real>();
                if (row <= col)
                    E.D(row, col) = val;
                else if (std::abs(E.D(row, col) - val) > 1e-10) {
                    throw std::runtime_error("Asymmetric material_matrix");
                }
                ++col;
            }
            ++row;
        }
    }
#endif

    void parseNVector(size_t N, const nlohmann::json &entry, std::vector<Real> &v) {
        v = entry.get<std::vector<Real>>();
        if (v.size() != N) {
            throw std::runtime_error("Failed to parse vector of size " + std::to_string(N));
        }
    }

    // Expected values:
    // "young": #
    // "poisson": #
    template<size_t _N>
    void parseIsotropic(const nlohmann::json &entry, ElasticityTensor<Real, _N> &E) {
        Real young = entry["young"];
        Real poisson = entry["poisson"];
        E.setIsotropic(young, poisson);
    }

    // Expected values:
    // 3D: "young": [young_x, young_y, young_z],
    //     "poisson": [poisson_yz, poisson_zy,
    //                 poisson_zx, poisson_xz,
    //                 poisson_xy, poisson_yx],
    //     "shear": [shear_yz, shear_zx, shear_xy],
    // 2D: "young": [young_x, young_y],
    //     "poisson": [poisson_xy, poisson_yx],
    //     "shear": [shear_xy],
    template<size_t _N>
    void parseOrthotropic(const nlohmann::json &entry, ElasticityTensor<Real, _N> &E) {
        std::vector<Real> poisson, young, shear;
        if (_N == 2) {
            parseNVector(2, entry["young"]  , young);
            parseNVector(2, entry["poisson"], poisson);
            parseNVector(1, entry["shear"]  , shear);
            // Ex, Ey, nuYX, muXY
            Real   E_x(  young[0]),   E_y(  young[1]),
                 nu_xy(poisson[0]), nu_yx(poisson[1]), mu(shear[0]);
            E.setOrthotropic2D(E_x, E_y, nu_yx, mu);

            if (std::abs(nu_yx / E_y - nu_xy / E_x) > 1e-10)
                throw std::runtime_error("Orthotopic parameters violate symmetry");
        }
        else {
            parseNVector(3, entry["young"]  , young);
            parseNVector(6, entry["poisson"], poisson);
            parseNVector(3, entry["shear"]  , shear);
            Real   E_x(  young[0]),   E_y(  young[1]), E_z(young[2]),
                 nu_yz(poisson[0]), nu_zy(poisson[1]),
                 nu_zx(poisson[2]), nu_xz(poisson[3]),
                 nu_xy(poisson[4]), nu_yx(poisson[5]),
                 mu_yz(  shear[0]), mu_zx(  shear[1]), mu_xy(shear[2]);

            E.setOrthotropic3D(E_x, E_y, E_z, nu_yx, nu_zx, nu_zy,
                               mu_yz, mu_zx, mu_xy);

            if ((std::abs(nu_yx / E_y - nu_xy / E_x) > 1e-10) ||
                (std::abs(nu_yz / E_y - nu_zy / E_z) > 1e-10) ||
                (std::abs(nu_zx / E_z - nu_xz / E_x) > 1e-10)) {
                throw std::runtime_error("Orthotopic parameters violate symmetry");
            }
        }
    }

    // Expected values:
    // "material_matrix": [[C_00, C_01, C02, C03, C04, C05],
    //                     [C_10, C_11, C12, C13, C14, C15],
    //                     [C_20, C_21, C22, C23, C24, C25],
    //                     [C_30, C_31, C32, C33, C34, C35],
    //                     [C_40, C_41, C42, C43, C44, C45],
    //                     [C_50, C_51, C52, C53, C54, C55]],
    template<size_t _N>
    void parseAnisotropic(const nlohmann::json &entry, ElasticityTensor<Real, _N> &E) {
        std::runtime_error err("Failed to parse material_matrix");
        size_t row = 0;
        for (const auto &rpt : entry["material_matrix"]) {
            if (rpt.size() != flatLen(_N)) { throw err; }
            size_t col = 0;
            for (Real val : rpt) {
                if (row <= col) {
                    E.D(row, col) = val;
                } else if (std::abs(E.D(row, col) - val) > 1e-10) {
                    throw std::runtime_error("Asymmetric material_matrix");
                }
                ++col;
            }
            ++row;
        }
    }

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

#if BOOST_PARSER
template<size_t _N>
void Constant<_N>::setFromPTree(const ptree &pt) {
    std::string type = pt.get<std::string>("type");
    if (type == "isotropic_material")        { parseIsotropic_ptree<_N>(pt, m_E);   }
    else if (type == "isotropic")            { parseIsotropic_ptree<_N>(pt, m_E);   }
    else if (type == "orthotropic_material") { parseOrthotropic_ptree<_N>(pt, m_E); }
    else if (type == "orthotropic")          { parseOrthotropic_ptree<_N>(pt, m_E); }
    else if (type == "symmetric_material")   { parseAnisotropic_ptree<_N>(pt, m_E); }
    else if (type == "anisotropic")          { parseAnisotropic_ptree<_N>(pt, m_E); }
    else { throw std::runtime_error("Invalid type."); }
}
#endif

template<size_t _N>
void Constant<_N>::setFromJson(const nlohmann::json &config) {
    std::string type = config["type"];
    if (type == "isotropic_material")        { parseIsotropic<_N>(config, m_E);   }
    else if (type == "isotropic")            { parseIsotropic<_N>(config, m_E);   }
    else if (type == "orthotropic_material") { parseOrthotropic<_N>(config, m_E); }
    else if (type == "orthotropic")          { parseOrthotropic<_N>(config, m_E); }
    else if (type == "symmetric_material")   { parseAnisotropic<_N>(config, m_E); }
    else if (type == "anisotropic")          { parseAnisotropic<_N>(config, m_E); }
    else { throw std::runtime_error("Invalid type."); }
}

template<size_t _N>
void Constant<_N>::setFromFile(const std::string &materialPath) {
    std::ifstream is(materialPath);
    if (!is.is_open()) {
        throw std::runtime_error("Couldn't open material " + materialPath);
    }
    nlohmann::json config;
    is >> config;
    setFromJson(config);
}

template<size_t _N>
nlohmann::json Constant<_N>::getJson() const {
    nlohmann::json config;

    std::array<std::array<Real, flatLen(N)>, flatLen(N)> mat;
    for (size_t i = 0; i < flatLen(N); ++i) {
        for (size_t j = 0; j < flatLen(N); ++j) {
            mat[i][j] = m_E.D(i, j);
        }
    }

    config["type"] = "anisotropic";
    config["material_matrix"] = mat;

    return config;
}

////////////////////////////////////////////////////////////////////////////////

namespace {

#if 0
    void parseVariableBounds_ptree(const ptree &pt, std::vector<Bounds::Bound> &bounds) {
        bounds.clear();
        std::runtime_error err("Failed to parse variable bounds.");
        std::vector<Real> tmp;
        for (const auto &bd : pt) {
            if (!bd.first.empty()) throw err;
            parseNVector(2, bd.second, tmp);
            size_t var = tmp[0];
            if ((double) var != tmp[0])
                throw std::runtime_error("Bound on non-integer variable index.");
            bounds.push_back(Bounds::Bound(var, tmp[1]));
        }
    }
#endif

    void parseVariableBounds(const nlohmann::json &entry, std::vector<Bounds::Bound> &bounds) {
        bounds.clear();
        std::vector<Real> tmp;
        for (const auto &bd : entry) {
            parseNVector(2, bd, tmp);
            size_t var = tmp[0];
            if ((double) var != tmp[0]) {
                throw std::runtime_error("Bound on non-integer variable index.");
            }
            bounds.push_back(Bounds::Bound(var, tmp[1]));
        }
    }

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

void MESHFEM_EXPORT Bounds::setFromJson(const nlohmann::json &config) {
    parseVariableBounds(config["lower"], m_lower);
    parseVariableBounds(config["upper"], m_upper);
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
// Has the nice side-effect that only code using valid dimensions 2 and 3 links.
////////////////////////////////////////////////////////////////////////////////
template struct MESHFEM_EXPORT Isotropic<2>;
template struct MESHFEM_EXPORT Isotropic<3>;

template struct MESHFEM_EXPORT Orthotropic<2>;
template struct MESHFEM_EXPORT Orthotropic<3>;

template struct MESHFEM_EXPORT Constant<2>;
template struct MESHFEM_EXPORT Constant<3>;

}
