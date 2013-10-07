////////////////////////////////////////////////////////////////////////////////
// AnalysisSettings.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Stores (and saves/parses) all the settings for CSGFEM.
//
//      These are now stored as a boost::variant-based database. This allows
//      more easy maintenance/saving/flexible queries at the cost of slightly
//      more verbose access (must use type-specifying accessors) and less
//      compile-time checks.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/01/2013 23:39:01
////////////////////////////////////////////////////////////////////////////////
#ifndef ANALYSIS_SETTINGS_HH
#define ANALYSIS_SETTINGS_HH
#include <boost/program_options.hpp>
#include <boost/variant.hpp>
#include <string>
#include <cassert>

namespace po = boost::program_options;

#include "GlobalTypes.hh"
#include "Quadrature.hh"
#include "utils.hh"

struct AnalysisSettings {
    typedef boost::variant<std::string, int, double, bool> Variant;
public:
    // Note: these must match the Variant typedef order.
    typedef enum {
        TYPE_STRING = 0,
        TYPE_INT = 1,
        TYPE_REAL = 2,
        TYPE_BOOL = 3
    } Type;

    AnalysisSettings() {
        // Be careful with initialization literal types here... they determine
        // the setting's type, and need to match exactly to avoid overload
        // ambiguity.
        m_values["solver"] = Variant(std::string("Gurobi"));
        m_values["Nx"] = Variant((int) 40);
        m_values["Ny"] = Variant((int) 40);
        m_values["borderWidth"] = Variant((int) 1);
        m_values["quadrature"] = Variant((int) UNIFORM_QUADRATURE);
        m_values["quadraturePoints"] = Variant((int) 81);
        m_values["cellOverlapThreshold"] = Variant((double) 0.15);

        m_values["useMSBoundary"] = Variant((bool) false);
        m_values["blurPointForces"] = Variant((bool) true);
        m_values["boundarySpacing"] = Variant((double) .15);
        m_values["kernelRadius"] = Variant((double) 1.0);

        m_values["exactFullElements"] = Variant((bool) true);
        m_values["antialiasedElements"] = Variant((bool) false);

        m_values["massMatrixType"] = Variant((int) MASS_QUARTER_CELL);

        m_values["laplacianModes"] = Variant((bool) false);
        m_values["consistentSigns"] = Variant((bool) true);
        m_values["numModes"] = Variant((int) 10);

        m_values["weakRegionsPerMode"] = Variant((int) 5);
        m_values["weaknessCutoff"] = Variant((double) 0.95);
        m_values["abstrace"] = Variant((bool) true);
        m_values["plusMinusObjective"] = Variant((bool) true);

        m_values["totalForceBound"] = Variant((double) 0.1);
        m_values["pointwisePressureBound"] = Variant((double) 0.1);

        m_values["fixedTranslation"] = Variant((bool) false);
        m_values["xTranslation"] = Variant((double) 0.0);
        m_values["yTranslation"] = Variant((double) 0.0);

        m_values["young_modulus"] = Variant((double) 1.0);
        m_values["poisson_ratio"] = Variant((double) 0.0);
        m_values["density"] = Variant((double) 1.0);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Accessors
    // Throw exceptions if settings name isn't found, or if wrong typed accessor
    // is called.
    ////////////////////////////////////////////////////////////////////////////
    bool          Bool(const std::string &name) const { return boost::get<bool>(m_values.at(name)); }
    int            Int(const std::string &name) const { return boost::get<int>(m_values.at(name)); }
    int           Enum(const std::string &name) const { return Int(name); }
    double        Real(const std::string &name) const { return boost::get<double>(m_values.at(name)); }
    const std::string &String(const std::string &name) const { return boost::get<std::string>(m_values.at(name)); }

    bool        &   Bool(const std::string &name) { return boost::get<bool>(m_values.at(name)); }
    int         &    Int(const std::string &name) { return boost::get<int>(m_values.at(name)); }
    int         &   Enum(const std::string &name) { return Int(name); }
    double      &   Real(const std::string &name) { return boost::get<double>(m_values.at(name)); }
    std::string & String(const std::string &name) { return boost::get<std::string>(m_values.at(name)); }

    std::string displayString(const std::string &name) const {
        switch (type(name)) {
            case TYPE_STRING:
                return String(name);
            case TYPE_INT:
                return toString(Int(name));
            case TYPE_BOOL:
                return Bool(name) ? "true" : "false";
            case TYPE_REAL:
                return toString(Real(name));
            default:
                assert(false);
        }
        return "";
    }

    // Memberwise comparator
    bool operator==(const AnalysisSettings &rhs) const {
        for (const auto &rpair : rhs.m_values) {
            auto it = m_values.find(rpair.first);
            if (it == m_values.end())
                return false;

            if (!(it->second == rpair.second))
                return false;
        }
        return true;
    }

    Type type(const std::string &name) const { return (Type) get(name).which(); }

    Variant &get(const std::string &name) { return m_values.at(name); }
    const Variant &get(const std::string &name) const { return m_values.at(name); }

    std::vector<std::string> getNames() const {
        std::vector<std::string> keys;
        for (const auto &pair : m_values) {
            keys.push_back(pair.first);
        }
        return keys;
    }

    void getOptions(po::options_description &opts) const {
        opts.add_options()
            ("Nx", po::value<int>()->default_value(40), "Grid rows")
            ("Ny", po::value<int>()->default_value(40), "Grid columns")
            ("quadrature", po::value<std::string>()->default_value("uniform"), "Quadrature type")
            ("quadrature_points", po::value<int>()->default_value(81), "Number of quadrature points")
            ("cell_overlap_threshold", po::value<double>()->default_value(0.15), "Quad point fraction needed to qualify as a cell")
            ("ms_boundary", "Use marching squares boundary")
            ("boundary_spacing", po::value<double>()->default_value(.02), "Boundary point spacing (when use_ms_boundary is false)")
            ("mass_matrix_type", po::value<std::string>()->default_value("quarter_cell"), "Type of mass matrix")
            ("laplacian_modes", "Use laplacian eigenvectors as modes.")
            ("num_modes", po::value<int>()->default_value(10), "Number of modes to compute")
            ("weak_regions_per_mode", po::value<int>()->default_value(5), "Number of weak regions to extract per mode")
            ("weakness_cutoff", po::value<double>()->default_value(.95), "Stress norm percentile above which a cell is considered weak")
            ("total_force_bound", po::value<double>()->default_value(0.1), "F_tot: equality constraint for the total force")
            ("pointwise_pressure_bound", po::value<double>()->default_value(0.1), "p_max: maximum pressure at each boundary point")

            ("young_modulus", po::value<double>()->default_value(1.0), "Material's young modulus")
            ("poisson_ratio", po::value<double>()->default_value(0.0), "Material's poisson ratio")
            ("density", po::value<double>()->default_value(1.0), "Material's density")
        ;
    }
private:
    std::map<std::string, Variant> m_values;
};

#endif // ANALYSIS_SETTINGS_HH
