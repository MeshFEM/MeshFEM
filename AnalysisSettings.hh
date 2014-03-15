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
#include <fstream>
#include <sstream>
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

    AnalysisSettings(std::ifstream &is) {
        parseOptions(is);
    }
    AnalysisSettings(std::string &path) {
        std::ifstream settingsFile(path);
        if (!settingsFile.is_open())	{
            std::cout << "WARNING: failed to read settings file" << path << '\''
                << std::endl;
            parseOptions();
        }
        else {
            parseOptions(settingsFile);
        }
    }

    AnalysisSettings() {
        parseOptions();
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
        po::options_description globalOpt("global"), elementsOpt("elements"),
                                materialOpt("material"),
                                modalAnalysisOpt("modalAnalysis"),
                                simulationOpt("simulation"),
                                weaknessAnalysisOpt("weaknessAnalysis"),
                                privateOpt("private");
        // NOTE: DEFAULTS SPECIFIED HERE
        globalOpt.add_options()
            ("solver", po::value<std::string>()->default_value("Gurobi"), "Which solver to use")
            ;

        elementsOpt.add_options()
            ("Nx", po::value<int>()->default_value(40), "Grid columns")
            ("Ny", po::value<int>()->default_value(40), "Grid rows")
            ("borderWidth", po::value<int>()->default_value(1), "Grid border width")
            ("quadrature", po::value<int>()->default_value(UNIFORM_QUADRATURE), "Type of quadrature")
            ("quadraturePoints", po::value<int>()->default_value(81), "Number of quadrature points")
            ("cellOverlapThreshold", po::value<double>()->default_value(0.05), "Overlap threshold above which a cell is an element")
            ("exactFullElements", po::value<bool>()->default_value(true), "Used closed formula for integrals over full elements")
            ("antialiasedElements", po::value<bool>()->default_value(false), "Treat cut cells as full cells with lower density")
            ;

        // Materials
        materialOpt.add_options()
            ("young_modulus", po::value<double>()->default_value(1.0), "Youngs modulus")
            ("poisson_ratio", po::value<double>()->default_value(0.0), "Poisson ratio")
            ("density", po::value<double>()->default_value(1.0), "Material density")
            ;

        // Simulation
        simulationOpt.add_options()
            ("useMSBoundary", po::value<bool>()->default_value(false), "Get boundary points from marching squares")
            ("boundarySpacing", po::value<double>()->default_value(.03), "Spacing between boundary points (if useMSBoundary is false)")
            ("blurPointForces", po::value<bool>()->default_value(true), "Blur point forces into volume forces")
            ("kernelRadius", po::value<double>()->default_value(1.0), "Blur radius")
            ;


        // Modal analysis
        modalAnalysisOpt.add_options()
            ("massMatrixType", po::value<int>()->default_value(MASS_QUARTER_CELL), "Type of mass matrix to use")
            ("laplacianModes", po::value<bool>()->default_value(false), "Use eigenfunctions of the Laplacian as modes")
            ("consistentSigns", po::value<bool>()->default_value(true), "Prevent randomness in mode signs by always making their max stress compressive")
            ("numModes", po::value<int>()->default_value(10), "Number of modes to compute")
            ;

        // Weakness analysis
        weaknessAnalysisOpt.add_options()
            ("weakRegionsPerMode", po::value<int>()->default_value(5), "Number of weak regions to extract permode")
            ("weaknessCutoff", po::value<double>()->default_value(0.95), "Percentile above which regions are considered weak")
            ("abstrace", po::value<bool>()->default_value(true), "Use modal stress signs to choose between optimizing +trace or -trace on a per-element basis")
            ("plusMinusObjective", po::value<bool>()->default_value(true), "Optimize both the + and the - objective")
            ("totalForceBound", po::value<double>()->default_value(0.1), "Total force allowed on the object")
            ("pointwisePressureBound", po::value<double>()->default_value(0.1), "Maximum pointwise pressure allowed (prevent needle poke)")
            ;

        // Private
        privateOpt.add_options()
            ("fixedTranslation", po::value<bool>()->default_value(false), "Use a fixed translation instead of a sweep for translation test")
            ("xTranslation", po::value<double>()->default_value(0.0), "Fixed translation x")
            ("yTranslation", po::value<double>()->default_value(0.0), "Fixed translation y")
            ;

        opts.add(globalOpt).add(elementsOpt).add(materialOpt).
             add(modalAnalysisOpt).add(simulationOpt).add(weaknessAnalysisOpt).
             add(privateOpt);
    }

    void parseOptions(const std::string &s = std::string()) {
        std::stringstream ss(s);
        parseOptions(ss);
    }

    void parseOptions(std::istream &is) {
        po::options_description opts;
        getOptions(opts);


        po::variables_map vm;
        po::store(po::parse_config_file(is, opts), vm);

        // Be careful with initialization literal types here... they determine
        // the setting's type, and need to match exactly to avoid overload
        // ambiguity.
        m_values["solver"] = Variant(vm["solver"].as<std::string>());

        // Elements
        m_values["Nx"] = Variant(vm["Nx"].as<int>());
        m_values["Ny"] = Variant(vm["Ny"].as<int>());
        m_values["borderWidth"] = Variant(vm["borderWidth"].as<int>());
        m_values["quadrature"] = Variant(vm["quadrature"].as<int>());
        m_values["quadraturePoints"] = Variant(vm["quadraturePoints"].as<int>());
        m_values["cellOverlapThreshold"] = Variant(vm["cellOverlapThreshold"].as<double>());
        m_values["exactFullElements"] = Variant(vm["exactFullElements"].as<bool>());
        m_values["antialiasedElements"] = Variant(vm["antialiasedElements"].as<bool>());

        // Materials
        m_values["young_modulus"] = Variant(vm["young_modulus"].as<double>());
        m_values["poisson_ratio"] = Variant(vm["poisson_ratio"].as<double>());
        m_values["density"] = Variant(vm["density"].as<double>());

        // Simulation
        m_values["useMSBoundary"] = Variant(vm["useMSBoundary"].as<bool>());
        m_values["boundarySpacing"] = Variant(vm["boundarySpacing"].as<double>());
        m_values["blurPointForces"] = Variant(vm["blurPointForces"].as<bool>());
        m_values["kernelRadius"] = Variant(vm["kernelRadius"].as<double>());


        // Modal analysis
        m_values["massMatrixType"] = Variant(vm["massMatrixType"].as<int>());
        m_values["laplacianModes"] = Variant(vm["laplacianModes"].as<bool>());
        m_values["consistentSigns"] = Variant(vm["consistentSigns"].as<bool>());
        m_values["numModes"] = Variant(vm["numModes"].as<int>());

        // Weakness analysis
        m_values["weakRegionsPerMode"] = Variant(vm["weakRegionsPerMode"].as<int>());
        m_values["weaknessCutoff"] = Variant(vm["weaknessCutoff"].as<double>());
        m_values["abstrace"] = Variant(vm["abstrace"].as<bool>());
        m_values["plusMinusObjective"] = Variant(vm["plusMinusObjective"].as<bool>());
        m_values["totalForceBound"] = Variant(vm["totalForceBound"].as<double>());
        m_values["pointwisePressureBound"] = Variant(vm["pointwisePressureBound"].as<double>());

        // Private
        m_values["fixedTranslation"] = Variant(vm["fixedTranslation"].as<bool>());
        m_values["xTranslation"] = Variant(vm["xTranslation"].as<double>());
        m_values["yTranslation"] = Variant(vm["yTranslation"].as<double>());
    }

private:
    std::map<std::string, Variant> m_values;
};

#endif // ANALYSIS_SETTINGS_HH
