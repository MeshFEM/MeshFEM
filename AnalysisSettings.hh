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
//
//      Reading settings is done with boost::program_options, and writing is
//      done with boost::property_tree. When a new setting is added, getOptions,
//      parseOptions, and writeOptions must all be updated.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/01/2013 23:39:01
////////////////////////////////////////////////////////////////////////////////
#ifndef ANALYSIS_SETTINGS_HH
#define ANALYSIS_SETTINGS_HH
#include <boost/program_options.hpp>
#include <boost/variant.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
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

    AnalysisSettings(std::istream &is) {
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
            ("global.solver", po::value<std::string>()->default_value("Gurobi"), "Which solver to use")
            ;

        elementsOpt.add_options()
            ("elements.Nx", po::value<int>()->default_value(40), "Grid columns")
            ("elements.Ny", po::value<int>()->default_value(40), "Grid rows")
            ("elements.borderWidth", po::value<int>()->default_value(1), "Grid border width")
            ("elements.quadrature", po::value<int>()->default_value(UNIFORM_QUADRATURE), "Type of quadrature")
            ("elements.quadraturePoints", po::value<int>()->default_value(81), "Number of quadrature points")
            ("elements.cellOverlapThreshold", po::value<double>()->default_value(0.05), "Overlap threshold above which a cell is an element")
            ("elements.exactFullElements", po::value<bool>()->default_value(true), "Used closed formula for integrals over full elements")
            ("elements.antialiasedElements", po::value<bool>()->default_value(false), "Treat cut cells as full cells with lower density")
            ;

        // Materials
        materialOpt.add_options()
            ("material.young_modulus", po::value<double>()->default_value(1.0), "Youngs modulus")
            ("material.poisson_ratio", po::value<double>()->default_value(0.0), "Poisson ratio")
            ("material.density", po::value<double>()->default_value(1.0), "Material density")
            ;

        // Simulation
        simulationOpt.add_options()
            ("simulation.useMSBoundary", po::value<bool>()->default_value(false), "Get boundary points from marching squares")
            ("simulation.boundarySpacing", po::value<double>()->default_value(.03), "Spacing between boundary points (if useMSBoundary is false)")
            ("simulation.blurPointForces", po::value<bool>()->default_value(true), "Blur point forces into volume forces")
            ("simulation.kernelRadius", po::value<double>()->default_value(1.0), "Blur radius")
            ;


        // Modal analysis
        modalAnalysisOpt.add_options()
            ("modalAnalysis.massMatrixType", po::value<int>()->default_value(MASS_QUARTER_CELL), "Type of mass matrix to use")
            ("modalAnalysis.laplacianModes", po::value<bool>()->default_value(false), "Use eigenfunctions of the Laplacian as modes")
            ("modalAnalysis.consistentSigns", po::value<bool>()->default_value(true), "Prevent randomness in mode signs by always making their max stress compressive")
            ("modalAnalysis.numModes", po::value<int>()->default_value(10), "Number of modes to compute")
            ;

        // Weakness analysis
        weaknessAnalysisOpt.add_options()
            ("weaknessAnalysis.weakRegionsPerMode", po::value<int>()->default_value(5), "Number of weak regions to extract permode")
            ("weaknessAnalysis.weaknessCutoff", po::value<double>()->default_value(0.95), "Percentile above which regions are considered weak")
            ("weaknessAnalysis.abstrace", po::value<bool>()->default_value(true), "Use modal stress signs to choose between optimizing +trace or -trace on a per-element basis")
            ("weaknessAnalysis.plusMinusObjective", po::value<bool>()->default_value(true), "Optimize both the + and the - objective")
            ("weaknessAnalysis.totalForceBound", po::value<double>()->default_value(0.1), "Total force allowed on the object")
            ("weaknessAnalysis.pointwisePressureBound", po::value<double>()->default_value(0.1), "Maximum pointwise pressure allowed (prevent needle poke)")
            ;

        // Private
        privateOpt.add_options()
            ("private.fixedTranslation", po::value<bool>()->default_value(false), "Use a fixed translation instead of a sweep for translation test")
            ("private.xTranslation", po::value<double>()->default_value(0.0), "Fixed translation x")
            ("private.yTranslation", po::value<double>()->default_value(0.0), "Fixed translation y")
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
        m_values["solver"] =                 Variant(vm["global.solver"].as<std::string>());

        // Elements
        m_values["Nx"] =                     Variant(vm["elements.Nx"].as<int>());
        m_values["Ny"] =                     Variant(vm["elements.Ny"].as<int>());
        m_values["borderWidth"] =            Variant(vm["elements.borderWidth"].as<int>());
        m_values["quadrature"] =             Variant(vm["elements.quadrature"].as<int>());
        m_values["quadraturePoints"] =       Variant(vm["elements.quadraturePoints"].as<int>());
        m_values["cellOverlapThreshold"] =   Variant(vm["elements.cellOverlapThreshold"].as<double>());
        m_values["exactFullElements"] =      Variant(vm["elements.exactFullElements"].as<bool>());
        m_values["antialiasedElements"] =    Variant(vm["elements.antialiasedElements"].as<bool>());

        // Materials
        m_values["young_modulus"] =          Variant(vm["material.young_modulus"].as<double>());
        m_values["poisson_ratio"] =          Variant(vm["material.poisson_ratio"].as<double>());
        m_values["density"] =                Variant(vm["material.density"].as<double>());

        // Simulation
        m_values["useMSBoundary"] =          Variant(vm["simulation.useMSBoundary"].as<bool>());
        m_values["boundarySpacing"] =        Variant(vm["simulation.boundarySpacing"].as<double>());
        m_values["blurPointForces"] =        Variant(vm["simulation.blurPointForces"].as<bool>());
        m_values["kernelRadius"] =           Variant(vm["simulation.kernelRadius"].as<double>());


        // Modal analysis
        m_values["massMatrixType"] =         Variant(vm["modalAnalysis.massMatrixType"].as<int>());
        m_values["laplacianModes"] =         Variant(vm["modalAnalysis.laplacianModes"].as<bool>());
        m_values["consistentSigns"] =        Variant(vm["modalAnalysis.consistentSigns"].as<bool>());
        m_values["numModes"] =               Variant(vm["modalAnalysis.numModes"].as<int>());

        // Weakness analysis
        m_values["weakRegionsPerMode"] =     Variant(vm["weaknessAnalysis.weakRegionsPerMode"].as<int>());
        m_values["weaknessCutoff"] =         Variant(vm["weaknessAnalysis.weaknessCutoff"].as<double>());
        m_values["abstrace"] =               Variant(vm["weaknessAnalysis.abstrace"].as<bool>());
        m_values["plusMinusObjective"] =     Variant(vm["weaknessAnalysis.plusMinusObjective"].as<bool>());
        m_values["totalForceBound"] =        Variant(vm["weaknessAnalysis.totalForceBound"].as<double>());
        m_values["pointwisePressureBound"] = Variant(vm["weaknessAnalysis.pointwisePressureBound"].as<double>());

        // Private
        m_values["fixedTranslation"] =       Variant(vm["private.fixedTranslation"].as<bool>());
        m_values["xTranslation"] =           Variant(vm["private.xTranslation"].as<double>());
        m_values["yTranslation"] =           Variant(vm["private.yTranslation"].as<double>());
    }

    void writeOptions(std::ostream &os) const {
        // boost::program_options doesn't support writing .ini files, so we
        // convert to a property_tree. Notice that property tree groups are
        // written as groups in the .ini, so we don't need to add the group name
        // to each setting (e.g. we write Nx instead of elements.Nx).
        using boost::property_tree::ptree;
        ptree root;

        ptree globalOpt, elementsOpt, materialOpt, modalAnalysisOpt,
              simulationOpt, weaknessAnalysisOpt, privateOpt;
        
        globalOpt.put("solver", boost::lexical_cast<std::string>(m_values.at("solver")));

        // Elements
        elementsOpt.put("Nx", boost::lexical_cast<std::string>(m_values.at("Nx")));
        elementsOpt.put("Ny", boost::lexical_cast<std::string>(m_values.at("Ny")));
        elementsOpt.put("borderWidth", boost::lexical_cast<std::string>(m_values.at("borderWidth")));
        elementsOpt.put("quadrature", boost::lexical_cast<std::string>(m_values.at("quadrature")));
        elementsOpt.put("quadraturePoints", boost::lexical_cast<std::string>(m_values.at("quadraturePoints")));
        elementsOpt.put("cellOverlapThreshold", boost::lexical_cast<std::string>(m_values.at("cellOverlapThreshold")));
        elementsOpt.put("exactFullElements", boost::lexical_cast<std::string>(m_values.at("exactFullElements")));
        elementsOpt.put("antialiasedElements", boost::lexical_cast<std::string>(m_values.at("antialiasedElements")));

        // Materials
        materialOpt.put("young_modulus", boost::lexical_cast<std::string>(m_values.at("young_modulus")));
        materialOpt.put("poisson_ratio", boost::lexical_cast<std::string>(m_values.at("poisson_ratio")));
        materialOpt.put("density", boost::lexical_cast<std::string>(m_values.at("density")));

        // Simulation
        simulationOpt.put("useMSBoundary", boost::lexical_cast<std::string>(m_values.at("useMSBoundary")));
        simulationOpt.put("boundarySpacing", boost::lexical_cast<std::string>(m_values.at("boundarySpacing")));
        simulationOpt.put("blurPointForces", boost::lexical_cast<std::string>(m_values.at("blurPointForces")));
        simulationOpt.put("kernelRadius", boost::lexical_cast<std::string>(m_values.at("kernelRadius")));


        // Modal analysis
        modalAnalysisOpt.put("massMatrixType", boost::lexical_cast<std::string>(m_values.at("massMatrixType")));
        modalAnalysisOpt.put("laplacianModes", boost::lexical_cast<std::string>(m_values.at("laplacianModes")));
        modalAnalysisOpt.put("consistentSigns", boost::lexical_cast<std::string>(m_values.at("consistentSigns")));
        modalAnalysisOpt.put("numModes", boost::lexical_cast<std::string>(m_values.at("numModes")));

        // Weakness analysis
        weaknessAnalysisOpt.put("weakRegionsPerMode", boost::lexical_cast<std::string>(m_values.at("weakRegionsPerMode")));
        weaknessAnalysisOpt.put("weaknessCutoff", boost::lexical_cast<std::string>(m_values.at("weaknessCutoff")));
        weaknessAnalysisOpt.put("abstrace", boost::lexical_cast<std::string>(m_values.at("abstrace")));
        weaknessAnalysisOpt.put("plusMinusObjective", boost::lexical_cast<std::string>(m_values.at("plusMinusObjective")));
        weaknessAnalysisOpt.put("totalForceBound", boost::lexical_cast<std::string>(m_values.at("totalForceBound")));
        weaknessAnalysisOpt.put("pointwisePressureBound", boost::lexical_cast<std::string>(m_values.at("pointwisePressureBound")));

        // Private
        privateOpt.put("fixedTranslation", boost::lexical_cast<std::string>(m_values.at("fixedTranslation")));
        privateOpt.put("xTranslation", boost::lexical_cast<std::string>(m_values.at("xTranslation")));
        privateOpt.put("yTranslation", boost::lexical_cast<std::string>(m_values.at("yTranslation")));

        root.push_back(ptree::value_type("global", globalOpt));
        root.push_back(ptree::value_type("elements", elementsOpt));
        root.push_back(ptree::value_type("material", materialOpt));
        root.push_back(ptree::value_type("modalAnalysis", modalAnalysisOpt));
        root.push_back(ptree::value_type("simulation", simulationOpt));
        root.push_back(ptree::value_type("weaknessAnalysis", weaknessAnalysisOpt));
        root.push_back(ptree::value_type("private",  privateOpt));

        write_ini(os, root);
    }

private:
    std::map<std::string, Variant> m_values;
};

#endif // ANALYSIS_SETTINGS_HH
