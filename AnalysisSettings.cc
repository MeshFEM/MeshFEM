#include "AnalysisSettings.hh"

#include <iostream>
#include <sstream>
#include <memory>

#include <boost/variant.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/lexical_cast.hpp>

////////////////////////////////////////////////////////////////////////////
// Accessors
// Throw exceptions if settings name isn't found, or if wrong typed accessor
// is called.
////////////////////////////////////////////////////////////////////////////
bool               AnalysisSettings::  Bool(const std::string &name) const { return boost::get<bool>(get(name)); }
int                AnalysisSettings::   Int(const std::string &name) const { return boost::get<int>(get(name)); }
int                AnalysisSettings::  Enum(const std::string &name) const { return Int(name); }
double             AnalysisSettings::  Real(const std::string &name) const { return boost::get<double>(get(name)); }
const std::string &AnalysisSettings::String(const std::string &name) const { return boost::get<std::string>(get(name)); }

bool        & AnalysisSettings::  Bool(const std::string &name) { return boost::get<bool>(get(name)); }
int         & AnalysisSettings::   Int(const std::string &name) { return boost::get<int>(get(name)); }
int         & AnalysisSettings::  Enum(const std::string &name) { return Int(name); }
double      & AnalysisSettings::  Real(const std::string &name) { return boost::get<double>(get(name)); }
std::string & AnalysisSettings::String(const std::string &name) { return boost::get<std::string>(get(name)); }

std::string AnalysisSettings::displayString(const std::string &name) const {
    switch (type(name)) {
        case TYPE_STRING:
            return String(name);
        case TYPE_INT:
            return std::to_string(Int(name));
        case TYPE_BOOL:
            return Bool(name) ? "true" : "false";
        case TYPE_REAL:
            return std::to_string(Real(name));
        default:
            assert(false);
    }
    return "";
}

// Memberwise comparator
bool AnalysisSettings::operator==(const AnalysisSettings &rhs) const {
    for (const auto &rpair : rhs.m_values) {
        auto it = m_values.find(rpair.first);
        if (it == m_values.end())
            return false;

        if (!(it->first == rpair.first))
            return false;

        if (!(*(it->second) == *(rpair.second)))
            return false;
    }

    return true;
}

AnalysisSettings &AnalysisSettings::operator=(const AnalysisSettings &rhs) {
    if (&rhs != this) {
        m_values.clear();
        for (const auto &rvalue : rhs.m_values)
            set(rvalue.first, *rvalue.second);
    }

    return *this;
}

AnalysisSettings::Type
AnalysisSettings::type(const std::string &name) const { return (Type) get(name).which(); }

const AnalysisSettings::Variant &AnalysisSettings::get(const std::string &name) const { return *m_values.at(name); }
      AnalysisSettings::Variant &AnalysisSettings::get(const std::string &name)       { return *m_values.at(name); }
template<typename T>
void AnalysisSettings::set(const std::string &name, T val) { m_values[name] = std::shared_ptr<Variant>(new Variant(val)); }

std::vector<std::string> AnalysisSettings::getNames() const {
    std::vector<std::string> keys;
    for (const auto &pair : m_values) {
        keys.push_back(pair.first);
    }
    return keys;
}

void AnalysisSettings::getOptions(po::options_description &opts) {
    po::options_description globalOpt("global"), elementsOpt("elements"),
                            materialOpt("material"),
                            modalAnalysisOpt("modalAnalysis"),
                            simulationOpt("simulation"),
                            weaknessAnalysisOpt("weaknessAnalysis"),
                            privateOpt("private");
    // NOTE: DEFAULTS SPECIFIED HERE
    globalOpt.add_options()
        ("global.solver", po::value<std::string>()->default_value("Eigen"), "Which solver to use")
        ;

    elementsOpt.add_options()
        ("elements.Nx",                    po::value<int>()->default_value(16),                  "Grid columns")
        ("elements.Ny",                    po::value<int>()->default_value(16),                  "Grid rows")
#if DIM==3
        ("elements.Nz",                    po::value<int>()->default_value(16),                  "Grid Slices")
#endif
        ("elements.borderWidth",           po::value<int>()->default_value(0),                   "Grid border width")
        ("elements.quadrature",            po::value<int>()->default_value(UNIFORM_QUADRATURE),  "Type of quadrature")
#if DIM==3
        ("elements.quadraturePoints",      po::value<int>()->default_value(512),                 "Number of quadrature points")
#else
        ("elements.quadraturePoints",      po::value<int>()->default_value(81),                  "Number of quadrature points")
#endif
        ("elements.cellOverlapThreshold",  po::value<double>()->default_value(0.05),             "Overlap threshold above which a cell is an element")
        ("elements.exactFullElements",     po::value<bool>()->default_value(true),               "Used closed formula for integrals over full elements")
        ("elements.antialiasedElements",   po::value<bool>()->default_value(false),              "Treat cut cells as full cells with lower density")
        ;

    // Materials
    materialOpt.add_options()
        ("material.young_modulus",  po::value<double>()->default_value(1.0),  "Youngs modulus")
        ("material.poisson_ratio",  po::value<double>()->default_value(0.0),  "Poisson ratio")
        ("material.density",        po::value<double>()->default_value(1.0),  "Material density")
        ;

    // Simulation
    simulationOpt.add_options()
        ("simulation.useMSBoundary",    po::value<bool>()->default_value(false),  "Get boundary points from marching squares")
        ("simulation.boundarySpacing",  po::value<double>()->default_value(.03),  "Spacing between boundary points (if useMSBoundary is false)")
        ("simulation.blurPointForces",  po::value<bool>()->default_value(true),   "Blur point forces into volume forces")
        ("simulation.kernelRadius",     po::value<double>()->default_value(1.0),  "Blur radius")
        ;


    // Modal analysis
    modalAnalysisOpt.add_options()
        ("modalAnalysis.massMatrixType",   po::value<int>()->default_value(MASS_QUARTER_CELL),  "Type of mass matrix to use")
        ("modalAnalysis.laplacianModes",   po::value<bool>()->default_value(false),             "Use eigenfunctions of the Laplacian as modes")
        ("modalAnalysis.consistentSigns",  po::value<bool>()->default_value(true),              "Prevent randomness in mode signs by always making their max stress compressive")
        ("modalAnalysis.numModes",         po::value<int>()->default_value(10),                 "Number of modes to compute")
        ;

    // Weakness analysis
    weaknessAnalysisOpt.add_options()
        ("weaknessAnalysis.weakRegionsPerMode",      po::value<int>()->default_value(5),        "Number of weak regions to extract permode")
        ("weaknessAnalysis.weaknessCutoff",          po::value<double>()->default_value(0.95),  "Percentile above which regions are considered weak")
        ("weaknessAnalysis.abstrace",                po::value<bool>()->default_value(true),    "Use modal stress signs to choose between optimizing +trace or -trace on a per-element basis")
        ("weaknessAnalysis.plusMinusObjective",      po::value<bool>()->default_value(true),    "Optimize both the + and the - objective")
        ("weaknessAnalysis.totalForceBound",         po::value<double>()->default_value(0.1),   "Total force allowed on the object")
        ("weaknessAnalysis.pointwisePressureBound",  po::value<double>()->default_value(0.1),   "Maximum pointwise pressure allowed (prevent needle poke)")
        ;

    // Private
    privateOpt.add_options()
        ("private.fixedTranslation",  po::value<bool>()->default_value(false),  "Use a fixed translation instead of a sweep for translation test")
        ("private.xTranslation",      po::value<double>()->default_value(0.0),  "Fixed translation x")
        ("private.yTranslation",      po::value<double>()->default_value(0.0),  "Fixed translation y")
        ;

    opts.add(globalOpt).add(elementsOpt).add(materialOpt).
         add(modalAnalysisOpt).add(simulationOpt).add(weaknessAnalysisOpt).
         add(privateOpt);
}

void AnalysisSettings::parseOptions(const std::string &s) {
    std::stringstream ss(s);
    parseOptions(ss);
}

void AnalysisSettings::parseOptions(std::istream &is) {
    po::options_description opts;
    getOptions(opts);

    po::variables_map vm;
    po::store(po::parse_config_file(is, opts), vm);

    readOptions(vm);
}

void AnalysisSettings::readOptions(const po::variables_map &vm) {
    // Be careful with initialization literal types here... they determine
    // the setting's type, and need to match exactly to avoid overload
    // ambiguity.
    set("solver",                 vm["global.solver"].as<std::string>());

    // Elements
    set("Nx",                     vm["elements.Nx"].as<int>());
    set("Ny",                     vm["elements.Ny"].as<int>());
#if DIM==3
    set("Nz",                     vm["elements.Nz"].as<int>());
#endif
    set("borderWidth",            vm["elements.borderWidth"].as<int>());
    set("quadrature",             vm["elements.quadrature"].as<int>());
    set("quadraturePoints",       vm["elements.quadraturePoints"].as<int>());
    set("cellOverlapThreshold",   vm["elements.cellOverlapThreshold"].as<double>());
    set("exactFullElements",      vm["elements.exactFullElements"].as<bool>());
    set("antialiasedElements",    vm["elements.antialiasedElements"].as<bool>());

    // Materials
    set("young_modulus",          vm["material.young_modulus"].as<double>());
    set("poisson_ratio",          vm["material.poisson_ratio"].as<double>());
    set("density",                vm["material.density"].as<double>());

    // Simulation
    set("useMSBoundary",          vm["simulation.useMSBoundary"].as<bool>());
    set("boundarySpacing",        vm["simulation.boundarySpacing"].as<double>());
    set("blurPointForces",        vm["simulation.blurPointForces"].as<bool>());
    set("kernelRadius",           vm["simulation.kernelRadius"].as<double>());


    // Modal analysis
    set("massMatrixType",         vm["modalAnalysis.massMatrixType"].as<int>());
    set("laplacianModes",         vm["modalAnalysis.laplacianModes"].as<bool>());
    set("consistentSigns",        vm["modalAnalysis.consistentSigns"].as<bool>());
    set("numModes",               vm["modalAnalysis.numModes"].as<int>());

    // Weakness analysis
    set("weakRegionsPerMode",     vm["weaknessAnalysis.weakRegionsPerMode"].as<int>());
    set("weaknessCutoff",         vm["weaknessAnalysis.weaknessCutoff"].as<double>());
    set("abstrace",               vm["weaknessAnalysis.abstrace"].as<bool>());
    set("plusMinusObjective",     vm["weaknessAnalysis.plusMinusObjective"].as<bool>());
    set("totalForceBound",        vm["weaknessAnalysis.totalForceBound"].as<double>());
    set("pointwisePressureBound", vm["weaknessAnalysis.pointwisePressureBound"].as<double>());

    // Private
    set("fixedTranslation",       vm["private.fixedTranslation"].as<bool>());
    set("xTranslation",           vm["private.xTranslation"].as<double>());
    set("yTranslation",           vm["private.yTranslation"].as<double>());
}

void AnalysisSettings::writeOptions(std::ostream &os) const {
    // boost::program_options doesn't support writing .ini files, so we
    // convert to a property_tree. Notice that property tree groups are
    // written as groups in the .ini, so we don't need to add the group name
    // to each setting (e.g. we write Nx instead of elements.Nx).
    using boost::property_tree::ptree;
    ptree root;

    ptree globalOpt, elementsOpt, materialOpt, modalAnalysisOpt,
          simulationOpt, weaknessAnalysisOpt, privateOpt;
    
    globalOpt.put("solver", boost::lexical_cast<std::string>(get("solver")));

    // Elements
    elementsOpt.put("Nx",                             boost::lexical_cast<std::string>(get("Nx")));
    elementsOpt.put("Ny",                             boost::lexical_cast<std::string>(get("Ny")));
#if DIM==3
    elementsOpt.put("Nz",                             boost::lexical_cast<std::string>(get("Nz")));
#endif
    elementsOpt.put("borderWidth",                    boost::lexical_cast<std::string>(get("borderWidth")));
    elementsOpt.put("quadrature",                     boost::lexical_cast<std::string>(get("quadrature")));
    elementsOpt.put("quadraturePoints",               boost::lexical_cast<std::string>(get("quadraturePoints")));
    elementsOpt.put("cellOverlapThreshold",           boost::lexical_cast<std::string>(get("cellOverlapThreshold")));
    elementsOpt.put("exactFullElements",              boost::lexical_cast<std::string>(get("exactFullElements")));
    elementsOpt.put("antialiasedElements",            boost::lexical_cast<std::string>(get("antialiasedElements")));

    // Materials
    materialOpt.put("young_modulus",                  boost::lexical_cast<std::string>(get("young_modulus")));
    materialOpt.put("poisson_ratio",                  boost::lexical_cast<std::string>(get("poisson_ratio")));
    materialOpt.put("density",                        boost::lexical_cast<std::string>(get("density")));

    // Simulation
    simulationOpt.put("useMSBoundary",                boost::lexical_cast<std::string>(get("useMSBoundary")));
    simulationOpt.put("boundarySpacing",              boost::lexical_cast<std::string>(get("boundarySpacing")));
    simulationOpt.put("blurPointForces",              boost::lexical_cast<std::string>(get("blurPointForces")));
    simulationOpt.put("kernelRadius",                 boost::lexical_cast<std::string>(get("kernelRadius")));


    // Modal analysis
    modalAnalysisOpt.put("massMatrixType",            boost::lexical_cast<std::string>(get("massMatrixType")));
    modalAnalysisOpt.put("laplacianModes",            boost::lexical_cast<std::string>(get("laplacianModes")));
    modalAnalysisOpt.put("consistentSigns",           boost::lexical_cast<std::string>(get("consistentSigns")));
    modalAnalysisOpt.put("numModes",                  boost::lexical_cast<std::string>(get("numModes")));

    // Weakness analysis
    weaknessAnalysisOpt.put("weakRegionsPerMode",     boost::lexical_cast<std::string>(get("weakRegionsPerMode")));
    weaknessAnalysisOpt.put("weaknessCutoff",         boost::lexical_cast<std::string>(get("weaknessCutoff")));
    weaknessAnalysisOpt.put("abstrace",               boost::lexical_cast<std::string>(get("abstrace")));
    weaknessAnalysisOpt.put("plusMinusObjective",     boost::lexical_cast<std::string>(get("plusMinusObjective")));
    weaknessAnalysisOpt.put("totalForceBound",        boost::lexical_cast<std::string>(get("totalForceBound")));
    weaknessAnalysisOpt.put("pointwisePressureBound", boost::lexical_cast<std::string>(get("pointwisePressureBound")));

    // Private
    privateOpt.put("fixedTranslation",                boost::lexical_cast<std::string>(get("fixedTranslation")));
    privateOpt.put("xTranslation",                    boost::lexical_cast<std::string>(get("xTranslation")));
    privateOpt.put("yTranslation",                    boost::lexical_cast<std::string>(get("yTranslation")));

    root.push_back(ptree::value_type("global",            globalOpt));
    root.push_back(ptree::value_type("elements",          elementsOpt));
    root.push_back(ptree::value_type("material",          materialOpt));
    root.push_back(ptree::value_type("modalAnalysis",     modalAnalysisOpt));
    root.push_back(ptree::value_type("simulation",        simulationOpt));
    root.push_back(ptree::value_type("weaknessAnalysis",  weaknessAnalysisOpt));
    root.push_back(ptree::value_type("private",           privateOpt));

    write_ini(os, root);
}
