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
#include <string>
#include <cassert>
#include <iosfwd>
#include <map>
#include <memory>

#include "GlobalTypes.hh"
#include "Quadrature.hh"
#include "utils.hh"

#include <boost/variant/variant_fwd.hpp>

// Forward declare options_description for compile speed.
namespace boost { namespace program_options {
    class options_description;
} }


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
    bool          Bool(const std::string &name) const;
    int            Int(const std::string &name) const;
    int           Enum(const std::string &name) const;
    double        Real(const std::string &name) const;
    const std::string &String(const std::string &name) const;

    bool        &   Bool(const std::string &name);
    int         &    Int(const std::string &name);
    int         &   Enum(const std::string &name);
    double      &   Real(const std::string &name);
    std::string & String(const std::string &name);

    std::string displayString(const std::string &name) const;

    // Memberwise comparator
    bool operator==(const AnalysisSettings &rhs) const;

    Type type(const std::string &name) const;

    const Variant &get(const std::string &name) const;
          Variant &get(const std::string &name);
    template<typename T>
    void set(const std::string &name, T val);

    std::vector<std::string> getNames() const;

    static void getOptions(boost::program_options::options_description &opts);

    void parseOptions(const std::string &s = std::string());

    void parseOptions(std::istream &is);

    void writeOptions(std::ostream &os) const;

private:
    std::map<std::string, std::shared_ptr<Variant> > m_values;
};

#endif // ANALYSIS_SETTINGS_HH
