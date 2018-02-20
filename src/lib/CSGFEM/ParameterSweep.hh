////////////////////////////////////////////////////////////////////////////////
// ParameterSweep.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements an sweep of settings or CSG tree parameters in one of two ways:
//
//      1) Zip
//         All parameter ranges are stepped through simultaneously. For this
//         mode, all ranges must be of equal length.
//      2) Product
//         All combinations of parameter values are tried.
//
//  For settings only, "dependent" parameters are supported. These are specified
//  by passing a range string that names an independent parameter (a parameter
//  whose range string directly specifies a MATLAB-style sequence). In the
//  future, arithmetic expressions on independent + dependent parameters may be
//  supported.
//
//  To perform the sweep, call getSettingValues and getCSGParameterValues
//  to read the current parameter assignments and advance() to move to the next
//  ones. advance() will return false when the sweep is done.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/17/2014 12:11:44
////////////////////////////////////////////////////////////////////////////////
#ifndef PARAMETERSWEEP_HH
#define PARAMETERSWEEP_HH

#include <vector>
#include <string>
#include <utility>
#include <iterator>
#include <cassert>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <regex>

#include "utils.hh"

template<typename Real>
class ParameterSweep {
public:
    typedef enum { SWEEP_ZIP = 0, SWEEP_PRODUCT = 1 } SweepMode;

    // Settings referred to by name, CSG parameters by index.
    ParameterSweep(SweepMode mode,
                   const std::vector<std::string> &settingParameters,
                   const std::vector<std::string> &settingsParameterRanges,
                   const std::vector<size_t>      &csgParameters,
                   const std::vector<std::string> &csgParameterRanges)
        : m_mode(mode), m_settingParameters(settingParameters),
          m_csgParameters(csgParameters)
    {
        assert(settingParameters.size() == settingsParameterRanges.size());
        assert(csgParameters.size() == csgParameterRanges.size());

        m_indepParameterID.assign(settingParameters.size(), -1);
        m_settingValues.reserve(settingParameters.size());

        // Store the name of each dependent parameter's dependency
        std::vector<int> dependencies(settingParameters.size(), -1);

        size_t minSize = std::numeric_limits<size_t>::max(), maxSize = 0;
        for (size_t i = 0; i < settingParameters.size(); ++i) {
            // Check if the setting parameter is dependent. This occurs when the
            // range string is another setting's name.
            std::regex wordPattern("\\s*([a-zA-Z_]\\S*)");
            std::smatch match;
            bool isDependent = std::regex_search(settingsParameterRanges[i],
                                                 match, wordPattern);

            std::string setting = settingParameters[i];

            if (isDependent) {
                auto it = find(settingParameters.begin(),
                               settingParameters.end(), match[1].str());
                if (it == settingParameters.end()) {
                    throw std::runtime_error(std::string("Unknown dependency: ")
                                + match[1].str());
                }
                dependencies[i] = std::distance(settingParameters.begin(), it);
            }
            else {
                m_indepParameterID[i] = m_settingValues.size();
                m_settingValues.push_back(expandRange<Real>(
                            settingsParameterRanges[i]));
                minSize = std::min(minSize, m_settingValues.back().size());
                maxSize = std::max(maxSize, m_settingValues.back().size());
            }
        }

        // Resolve dependencies (in the future this could be smarter to allow
        // nontrivial dependency chains).
        for (size_t i = 0; i < dependencies.size(); ++i) {
            int dep = dependencies[i];
            if (dep >= 0) {
                assert(m_indepParameterID[i] == -1);
                int resolvedID = m_indepParameterID[dep];
                if (resolvedID < 0) {
                    throw std::runtime_error(std::string("Dependent dependency ")
                            + settingParameters[dep]);
                }
                m_indepParameterID[i] = resolvedID;
            }
        }

        m_csgValues.reserve(csgParameters.size());
        for (size_t i = 0; i < csgParameters.size(); ++i) {
            m_csgValues.push_back(expandRange<Real>(csgParameterRanges[i]));
            minSize = std::min(minSize, m_csgValues[i].size());
            maxSize = std::max(maxSize, m_csgValues[i].size());
        }
        
        if (minSize == 0)
            throw std::runtime_error(std::string("Ranges must be nonempty."));
        if ((mode == SWEEP_ZIP) && (minSize != maxSize))
            throw std::runtime_error(std::string("Ranges must be equi-sized."));

        reset();
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Advance to the next set of values.
    //  @return     false if we have already reached the end and wrapped around.
    *///////////////////////////////////////////////////////////////////////////
    bool advance() {
        bool advanced = false;
        if (m_mode == SWEEP_ZIP) {
            size_t minIdx = std::numeric_limits<size_t>::max(), maxIdx = 0;
            for (size_t i = 0; i < m_numIndepSettings(); ++i) {
                if (m_settingsCounters[i] + 1 < m_numSettingValues(i)) {
                    ++m_settingsCounters[i];
                    advanced = true;
                }
                minIdx = std::min(minIdx, m_settingsCounters[i]);
                maxIdx = std::max(maxIdx, m_settingsCounters[i]);
            }
            for (size_t i = 0; i < numCSGParameters(); ++i) {
                if (m_csgCounters[i] + 1 < m_numCSGParameterValues(i)) {
                    ++m_csgCounters[i];
                    advanced = true;
                }
                minIdx = std::min(minIdx, m_csgCounters[i]);
                maxIdx = std::max(maxIdx, m_csgCounters[i]);
            }

            assert(minIdx == maxIdx);
        }
        else if (m_mode == SWEEP_PRODUCT) {
            // rev(m_settingsCounters : m_csgCounters) conceptually makes up a
            // mixed-base number that we increment to advance...
            bool carry = true;
            for (size_t i = 0; carry && (i < m_numIndepSettings()); ++i) {
                if (m_settingsCounters[i] + 1 < m_numSettingValues(i)) {
                    ++m_settingsCounters[i];
                    carry = false;
                }
                else
                    m_settingsCounters[i] = 0;
            }
            for (size_t i = 0; carry && (i < numCSGParameters()); ++i) {
                if (m_csgCounters[i] + 1 < m_numCSGParameterValues(i)) {
                    ++m_csgCounters[i];
                    carry = false;
                }
                else
                    m_csgCounters[i] = 0;
            }

            advanced = !carry;
        }
        else {
            assert(false);
        }

        if (!advanced)
            reset();
        
        return advanced;
    }

    void reset() {
        m_settingsCounters.assign(m_numIndepSettings(), 0);
        m_csgCounters.assign(numCSGParameters(), 0);
    }

    std::vector<Real> getSettingValues() const {
        std::vector<Real> values;
        values.reserve(numSettings());
        for (size_t i = 0; i < numSettings(); ++i) {
            size_t ii = m_indepParameterID[i];
            assert(ii < m_settingsCounters.size());
            size_t vi = m_settingsCounters[ii];
            assert(vi < m_numSettingValues(ii));
            values.push_back(m_settingValues[ii][vi]);
        }

        return values;
    }

    std::vector<Real> getCSGParameterValues() const {
        std::vector<Real> values(numCSGParameters());
        for (size_t i = 0; i < numCSGParameters(); ++i) {
            size_t vi = m_csgCounters[i];
            assert(vi < m_numCSGParameterValues(i));
            values[i] = m_csgValues[i][vi];
        }
        return values;
    }

    std::vector<std::string> getSettingNames() const { return m_settingParameters; }
    std::vector<size_t> getCSGParameterIndices() const { return m_csgParameters; }

    size_t numSettings()      const { return m_settingParameters.size(); }
    size_t numCSGParameters() const { return m_csgParameters.size(); }

private:
    size_t m_numIndepSettings() const { return m_settingValues.size(); }
    // Get the range size for independent setting i
    size_t m_numSettingValues(size_t i) const {
        assert(i < m_numIndepSettings());
        return m_settingValues[i].size();
    }

    size_t  m_numCSGParameterValues(size_t i) const {
        assert(i < numCSGParameters());
        return m_csgValues[i].size();
    }

    SweepMode m_mode;
    std::vector<int>                m_indepParameterID;;
    std::vector<std::string>        m_settingParameters;
    std::vector<std::vector<Real> > m_settingValues;
    std::vector<size_t>             m_csgParameters;
    std::vector<std::vector<Real> > m_csgValues;

    // List of (Dependent setting name, dependency index)
    std::vector<std::string>        m_dependentSettings;
    std::vector<size_t>             m_dependencies;

    // Indices into the possible values of settings/csg parameters (these hold
    // the enumeration state).
    std::vector<size_t> m_settingsCounters;
    std::vector<size_t> m_csgCounters;
};

#endif /* end of include guard: PARAMETERSWEEP_HH */
