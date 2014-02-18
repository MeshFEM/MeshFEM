////////////////////////////////////////////////////////////////////////////////
// ParameterSweep.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Implements an automatic sweep of settings or CSG tree parameters.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/17/2014 12:11:44
////////////////////////////////////////////////////////////////////////////////
#ifndef PARAMETERSWEEP_HH
#define PARAMETERSWEEP_HH

#include <vector>
#include <string>
#include <cassert>
#include <limits>
#include <algorithm>
#include <stdexcept>

#include "utils.hh"

template<typename Real>
class ParameterSweep {
public:
    typedef enum { SWEEP_ZIP = 0, SWEEP_PRODUCT = 1 } SweepMode;

    // Settings referred to by name, CSG parameters by index.
    ParameterSweep(SweepMode mode,
                   const std::vector<std::string> &settingsParameters,
                   const std::vector<std::string> &settingsParameterRanges,
                   const std::vector<size_t>      &csgParameters,
                   const std::vector<std::string> &csgParameterRanges)
        : m_mode(mode), m_settingsParameters(settingsParameters),
          m_csgParameters(csgParameters)
    {
        assert(settingsParameters.size() == settingsParameterRanges.size());
        assert(csgParameters.size() == csgParameterRanges.size());

        m_settingValues.reserve(settingsParameters.size());
        size_t minSize = std::numeric_limits<size_t>::max(), maxSize = 0;
        for (size_t i = 0; i < settingsParameters.size(); ++i) {
            m_settingValues.push_back(expandRange<Real>(settingsParameterRanges[i]));
            minSize = std::min(minSize, m_settingValues[i].size());
            maxSize = std::max(maxSize, m_settingValues[i].size());
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
            for (size_t i = 0; i < numSettings(); ++i) {
                if (m_settingsCounters[i] + 1 < numSettingValues(i)) {
                    ++m_settingsCounters[i];
                    advanced = true;
                }
                minIdx = std::min(minIdx, m_settingsCounters[i]);
                maxIdx = std::max(maxIdx, m_settingsCounters[i]);
            }
            for (size_t i = 0; i < numCSGParameters(); ++i) {
                if (m_csgCounters[i] + 1 < numCSGParameterValues(i)) {
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
            for (size_t i = 0; carry && (i < numSettings()); ++i) {
                if (m_settingsCounters[i] + 1 < numSettingValues(i)) {
                    ++m_settingsCounters[i];
                    carry = false;
                }
                else
                    m_settingsCounters[i] = 0;
            }
            for (size_t i = 0; carry && (i < numCSGParameters()); ++i) {
                if (m_csgCounters[i] + 1 < numCSGParameterValues(i)) {
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
        m_settingsCounters.assign(numSettings(), 0);
        m_csgCounters.assign(numCSGParameters(), 0);
    }

    std::vector<Real> getSettingsValues() const {
        std::vector<Real> values(numSettings());
        for (size_t i = 0; i < numSettings(); ++i) {
            size_t vi = m_settingsCounters[i];
            assert(vi < numSettingValues(i));
            values[i] = m_settingValues[i][vi];
        }
        return values;
    }

    std::vector<Real> getCSGParameterValues() const {
        std::vector<Real> values(numCSGParameters());
        for (size_t i = 0; i < numCSGParameters(); ++i) {
            size_t vi = m_csgCounters[i];
            assert(vi < numCSGParameterValues(i));
            values[i] = m_csgValues[i][vi];
        }
        return values;
    }

    std::vector<std::string> getSettingsNames() const { return m_settingsParameters; }
    std::vector<size_t> getCSGParameterIndices() const { return m_csgParameters; }

    size_t numSettings() const { return m_settingsParameters.size(); }
    size_t numCSGParameters() const { return m_csgParameters.size(); }
    size_t  numSettingValues(size_t i) const {
        assert(i < numSettings());
        return m_settingValues[i].size();
    }

    size_t  numCSGParameterValues(size_t i) const {
        assert(i < numCSGParameters());
        return m_csgValues[i].size();
    }

private:
    SweepMode m_mode;
    std::vector<std::string>        m_settingsParameters;
    std::vector<std::vector<Real> > m_settingValues;
    std::vector<size_t>             m_csgParameters;
    std::vector<std::vector<Real> > m_csgValues;

    std::vector<size_t> m_settingsCounters;
    std::vector<size_t> m_csgCounters;
};

#endif /* end of include guard: PARAMETERSWEEP_HH */
