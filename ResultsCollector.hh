////////////////////////////////////////////////////////////////////////////////
// ResultsCollector.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Collects CSGFEM simulation/weakness analysis results for different
//        models/settings.
//
//        Note: stores copies of the distinct model and settings used to
//        generate the results.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/15/2013 16:51:03
////////////////////////////////////////////////////////////////////////////////
#ifndef RESULTS_COLLECTOR_HH
#define RESULTS_COLLECTOR_HH

#include <map>
#include <string>
#include <cassert>
#include "AnalysisSettings.hh"


template<typename Generator>
class ResultsCollector {
    typedef typename Generator::Model Model;
public:
    class ResultTree;
    class Result;

    std::string addModel(const std::string &nameSuggestion, const Model &model);
    std::string addSettings(const std::string &nameSuggestion,
                            const AnalysisSettings &settings);

    bool selectModel(const std::string &name) {
        if (m_models.find(name) != m_models.end())
            return false;
        m_selectedModel = name;
        return true;
    }

    bool selectSettings(const std::string &name) {
        if (m_settings.find(name) != m_settings.end())
            return false;
        m_selectedSettings = name;
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Add/update a result to the collection (for the currently selected model
    //  and settings). The name given to the result encodes its position in a
    //  tree (each colon denotes an edge). Any preexisting result of the same
    //  name is overwritten.
    //  Note: ResultsCollector assumes ownership of the result object
    //  @param[in]  name    result's name in the collection
    //  @param[in]  result  object holding result data.
    *///////////////////////////////////////////////////////////////////////////
    void setResult(const std::string &name, Result *result) {
        assert(m_models.find(m_selectedModel) != m_models.end());
        assert(m_settings.find(m_selectedSettings) != m_settings.end());
        // Note: pointers are default-initialized to NULL
        ResultTree *&msc_entry = m_models_settings_collection[m_selectedModel][m_selectedSettings];
        ResultTree *&smc_entry = m_settings_models_collection[m_selectedSettings][m_selectedModel];

        // Both collections better hold the same pointer (whether NULL or
        // existing)
        assert(msc_entry == smc_entry);
        if (msc_entry == NULL) {
            msc_entry = smc_entry = new ResultsCollector();
        }

        msc_entry->setResult(name, result);
    }

    void clear() {
        // Destroy this collection's dynamically allocated contents
        for (auto &entry : m_models_settings_collection) {
            std::map<std::string, ResultTree *> &scollection = entry.second;
            for (auto e2 : scollection) {
                ResultTree *t = e2.second;
                delete t;
            }
        }
    }

    ~ResultsCollector() {
        clear();
    }

private:
    std::map<std::string, Model> m_models;
    std::map<std::string, AnalysisSettings> m_settings;
    std::map<std::string, std::map<std::string, ResultTree *> > 
                m_models_settings_collection;
    std::map<std::string, std::map<std::string, ResultTree *> >
                m_settings_models_collection;

    std::string m_selectedModel, m_selectedSettings;
};

#include "ResultsCollector.inl"

#endif // RESULTS_COLLECTOR_HH
