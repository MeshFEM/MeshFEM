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
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include "AnalysisSettings.hh"


template<typename Generator>
class ResultsCollector {
    typedef typename Generator::Model Model;
    typedef typename std::pair<Model, BBox_t> RModel;
public:
    class ResultTree;
    class Result;

    // Note: the "model" a ResultsCollector stores includes the computation grid
    // bounding box as well as the model itself (storing bounding boxes is
    // needed for translation tests, for instance). These together represent the
    // full simulated geometry.
    std::string addModel(const std::string &nameSuggestion, const Model &model,
                         const BBox_t &gridBBox);
    std::string addSettings(const std::string &nameSuggestion,
                            const AnalysisSettings &settings);

    void selectModel(const std::string &name) {
        if (m_models.find(name) == m_models.end())
            throw std::runtime_error(std::string("model not found: ") + name);
        m_selectedModel = name;
    }

    void selectSettings(const std::string &name) {
        if (m_settings.find(name) == m_settings.end())
            throw std::runtime_error(std::string("settings not found: ") + name);
        m_selectedSettings = name;
    }

    void getModel(const std::string &name, Model &model,
                  BBox_t &gridBBox) const {
        auto model_it = m_models.find(name);
        if (model_it == m_models.end())
            throw std::runtime_error(std::string("model not found: ") + name);
        model = model_it->second.first;
        gridBBox = model_it->second.second;
    }

    // Gets a reference to the currently selected model
    void getModel(Model &model, BBox_t &gridBBox) const {
        getModel(m_selectedModel, model, gridBBox);
    }

    // Checks if a model differs from the currently selected model.
    bool modelIsDifferent(const Model &m, const BBox_t &b) const {
        if (m_selectedModel.size() == 0)
            return true;
        else {
            auto mit = m_models.find(m_selectedModel);
            if (mit == m_models.end())
                return true;
            return (mit->second.first == m) && (mit->second.second == b);
        }
    }

    // Path consists of model_name:settings_name:name
    const Result *getResultWithPath(const std::string &path) const {
        std::vector<std::string> nameComponents;
        boost::split(nameComponents, path, boost::is_any_of(":"));
        for (std::string &str : nameComponents)
            boost::trim(str);
        if (nameComponents.size() < 3) {
            throw std::runtime_error(std::string("Invalid path: ") + path);
        }
        auto it = nameComponents.begin();
        std::string model    = *(it++);
        std::string settings = *(it++);

        auto mit = m_models.find(model);
        if (mit == m_models.end())
            throw std::runtime_error(std::string("collection not found: " +
                        model + ":" + settings));
        auto sit = mit->second.find(settings);
        if (sit == m_settings.end())
            throw std::runtime_error(std::string("collection not found: " +
                        model + ":" + settings));
        return sit->second.getResult(it, nameComponents.end());
    }


    ////////////////////////////////////////////////////////////////////////////
    /*! Get a result from the the collection (for the currently selected model
    //  and settings).
    //  @param[in]  name    result's name in the collection
    //  @return     result pointer
    *///////////////////////////////////////////////////////////////////////////
    const Result *getResult(const std::string &name) const {
        return getResultWithPath(m_selectedModel + ":" + m_selectedSettings +
                                 ":" + name);
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
            for (auto &e2 : scollection) {
                ResultTree *t = e2.second;
                delete t;
            }
        }
    }

    void print() const {
        for (const auto &entry : m_models_settings_collection) {
            std::map<std::string, ResultTree *> &scollection = entry.second;
            std::cout << entry.first << std::endl;
            for (const auto &e2 : scollection) {
                std::cout << "    " << e2.first << std::endl;
                e2.second->print(2);
            }
        }
    }

    ~ResultsCollector() {
        clear();
    }

private:
    std::map<std::string, RModel> m_models;
    std::map<std::string, AnalysisSettings> m_settings;
    std::map<std::string, std::map<std::string, ResultTree *> > 
                m_models_settings_collection;
    std::map<std::string, std::map<std::string, ResultTree *> >
                m_settings_models_collection;

    std::string m_selectedModel, m_selectedSettings;
    std::string m_last_result; // The last result added (model:settings:name)
};

#include "ResultsCollector.inl"

#endif // RESULTS_COLLECTOR_HH
