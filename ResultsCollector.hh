////////////////////////////////////////////////////////////////////////////////
// ResultsCollector.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Collects CSGFEM simulation/weakness analysis results for different
//        models/settings. The collection assumes ownership of all results
//        stored within it.
//
//        Results are identified using a colon-separated "path" of the format:
//              model_name:settings_name:result_name
//        where colons in result_name allow results to be organized in
//        "folders."
//
//        Also stores copies of the distinct model and settings used to
//        generate the results. Only those models and settings that actually
//        have results attached are kept (though the selected model is never
//        removed because results may be added to it).
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/15/2013 16:51:03
////////////////////////////////////////////////////////////////////////////////
#ifndef RESULTS_COLLECTOR_HH
#define RESULTS_COLLECTOR_HH

#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <boost/algorithm/string.hpp>
#include "utils.hh"
#include "AnalysisSettings.hh"

////////////////////////////////////////////////////////////////////////////////
// Path operations
////////////////////////////////////////////////////////////////////////////////
inline std::string getModelPathComponent(const std::string &path) {
    std::vector<std::string> nameComponents;
    boost::split(nameComponents, path, boost::is_any_of(":"));
    assert(nameComponents.size() > 1);
    return nameComponents[0];
}

inline std::string getSettingsPathComponent(const std::string &path) {
    std::vector<std::string> nameComponents;
    boost::split(nameComponents, path, boost::is_any_of(":"));
    assert(nameComponents.size() > 1);
    return nameComponents[1];
}

template<typename Generator>
class ResultsCollector {
    typedef typename Generator::Model Model;
    typedef typename std::pair<Model, BBox_t> RModel;
public:
    class ResultTree;
    class Result;

    typedef enum { KEY_ORDER_MODEL_SETTINGS, KEY_ORDER_SETTINGS_MODEL } KeyOrder;

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

    bool modelDiffers(const std::string &name, const Model &model,
                      const BBox_t &gridBBox) const {
        auto model_it = m_models.find(name);
        if (model_it == m_models.end())
            throw std::runtime_error(std::string("model not found: ") + name);
        return !((model == model_it->second.first) &&
                 (gridBBox == model_it->second.second));
    }

    void getModel(const std::string &name, Model &model,
                  BBox_t &gridBBox) const {
        auto model_it = m_models.find(name);
        if (model_it == m_models.end())
            throw std::runtime_error(std::string("model not found: ") + name);
        model = model_it->second.first;
        gridBBox = model_it->second.second;
    }

    void getSettings(const std::string &name, AnalysisSettings &settings) const
    {
        auto settings_it = m_settings.find(name);
        if (settings_it == m_settings.end())
            throw std::runtime_error(std::string("settings not found: ") + name);
        settings = settings_it->second;
    }

    bool settingsDiffer(const std::string &name,
                        const AnalysisSettings &settings) const {
        auto settings_it = m_settings.find(name);
        if (settings_it == m_settings.end())
            throw std::runtime_error(std::string("settings not found: ") + name);
        return !(settings == settings_it->second);
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

    // Path always has the format model_name:settings_name:name
    // (Despite the fact DFS can visit in settings:model order).
    std::shared_ptr<const Result>
    getResultWithPath(const std::string &path) const {
        std::vector<std::string> nameComponents;
        boost::split(nameComponents, path, boost::is_any_of(":"));
        for (std::string &str : nameComponents)
            boost::trim(str);
        if (nameComponents.size() < 3) {
            throw std::runtime_error(std::string("Invalid path: ") + path);
        }
        auto name_it = nameComponents.begin();
        std::string model    = *(name_it++);
        std::string settings = *(name_it++);

        auto mit = m_models_settings_collection.find(model);
        if (mit == m_models_settings_collection.end())
            throw std::runtime_error(std::string("collection not found: " +
                        model + ":" + settings));
        auto sit = mit->second.find(settings);
        if (sit == mit->second.end())
            throw std::runtime_error(std::string("collection not found: " +
                        model + ":" + settings));
        return sit->second->getResult(name_it, nameComponents.end());
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Get a result from the the collection (for the currently selected model
    //  and settings).
    //  @param[in]  name    result's name in the collection
    //  @return     result pointer
    *///////////////////////////////////////////////////////////////////////////
    std::shared_ptr<const Result> getResult(const std::string &name) const {
        return getResultWithPath(m_selectedModel + ":" + m_selectedSettings +
                                 ":" + name);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the path of the last result inserted.
    //  @return     string holding the path of the last result inserted
    *///////////////////////////////////////////////////////////////////////////
    std::string lastResultPath() const {
        return m_lastResult;
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
        _RTPtr &msc_entry = m_models_settings_collection[m_selectedModel][m_selectedSettings];
        _RTPtr &smc_entry = m_settings_models_collection[m_selectedSettings][m_selectedModel];

        // Both collections better hold the same pointer (whether NULL or
        // existing)
        assert(msc_entry == smc_entry);
        if (msc_entry == nullptr) {
            msc_entry = smc_entry = _RTPtr(new ResultTree());
        }

        msc_entry->setResult(name, result);

        m_lastResult = m_selectedModel + ":" + m_selectedSettings + ":" + name;
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Delete all models/settings for which no results are recorded
    //  (except the currently selected pair)
    *///////////////////////////////////////////////////////////////////////////
    void clean();

    ////////////////////////////////////////////////////////////////////////////
    /*! Remove all the results specified in paths.
    //  @param[in]  paths   paths of results to remove
    *///////////////////////////////////////////////////////////////////////////
    void removeResultsWithPaths(std::vector<std::string> paths) {
        m_lastResult.clear();

        // ASCII sort of paths in descending order gives us a topological sort
        // (So we never try to delete a child after its parent).
        std::sort(paths.begin(), paths.end(), std::greater<std::string>());

        for (const std::string &path : paths) {
            std::vector<std::string> nameComponents;
            boost::split(nameComponents, path, boost::is_any_of(":"));
            for (std::string &str : nameComponents)
                boost::trim(str);

            std::runtime_error e_invalid(std::string("Removal path invalid!"));
            const std::string &mname = nameComponents[0];

            auto mscit = m_models_settings_collection.find(mname);
            if (mscit == m_models_settings_collection.end())
                throw e_invalid;

            _InnerMapType &imap = mscit->second;
            if (nameComponents.size() > 1) {
                const std::string &sname = nameComponents[1];
                auto scit = imap.find(sname);
                if (scit == imap.end())
                    throw e_invalid;
                else {
                    if (nameComponents.size() > 2) {
                        // Remove a subtree of the (model, setting) results tree
                        scit->second->remove(nameComponents.begin() + 2,
                                             nameComponents.end());
                    }
                    else {
                        // Delete the results tree for a (model, setting)
                        imap.erase(scit);

                        // Delete the associated results tree in the
                        // settings->model collections (it must exist...)
                        auto &mc = m_settings_models_collection.at(sname);
                        auto mcit = mc.find(mname);
                        assert(mcit != mc.end());
                        mc.erase(mcit);
                    }
                }
            }
            else {
                // Deleting all result trees for a model
                m_models_settings_collection.erase(mscit);

                // Delete all the associated result trees in the settings->model
                // collections
                for (auto &smc_val : m_settings_models_collection) {
                    auto mcit = smc_val.second.find(mname);
                    if (mcit != smc_val.second.end())
                        smc_val.second.erase(mcit);
                }
            }
        }

        // We might want to remove models and settings now that results have
        // been removed.
        clean();
    }

    // Visit each node of the result collection tree. Visitor's preVisit takes
    // two arguments:
    //  1) Node name (std::string)
    //  2) whether this node holds a result (bool)
    template<typename Visitor>
    void dfs(KeyOrder order, Visitor &v) {
        auto &topLevel = (order == KEY_ORDER_MODEL_SETTINGS) ?
                        m_models_settings_collection :
                        m_settings_models_collection;

        for (auto &e1 : topLevel) {
            v.preVisit(e1.first, false);
            _InnerMapType &scollection = e1.second;
            for (auto &e2 : scollection) {
                v.preVisit(e2.first, false);
                _RTPtr t = e2.second;
                t->dfs(v);
                v.postVisit();
            }
            v.postVisit();
        }
    }

    void clear() {
        // Smart pointers now make clean-up easy!
        m_models.clear();
        m_settings.clear();
        m_models_settings_collection.clear();
        m_settings_models_collection.clear();
        m_lastResult.clear();
    }

    void print() const {
        for (const auto &entry : m_models_settings_collection) {
            const _InnerMapType &scollection = entry.second;
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
    typedef std::shared_ptr<ResultTree>       _RTPtr;
    typedef std::shared_ptr<const ResultTree> _ConstRTPtr;
    typedef std::map<std::string, _RTPtr, NaturalLess> _InnerMapType;
    typedef std::map<std::string, _InnerMapType>       _OuterMapType;
    std::map<std::string, RModel> m_models;
    std::map<std::string, AnalysisSettings> m_settings;
    _OuterMapType m_models_settings_collection, m_settings_models_collection;

    std::string m_selectedModel, m_selectedSettings;
    std::string m_lastResult; // The last result added (model:settings:name)
};

#include "ResultsCollector.inl"

#endif // RESULTS_COLLECTOR_HH
