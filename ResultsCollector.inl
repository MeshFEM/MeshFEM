////////////////////////////////////////////////////////////////////////////////
// ResultsCollector.inl
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
#include "utils.hh"
#include <string>
#include <vector>
#include <list>
#include <map>
#include <cassert>
#include <memory>
#include <iostream>

// A single result can consist of up to a scalar AND vector field per:
//      node
//      element
//      boundary point
// I.e., there can be up to 6 fields in a results object.
template<typename Generator>
class ResultsCollector<Generator>::Result
{
public:
    typedef enum { PER_NODE = 0, PER_ELEM = 1,
                   PER_BDRY = 2, NUM_DOMAINS } ResultDomain;

    typedef enum {
        NODE_SCALAR = (1 << 0), ELEM_SCALAR = (1 << 1), BDRY_SCALAR = (1 << 2),
        NODE_VECTOR = (1 << 3), ELEM_VECTOR = (1 << 4), BDRY_VECTOR = (1 << 5)
    } ResultType;

    typedef enum {
        NODE_VECTOR_ELEM_SCALAR = NODE_VECTOR | ELEM_SCALAR,
    } ResultCompoundType;

    typedef typename Generator::SField SField;
    typedef typename Generator::VField VField;

    Result() { init(); }
    Result(ResultDomain stype, const SField &sfield) {
        init(); setScalarField(stype, sfield);
    }

    Result(ResultDomain vtype, const VField &vfield) {
        init(); setVectorField(vtype, vfield);
    }

    Result(ResultDomain stype, const SField &sfield,
           ResultDomain vtype, const VField &vfield) {
        init(); setScalarField(stype, sfield);
                setVectorField(vtype, vfield);
    }

    void setVectorField(ResultDomain type, const VField &vfield) {
        assert(type < NUM_DOMAINS);
        m_vfields[type] = vfield;
        m_hasVField[type] = true;
    }

    void setScalarField(ResultDomain type, const SField &sfield) {
        assert(type < NUM_DOMAINS);
        m_sfields[type] = sfield;
        m_hasSField[type] = true;
    }

    const VField &getVectorField(ResultDomain type) const {
        return m_vfields[type];
    }

    const SField &getScalarField(ResultDomain type) const {
        return m_sfields[type];
    }

    typename SField::value_type getMaxScalar(ResultDomain type) const {
        return m_sfields[type].max();
    }

    typename SField::value_type getMinScalar(ResultDomain type) const {
        return m_sfields[type].min();
    }

    unsigned char resultType() const {
    return ((m_hasSField[PER_NODE] ? NODE_SCALAR : 0) |
            (m_hasSField[PER_ELEM] ? ELEM_SCALAR : 0) |
            (m_hasSField[PER_BDRY] ? BDRY_SCALAR : 0) |
            (m_hasVField[PER_NODE] ? NODE_VECTOR : 0) |
            (m_hasVField[PER_ELEM] ? ELEM_VECTOR : 0) |
            (m_hasVField[PER_BDRY] ? BDRY_VECTOR : 0));
    }

    bool hasElemSField() const { return m_hasSField[PER_ELEM]; }
    bool hasNodeVField() const { return m_hasVField[PER_NODE]; }
    bool hasBdrySField() const { return m_hasSField[PER_BDRY]; }
    bool hasBdryVField() const { return m_hasVField[PER_BDRY]; }

    // Dump result data to raw ASCII text file(s) at
    //      basename.extension
    // where extension is determined by the field type (esfield, nvfield, etc.)
    void dump(const std::string &basename) const {
        static const char domainLabels[NUM_DOMAINS] = { 'n', 'e', 'b' };
        std::string path;
        for (int i = 0; i < NUM_DOMAINS; ++i) {
            if (m_hasSField[i]) {
                path = basename + "." + domainLabels[i] + std::string("sfield");
                m_sfields[i].dump(path);
            }
            if (m_hasVField[i]) {
                path = basename + "." + domainLabels[i] + std::string("vfield");
                m_vfields[i].dump(path);
            }
        }
    }

private:
    void init() {
        m_sfields.assign((size_t) NUM_DOMAINS, SField());
        m_vfields.assign((size_t) NUM_DOMAINS, VField());
        m_hasSField.assign((size_t) NUM_DOMAINS, false);
        m_hasVField.assign((size_t) NUM_DOMAINS, false);
    }

    std::vector<bool> m_hasSField, m_hasVField;
    std::vector<SField> m_sfields;
    std::vector<VField> m_vfields;
};

template<typename Generator>
class ResultsCollector<Generator>::ResultTree
{
public:
    ResultTree()
        : m_result(NULL) { }

    void setResult(const std::string &name, Result *result) {
        std::vector<std::string> nameComponents;
        boost::split(nameComponents, name, boost::is_any_of(":"));
        for (std::string &str : nameComponents)
            boost::trim(str);
        setResult(nameComponents.begin(), nameComponents.end(), result);
    }

    std::shared_ptr<const Result> getResult(const std::string &name) const {
        std::vector<std::string> nameComponents;
        boost::split(nameComponents, name, boost::is_any_of(":"));
        for (std::string &str : nameComponents)
            boost::trim(str);
        return getResult(nameComponents.begin(), nameComponents.end());
    }

    // Set a result, releasing any existing one and assuming ownership.
    void setResult(Result *r) {
        m_result = std::shared_ptr<Result>(r);
    }

    // Get the result stored at this node.
    std::shared_ptr<const Result> getResult() const {
        if (m_result == nullptr)
            throw std::runtime_error(std::string("result not found!"));
        return m_result;
    }

    bool hasResult() const {
        return (m_result != nullptr);
    }

    // Get the count of all results in this collection (recursive)
    size_t numResults() const {
        size_t count = 0;
        if (hasResult())
            count = 1;
            
        for (const typename _ChildMap::value_type &c : m_children) {
            count += c.second->numResults();
        }

        return count;
    }

    // Remove all child subtrees with no results, and report if this subtree
    // should be removed.
    bool prune() {
        if (hasResult()) {
            return false;
        }
        
        bool shouldPruneRoot = true;
        for (auto ci = m_children.begin(); ci != m_children.end(); /* nop */) {
            if (ci->second->prune()) {
                ci = m_children.erase(ci);
            }
            else {
                shouldPruneRoot = false;
                ++ci;
            }
        }

        return shouldPruneRoot;
    }

    // Recursively destroy this tree's contents.
    // Made super easy now by smart pointers
    void clear() {
        m_result.reset();
        m_children.clear();
    }

    
    template<typename Visitor>
    void dfs(Visitor &v) {
        for (typename _ChildMap::value_type &c : m_children) {
            v.preVisit(c.first, c.second->hasResult());
            c.second->dfs(v);
            v.postVisit();
        }
    }

    void print(int indent = 0) const {
        for (const auto &entryPair : m_children) {
            for (int i = 0; i < indent; ++i)
                std::cout << "    ";
            std::cout << entryPair.first  << std::endl;
            entryPair.second->print(indent + 1);
        }
    }

    ~ResultTree() {
        clear();
    }

    // Insert a result object into the result tree with edges indicated by the
    // sequence of strings curr..end
    void setResult(std::vector<std::string>::const_iterator curr,
                   std::vector<std::string>::const_iterator end, Result *result)
    {
        if (curr == end) {
            setResult(result);
        }
        else {
            auto existing = m_children.find(*curr);
            if (existing != m_children.end()) {
                existing->second->setResult(++curr, end, result);
            }
            else {
                _RTPtr newNode(new ResultTree());
                m_children.insert(make_pair(*curr, newNode));
                newNode->setResult(++curr, end, result);
            }
        }
    }

    void remove(std::vector<std::string>::const_iterator curr,
                std::vector<std::string>::const_iterator end)
    {
        assert(curr != end);
        auto child_it = m_children.find(*curr);
        if (child_it == m_children.end())
            throw std::runtime_error(std::string("Removal path invalid!"));
        if (++curr == end)
            m_children.erase(child_it);
        else
            child_it->second->remove(curr, end);
    }

    _ConstRTPtr
    getSubtree(std::vector<std::string>::const_iterator curr,
               std::vector<std::string>::const_iterator end) const
    {
        assert(curr != end);
        auto child_it = m_children.find(*curr);
        if (child_it == m_children.end())
            throw std::runtime_error(std::string("Path not found!"));
        return (++curr == end) ? child_it->second :
                                 child_it->second->getSubtree(curr, end);
    }

    std::shared_ptr<const Result>
    getResult(std::vector<std::string>::const_iterator curr,
              std::vector<std::string>::const_iterator end) const
    {
        return getSubtree(curr, end)->getResult();
    }

private:
    ////////////////////////////////////////////////////////////////////////////
    // Members
    ////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<Result> m_result;

    typedef std::map<std::string, _RTPtr, NaturalLess> _ChildMap;
    _ChildMap m_children;
};

template<typename T>
inline std::string generateEntryName(const std::string &nameSuggestion,
             std::map<std::string, T> &collection,
             const T &entry)
{
    std::string name = nameSuggestion;

    auto it = collection.find(name);
    if (it != collection.end()) {
        // Only create a new entry if the existing one differs.
        if (!(it->second == entry)) {
            std::vector<std::string> keys;
            for (auto existing: collection)
                keys.push_back(existing.first);
            name = uniqueName(nameSuggestion, keys);
        }
    }

    return name;
}

template<typename Generator>
std::string ResultsCollector<Generator>::
addModel(const std::string &nameSuggestion, const Model &model,
         const BBox_t &gridBBox)
{
    // Delete the previously selected model if no results were added.
    // This allows overwriting of an existing model of the same name with no
    // results.
    m_selectedModel.clear();
    clean();

    RModel entry(model, gridBBox);
    std::string name = generateEntryName(nameSuggestion, m_models, entry);
    m_models[name] = entry;

    // Select the newly added model
    m_selectedModel = name;

    return name;
}

template<typename Generator>
std::string ResultsCollector<Generator>::
addSettings(const std::string &nameSuggestion, const AnalysisSettings &settings)
{
    // Delete the previously selected settings if no results were added.
    // This allows overwriting of an existing settings of the same name with no
    // results.
    m_selectedSettings.clear();
    clean();

    std::string name = generateEntryName(nameSuggestion, m_settings, settings);
    m_settings[name] = settings;

    // Select the newly added settings
    m_selectedSettings = name;

    return name;
}

template<typename Generator>
bool ResultsCollector<Generator>::modelNameConflict(const std::string &name,
        const Model &model, const BBox_t &gridBBox)
{
    return (generateEntryName(name, m_models,
                              std::make_pair(model, gridBBox)) != name);
}

template<typename Generator>
bool ResultsCollector<Generator>::settingsNameConflict(const std::string &name,
        const AnalysisSettings &settings)
{
    return (generateEntryName(name, m_settings, settings) != name);
}


////////////////////////////////////////////////////////////////////////////////
/*! Delete all models/settings for which no results are recorded
//  (except the currently selected pair)
*///////////////////////////////////////////////////////////////////////////////
template<typename Generator>
void ResultsCollector<Generator>::clean()
{
    // Prune empty result subtrees from models->settings hierarchy
    for (auto mscit = m_models_settings_collection.begin();
              mscit != m_models_settings_collection.end(); /* nop */) {
        bool pruneModel = true;
        _InnerMapType &sc = mscit->second;
        for (auto scit = sc.begin(); scit != sc.end(); /* nop */) {
            // Delete the settings entry if it has no results under this model
            if (scit->second->prune()) {
                scit = sc.erase(scit);
            }
            else {
                ++scit;
                // There is at least one result for this model
                pruneModel = false;
            }
        }
        // Delete the model collection, model if it has no results.
        if (pruneModel)
            mscit = m_models_settings_collection.erase(mscit);
        else
            ++mscit;
    }

    // Prune empty result subtrees from settings->models hierarchy
    for (auto smcit = m_settings_models_collection.begin();
              smcit != m_settings_models_collection.end(); /* nop */) {
        bool pruneSettings = true;
        _InnerMapType &mc = smcit->second;
        for (auto mcit = mc.begin(); mcit != mc.end(); /* nop */) {
            // Delete the model entry if it has no results under these settings
            if (mcit->second->prune()) {
                mcit = mc.erase(mcit);
            }
            else {
                ++mcit;
                // There is at least one result for these settings
                pruneSettings = false;
            }
        }
        // Delete the settings collection, settings if it has no results.
        if (pruneSettings)
            smcit = m_settings_models_collection.erase(smcit);
        else
            ++smcit;
    }

    // Delete non-existent models
    for (auto it = m_models.begin(); it != m_models.end(); /* nop */) {
        const std::string &name = it->first;
        // Only remove the non-selected models
        bool remove = (name != m_selectedModel) &&
            (m_models_settings_collection.find(name) ==
             m_models_settings_collection.end());
        if (remove)
            it = m_models.erase(it);
        else
            ++it;
    }

    // Delete non-existent settings.
    for (auto it = m_settings.begin(); it != m_settings.end(); /* nop */) {
        const std::string &name = it->first;
        // Only remove the non-selected settings
        bool remove = (name != m_selectedSettings) &&
            (m_settings_models_collection.find(name) ==
             m_settings_models_collection.end());
        if (remove)
            it = m_settings.erase(it);
        else
            ++it;
    }
}
