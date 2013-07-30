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
#include <cassert>
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
    typedef enum { RESULT_PER_NODE = 0, RESULT_PER_ELEM = 1,
                   RESULT_PER_BOUNDARY = 2, RESULT_NUM_TYPES } ResultType;
    typedef typename Generator::SField SField;
    typedef typename Generator::VField VField;
    Result() { init(); }
    Result(ResultType stype, const SField &sfield) {
        init(); setScalarField(stype, sfield);
    }

    Result(ResultType vtype, const VField &vfield) {
        init(); setVectorField(vtype, vfield);
    }

    Result(ResultType stype, const SField &sfield,
           ResultType vtype, const VField &vfield) {
        init(); setScalarField(stype, sfield);
                setVectorField(vtype, vfield);
    }

public:
    void setVectorField(ResultType type, const VField &vfield) {
        assert(type < RESULT_NUM_TYPES);
        m_vfields[type] = vfield;
        m_hasVField[type] = true;
    }

    void setScalarField(ResultType type, const SField &sfield) {
        assert(type < RESULT_NUM_TYPES);
        m_sfields[type] = sfield;
        m_hasSField[type] = true;
    }

private:
    void init() {
        m_sfields.assign((size_t) RESULT_NUM_TYPES, SField());
        m_vfields.assign((size_t) RESULT_NUM_TYPES, VField());
        m_hasSField.assign((size_t) RESULT_NUM_TYPES, false);
        m_hasVField.assign((size_t) RESULT_NUM_TYPES, false);
    }

    std::vector<bool> m_hasSField, m_hasVField;
    std::vector<SField> m_sfields;
    std::vector<VField> m_vfields;
};

template<typename Generator>
class ResultsCollector<Generator>::ResultTree
{
private:
    ResultTree(ResultTree *parent)
        : m_result(NULL), m_parent(parent) { }
    
public:
    ResultTree()
        : m_result(NULL), m_parent(NULL) { }

    void setResult(const std::string &name, Result *result) {
        std::vector<std::string> nameComponents;
        boost::split(nameComponents, name, boost::is_any_of(":"));
        for (std::string &str : nameComponents)
            boost::trim(str);
        setResult(nameComponents.begin(), nameComponents.end(), result);
    }

    const Result *getResult(const std::string &name) const {
        std::vector<std::string> nameComponents;
        boost::split(nameComponents, name, boost::is_any_of(":"));
        for (std::string &str : nameComponents)
            boost::trim(str);
        return getResult(nameComponents.begin(), nameComponents.end());
    }

    // Set a result, overwriting any existing one.
    void setResult(Result *r) {
        // Originally I thought only leaves should hold results, but it might be
        // useful to have non-terminal results hold results too...
        // assert(m_children.size() == 0);
        delete m_result;
        m_result = r;
    }

    // Get the result stored at this node.
    const Result *getResult() const {
        if (m_result == NULL)
            throw std::runtime_error(std::string("result not found!"));
        return m_result;
    }

    bool hasResult() const {
        return (m_result != NULL);
    }

    // Get the count of all results in this collection (recursive)
    size_t numResults() const {
        size_t count = 0;
        if (hasResult())
            count = 1;
            
        for (const std::pair<const std::string, ResultTree *> &c : m_children) {
            count += c.second->numResults();
        }

        return count;
    }

    // Recursively destroy this tree's contents.
    void clear() {
        delete m_result;
        m_result = NULL;

        for (std::pair<const std::string, ResultTree *> &c : m_children)
            delete c.second;

        m_children.clear();
    }

    
    template<typename Visitor>
    void dfs(Visitor &v) {
        for (std::pair<const std::string, ResultTree *> &c : m_children) {
            v.preVisit(c.first);
            c.second->dfs(v);
            v.postVisit();
        }
    }

    int indexOfChild(const ResultTree *c) const {
        int pos = 0;
        for (const auto &entryPair : m_children) {
            if (entryPair.second == c)
                return pos;
            ++pos;
        }
        return -1;
    }


    void print(int indent = 0) const {
        for (const auto &entryPair : m_children) {
            for (size_t i = 0; i < indent; ++i)
                std::cout << "    ";
            std::cout << entryPair.first  << std::endl;
            entryPair.second->print(indent + 1);
        }
    }

    ~ResultTree() {
        clear();
    }

private:
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
                ResultTree *newNode = new ResultTree(this);
                m_children.insert(make_pair(*curr, newNode));
                newNode->setResult(++curr, end, result);
            }
        }
    }

    const Result *getResult(std::vector<std::string>::const_iterator curr,
                            std::vector<std::string>::const_iterator end) const
    {
        if (curr == end)
            return getResult();
        auto existing = m_children.find(*curr);
        if (existing == m_children.end())
            throw std::runtime_error(std::string("result not found!"));
        return existing->second->getResult(++curr, end);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Members
    ////////////////////////////////////////////////////////////////////////////
    Result *m_result;
    ResultTree *m_parent;
    std::map<std::string, ResultTree *> m_children;
};

template<typename T>
inline std::string addNamedEntry(const std::string &nameSuggestion,
                                 std::map<std::string, T> &collection,
                                 const T &entry)
{
    std::string name = nameSuggestion;

    auto it = collection.find(name);
    if (it == collection.end())  {
        collection[name] = entry;
    }
    else {
        // Only create a new entry if the existing one differs.
        if (!(it->second == entry)) {
            std::vector<std::string> keys;
            for (auto existing: collection)
                keys.push_back(existing.first);
            name = uniqueName(nameSuggestion, keys);
            collection[name] = entry;
        }
    }

    return name;
}

template<typename Generator>
std::string ResultsCollector<Generator>::
addModel(const std::string &nameSuggestion, const Model &model,
         const BBox_t &gridBBox)
{
    std::string name = addNamedEntry(nameSuggestion, m_models,
                                     std::make_pair(model, gridBBox));

    // Select the newly added model
    m_selectedModel = name;

    // Possibly delete the previously selected model if no results were added.
    clean();

    return name;
}

template<typename Generator>
std::string ResultsCollector<Generator>::
addSettings(const std::string &nameSuggestion, const AnalysisSettings &settings)
{
    std::string name = addNamedEntry(nameSuggestion, m_settings, settings);

    // Select the newly added settings
    m_selectedSettings = name;

    // Possibly delete the previously selected settings if no results were
    // added.
    clean();

    return name;
}

////////////////////////////////////////////////////////////////////////////////
/*! Delete all models/settings for which no results are recorded
//  (except the currently selected pair)
*///////////////////////////////////////////////////////////////////////////////
template<typename Generator>
void ResultsCollector<Generator>::clean()
{
    // TODO: Also remove entries from m_models_settings_collection and
    //       m_settings_models_collection.

    // Delete non-existent models
    for (auto it = m_models.begin(); it != m_models.end(); /* nop */) {
        const std::string &name = it->first;
        bool remove = false;
        // Only remove the non-selected models
        if (name != m_selectedModel) {
            auto mit = m_models_settings_collection.find(name);
            if (mit == m_models_settings_collection.end()) {
                remove = true;
            }
            else {
                size_t count = 0;
                for (auto &ms_entry : mit->second)
                    count += ms_entry.second->numResults();
                remove = (count == 0);
            }
        }
        if (remove)
            it = m_models.erase(it);
        else
            ++it;
    }

    // Delete non-existent settings.
    for (auto it = m_settings.begin(); it != m_settings.end(); /* nop */) {
        const std::string &name = it->first;
        bool remove = false;
        // Only remove the non-selected settings
        if (name != m_selectedSettings) {
            auto sit = m_settings_models_collection.find(name);
            if (sit == m_settings_models_collection.end()) {
                remove = true;
            }
            else {
                size_t count = 0;
                for (auto &sm_entry : sit->second)
                    count += sm_entry.second->numResults();
                remove = (count == 0);
            }
        }
        if (remove)
            it = m_settings.erase(it);
        else
            ++it;
    }
    
}
