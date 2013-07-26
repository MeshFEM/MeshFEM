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
#include <boost/algorithm/string.hpp>

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
        m_sfields.assign(RESULT_NUM_TYPES);
        m_vfields.assign(RESULT_NUM_TYPES);
        m_hasSField.assign(RESULT_NUM_TYPES, false);
        m_hasVField.assign(RESULT_NUM_TYPES, false);
    }

    std::vector<bool> m_hasSField, m_hasVField;
    std::vector<SField> m_sfields;
    std::vector<VField> m_vfields;
};

template<typename Generator>
class ResultsCollector<Generator>::ResultTree
{
private:
    ResultTree(ResultTree *parent = NULL)
        : m_result(NULL), m_parent(parent) { }
    
public:
    ResultTree()
        : m_result(NULL), m_parent(NULL) { }

    void add(const std::string &name, Result *result) {
        std::vector<std::string> nameComponents;
        boost::split(nameComponents, name, boost::is_any_of(":"));
        for (std::string &str : nameComponents)
            boost::trim(str);
        add(nameComponents.begin(), nameComponents.end(), result);
    }

    const Result *getResult(const std::string &name) const {
        std::vector<std::string> nameComponents;
        boost::split(nameComponents, name, boost::is_any_of(":"));
        for (std::string &str : nameComponents)
            boost::trim(str);
        return getResult(nameComponents.begin(), nameComponents.end());
    }

private:
    // Insert a result object into the result tree with edges indicated by the
    // sequence of strings curr..end
    void add(std::vector<std::string>::const_iterator curr,
             std::vector<std::string>::const_iterator end, Result *result) {
        if (curr == end) {
            setResult(result);
        }
        else {
            auto existing = m_children.find(*curr);
            if (existing != m_children.end()) {
                existing->second->add(++curr, end, result);
            }
            else {
                ResultTree *newNode = new ResultTree(this);
                m_children.insert(make_pair(*curr, newNode));
                newNode->add(++curr, end, result);
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

public:

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

    // Recursively destroy this tree's contents.
    void clear() {
        delete m_result;
        m_result = NULL;

        for (std::pair<const std::string, ResultTree *> c : m_children)
            delete c.second;

        m_children.clear();
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
    Result *m_result;
    ResultTree *m_parent;
    std::map<std::string, ResultTree *> m_children;
};

template<typename T>
inline std::string addNamedEntry(const std::string &nameSuggestion,
                                 std::map<std::string, T> &collection,
                                 const T &entry)
{
    std::string name;
    auto it = collection.find();
    if (it == collection.end())  {
        name = nameSuggestion;
        collection[name] = entry;
    }
    else {
        if (it->second == entry) {
            // If the entry already is present, keep it
            name = nameSuggestion;
        }
        else {
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

    // Select the newly added model if none was selected
    if (m_selectedModel.size() == 0)
        m_selectedModel = name;

    return name;
}

template<typename Generator>
std::string ResultsCollector<Generator>::
addSettings(const std::string &nameSuggestion, const AnalysisSettings &settings)
{
    std::string name = addNamedEntry(nameSuggestion, m_settings, settings);

    // Select the newly added settings if none was selected
    if (m_selectedSettings.size() == 0)
        m_selectedSettings = name;

    return name;
}
