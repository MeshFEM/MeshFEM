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
        for (std::string &str : nameComponents) {
            boost::trim(str);
            add(nameComponents.begin(), nameComponents.end(), result);
        }
    }

private:
    void add(std::vector<std::string>::const_iterator curr,
             std::vector<std::string>::const_iterator end, Result *result) {
        if (curr == end) {
            setResult(result);
        }
        else {
            ResultTree *newNode = new ResultTree(this);
            m_children.insert(make_pair(*curr, newNode));
            newNode->add(++curr, end, result);
        }
    }
public:

    // Set a result, overwriting any existing one.
    // Note: only leaves are allowed to hold results.
    void setResult(Result *r) {
        assert(m_children.size() == 0);
        if (m_result != NULL)
            delete m_result;
        m_result = r;
    }

    // Recursively destroy this tree's contents.
    void clear() {
        if (m_result != NULL) {
            assert(m_children.size() == 0);
            delete m_result;
            m_result = NULL;
        }
        else {
            for (std::pair<const std::string, ResultTree *> c : m_children)
                delete c.second;
            m_children.clear();
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

    ~ResultTree() {
        clear();
    }

private:
    Result *m_result;
    ResultTree *m_parent;
    std::map<std::string, ResultTree *> m_children;
};

template<typename T>
std::string addNamedEntry(const std::string &nameSuggestion,
                          std::map<std::string, T> &collection, const T &entry)
{
    std::string name;
    auto it = collection.find();
    if (it == collection.end())  {
        name = nameSuggestion;
        collection.insert(make_pair(nameSuggestion, entry));
    }
    else {
        if (it->second == entry) {
            name = nameSuggestion;
        }
        else {
            std::vector<std::string> keys;
            for (auto existing: collection)
                keys.push_back(existing.first);
            name = uniqueName(nameSuggestion, keys);
            collection.insert(make_pair(name, entry));
        }
    }

    return name;
}

template<typename Generator>
std::string ResultsCollector<Generator>::
addModel(const std::string &nameSuggestion, const Model &model)
{
    return addNamedEntry(nameSuggestion, m_models, model);
}

template<typename Generator>
std::string ResultsCollector<Generator>::
addSettings(const std::string &nameSuggestion, const AnalysisSettings &settings)
{
    return addNamedEntry(nameSuggestion, m_settings, settings);
}
