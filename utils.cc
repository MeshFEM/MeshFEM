#include "utils.hh"
#include <vector>
#include <sstream>
#include <regex>
#include <map>
#include <list>
#include <queue>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/*! Create a name with the pattern "suggestion (#)" that is distinct from all
//  names in the collection "names"
//  Note: this is a very inefficient O(|names|^2) hack!
//  @param[in]  suggestion  name to make unique
//  @param[in]  names       collection of existing names
//  @return     generated name
*///////////////////////////////////////////////////////////////////////////////
template<typename Collection>
std::string uniqueName(std::string suggestion, const Collection &names)
{
    if (find(names.begin(), names.end(), suggestion) == names.end())
        return suggestion;

    // Trim any potential (#) suffix from the suggested name
    std::regex pattern("(.*)\\s\\([0-9]+\\)$");
    std::smatch match;
    if (std::regex_search(suggestion, match, pattern)) {
        suggestion = std::string(match[1].first, match[1].second);
    }

    boost::format formatter("%s (%i)");
    
    bool unique = false;
    std::string newName;
    for (int i = 0; !unique && (i <= (int) names.size()); ++i) {
        newName = (i == 0) ? suggestion
                           : boost::str(formatter % suggestion % i);
        unique = (find(names.begin(), names.end(), newName) == names.end());
    }

    assert(unique);
    return newName;
}

////////////////////////////////////////////////////////////////////////////////
/*! Create a new copy of str with all single quotes and backslashes escaped.
*///////////////////////////////////////////////////////////////////////////////
std::string escapedString(const std::string &str)
{
    size_t n = str.length();
    size_t resultSize = 0;
    std::string result;

    for (size_t i = 0; i < n; ++i) {
        char c = str[i];
        if (c == '\\' || c == '\'')
            ++resultSize;
        ++resultSize;
    }

    result.reserve(resultSize);
    for (size_t i = 0; i < n; ++i) {
        char c = str[i];
        if (c == '\\' || c == '\'')
            result += '\\';
        result += c;
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
/*! Expand an encoded sequence of numbers in the MATLAB-esque format:
//      1, 2:4, 1:10:20
//  Invalid ranges, e.g. 1:-1:2, generate no elements just as in MATLAB
//  @param[in]  range   encoded sequence
//  @return     vector holding each sequence element
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
std::vector<Real> expandRange(const std::string &range)
{
    std::vector<Real> sequence;
    std::vector<std::string> components;
    boost::split(components, range, boost::is_any_of(","));

    for (size_t i = 0; i < components.size(); ++i) {
        std::vector<std::string> parts;
        boost::split(parts, components[i], boost::is_any_of(":"));
        if (parts.size() == 1) {
            sequence.push_back(std::stod(parts[0]));
        }
        else if (parts.size() < 4) {
            Real first = std::stod(parts[0]);
            Real step = 1.0;
            Real last = std::stod(parts[1]);
            if (parts.size() == 3) {
                step = last;
                last = std::stod(parts[2]);
            }
            if (step == 0)
                continue;
            if ((last - first) / step < 0)
                continue;
            for (Real a = first; (step > 0 && a <= last) ||
                                 (step < 0 && a >= last); a += step) {
                sequence.push_back(a);
            }
        }
    }

    return sequence; 
}

////////////////////////////////////////////////////////////////////////////////
/*! Compute a spanning forest for an undirected graph specifed as a list of
//  edges. The graph's vertices are implicitly given by the vertices appearing
//  in this edge list.
//  @param[in]  in_edges    input graph's edges
//  @param[out] out_edges   output forest edges
*///////////////////////////////////////////////////////////////////////////////
void spanningForest(const std::vector<std::pair<size_t, size_t> > &in_edges,
                          std::vector<std::pair<size_t, size_t> > &out_edges)
{
    // Build traversable graph representation
    map<size_t, list<size_t> > adj;
    for (const pair<size_t, size_t> &i: in_edges) {
        adj[i.first ].push_back(i.second);
        adj[i.second].push_back(i.first);
    }

    // Build a spanning forest in out_edges using a BFS
    out_edges.clear();
    map<size_t, bool> visited;
    for (const auto &e: adj) {
        if (!visited[e.first]) {
            queue<size_t> bfsQueue;
            bfsQueue.push(e.first);
            visited[e.first] = true;
            while (!bfsQueue.empty()) {
                size_t u = bfsQueue.front();
                bfsQueue.pop();
                const list<size_t> &adj_u = adj[u];
                for (size_t v : adj_u) {
                    if (!visited[v]) {
                        visited[v] = true;
                        out_edges.push_back(make_pair(u, v));
                        bfsQueue.push(v);
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Template instantiations
////////////////////////////////////////////////////////////////////////////////
template string uniqueName(string suggsetion, const vector<string> &names);
template vector<double> expandRange(const std::string &range);
template vector<float>  expandRange(const std::string &range);
