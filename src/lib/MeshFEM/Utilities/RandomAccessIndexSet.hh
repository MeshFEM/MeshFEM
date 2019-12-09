////////////////////////////////////////////////////////////////////////////////
// RandomAccessIndexSet.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Data structure that gives O(1) random access to a subset of the integers
//      1...N. This subset allows insertion and deletion of the ith element in
//      the set in O(1) but the ordering of elements is arbitrary.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  04/13/2016 16:31:55
////////////////////////////////////////////////////////////////////////////////
#ifndef RANDOMACCESSINDEXSET_HH
#define RANDOMACCESSINDEXSET_HH

#include <vector>
#include <cassert>
#include <stdexcept>

struct RandomAccessIndexSet {
    RandomAccessIndexSet(size_t N) : m_locInSetPlusOne(N, 0) { }

    // returns true if the set changed
    bool insert(size_t i) {
        if (contains(i)) return false;
        m_indices.push_back(i);
        m_locInSetPlusOne.at(i) = m_indices.size();
        return true;
    }

    void remove(size_t i) { removeIndexAtLocation(locOfIndex(i)); }

    void removeIndexAtLocation(size_t loc) {
        size_t i = m_indices.at(loc);
        m_indices[loc] = m_indices.back();
        m_indices.pop_back();

        assert(contains(i));
        m_locInSetPlusOne[i] = 0;
    }

    size_t locOfIndex(size_t idx) const {
        size_t loc = m_locInSetPlusOne.at(idx);
        if (loc == 0) throw std::runtime_error("findIndex error: idx not in RandomAccessIndexSet");
        --loc;
        assert(m_indices.at(loc) == idx);
        return loc;
    }

    size_t indexAtLocation(size_t i) const { return m_indices.at(i); }

    size_t size() const { return m_indices.size(); }
    bool  empty() const { return m_indices.empty(); }

    bool contains(size_t i) const { return m_locInSetPlusOne.at(i) > 0; }
private:
    std::vector<size_t> m_indices;
    std::vector<size_t> m_locInSetPlusOne;
    //size_t m_size = 0;
};

#endif /* end of include guard: RANDOMACCESSINDEXSET_HH */
