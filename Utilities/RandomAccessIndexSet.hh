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

struct RandomAccessIndexSet {
    RandomAccessIndexSet(size_t N) : m_inSet(N, false) { }

    // returns true if the set changed
    bool insert(size_t i) {
        if (contains(i)) return false;
        m_indices.push_back(i);
        m_inSet.at(i) = true;
        ++m_size;
        return true;
    }

    void removeIndexAtLocation(size_t loc) {
        size_t i = m_indices.at(loc);
        m_indices[loc] = m_indices.back();
        m_indices.resize(m_indices.size() - 1);

        assert(m_inSet.at(i) == true);
        m_inSet.at(i) = false;
        assert(m_size > 0);
        --m_size;
    }

    // WARNING: linear time operation
    size_t findIndex(size_t idx) const {
        for (size_t loc = 0; loc < m_indices.size(); ++loc)
            if (m_indices[loc] == idx) return loc;
        throw std::runtime_error("findIndex error: idx not in RandomAccessIndexSet");
    }

    size_t indexAtLocation(size_t i) const {
        return m_indices.at(i);
    }

    size_t size() const { return m_size; }
    bool  empty() const { return size() == 0; }

    bool contains(size_t i) const { return m_inSet.at(i); }
private:
    std::vector<size_t> m_indices;
    std::vector<bool> m_inSet;
    size_t m_size = 0;
};

#endif /* end of include guard: RANDOMACCESSINDEXSET_HH */
