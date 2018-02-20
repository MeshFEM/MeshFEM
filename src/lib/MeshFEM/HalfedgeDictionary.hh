////////////////////////////////////////////////////////////////////////////////
// HalfedgeDictionary.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a custom hashmap-like data structure for speeding up
//      halfedge queries from (start vertex, end vertex) pairs.
//
//      Assumes that vertices don't have high incident edge count:
//      Lookups are O(max_{v \in vertices} incidentCount(v))
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//
//  Created:  07/19/2011 16:35:46
//  Revision History:
//      07/19/2011  Julian Panetta    Initial Revision
////////////////////////////////////////////////////////////////////////////////
#ifndef HALFEDGE_DICTIONARY_HH
#define HALFEDGE_DICTIONARY_HH
#include <list>
#include <vector>
#include <cassert>


/** Returned for failed queries */
#define INVALID_HALFEDGE -1L
#define INVALID_RECORD -1L
#define INVALID_VERTEX -1L

class HalfedgeDictionary
{
private:
    // NOTE: IF THIS IS CHANGED, THE getBucket FUNCTION MUST BE UPDATED
    static const size_t BUCKETS_PER_VERTEX = 8;

    struct EdgeRecord {
        size_t endVertex;
        size_t halfedge;
        size_t nextRecord;

        EdgeRecord()
            : endVertex(INVALID_VERTEX), halfedge(INVALID_HALFEDGE),
              nextRecord(INVALID_RECORD)
        { }

        EdgeRecord(size_t end, size_t halfedge)
            : endVertex(end), halfedge(halfedge),
              nextRecord(INVALID_RECORD)
        { }
    };

    typedef size_t EdgeBucket;
    typedef size_t EdgeRecordIdx;

    // Indices of the first EdgeRecord
    std::vector<size_t> m_halfedgeBuckets;
    std::vector<EdgeRecord>   m_halfedges;

    EdgeBucket &getBucket(size_t start, size_t end)
    {
        assert((start * BUCKETS_PER_VERTEX) < m_halfedgeBuckets.size());
        return m_halfedgeBuckets[(start << 3) + (end & 7)];
    }

    EdgeBucket getBucket(size_t start, size_t end) const
    {
        assert((start * BUCKETS_PER_VERTEX) < m_halfedgeBuckets.size());
        return m_halfedgeBuckets[(start << 3) + (end & 7)];
    }

public:
    HalfedgeDictionary() { }

    ////////////////////////////////////////////////////////////////////////////
    /*! Constructor allocates all space ever needed by HE structure.
    //  @param[in]  numVertices     the number of vertices
    //  @param[in]  numHalfedges    the number of halfedges
    *///////////////////////////////////////////////////////////////////////////
    HalfedgeDictionary(size_t numVertices, size_t numHalfedges)
    {
        init(numVertices, numHalfedges);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! (Re)allocates all space ever needed by HE structure.
    //  Can be used to reset the dictionary.
    //  @param[in]  numVertices     the number of vertices
    //  @param[in]  numHalfedges    the number of halfedges
    *///////////////////////////////////////////////////////////////////////////
    void init(size_t numVertices, size_t numHalfedges)
    {
        m_halfedgeBuckets.resize(0);
        m_halfedgeBuckets.resize(BUCKETS_PER_VERTEX * numVertices,
                                 INVALID_RECORD);
        m_halfedges.resize(0);
        m_halfedges.reserve(numHalfedges);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Inserts a halfedge (start --> end)
    //  @param[in]  start       halfedge start vertex index
    //  @param[in]  end         halfedge end vertex index
    //  @param[in]  halfedge    halfedge index
    *///////////////////////////////////////////////////////////////////////////
    void insert(size_t start, size_t end, size_t halfedge)
    {
        EdgeBucket &bucket = getBucket(start, end);
        EdgeRecordIdx curr = bucket;
        EdgeRecordIdx prev = INVALID_RECORD;
        while ((curr != (size_t) INVALID_RECORD) &&
               (m_halfedges[curr].endVertex != end)) {
            prev = curr;
            curr = m_halfedges[curr].nextRecord;
        };

        if (curr == (size_t) INVALID_RECORD) {
            // Insert a new halfedge, not exceeding the known HE count
            curr = m_halfedges.size();
            assert(curr < m_halfedges.capacity());
            m_halfedges.push_back(EdgeRecord(end, halfedge));

            if (prev != (size_t) INVALID_RECORD) {
                // Insert at end of list
                assert(m_halfedges[prev].nextRecord == (size_t) INVALID_RECORD);
                m_halfedges[prev].nextRecord = curr;
            }
            else    {
                // Insert into empty list
                bucket = curr;
            }
        }
        else    {
            // Update an existing record
            EdgeRecord &rec = m_halfedges[curr];
            if (rec.halfedge != halfedge)    {
                printf("WARNING: changing index for halfedge %i --> %i to %i\n",
                        (unsigned int) start, (unsigned int) end,
                        (unsigned int) halfedge);
                rec.halfedge = halfedge;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Retrieves the index for the halfedge (start --> end)
    //  @param[in]  start       halfedge start vertex index
    //  @param[in]  end         halfedge end vertex index
    //  @return     index of (start --> end), or
    //              INVALID_HALFEDGE if none is found
    *///////////////////////////////////////////////////////////////////////////
    size_t halfedge(size_t start, size_t end) const
    {
        EdgeRecordIdx curr = getBucket(start, end);
        while (curr != (size_t) INVALID_RECORD) {
            if (m_halfedges[curr].endVertex == end)
                break;
            curr = m_halfedges[curr].nextRecord;
        };

        return (curr != (size_t) INVALID_RECORD) ? m_halfedges[curr].halfedge
                                                 : INVALID_HALFEDGE;
    }
};

#endif
