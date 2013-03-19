////////////////////////////////////////////////////////////////////////////////
// CSGTree.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a binary tree constructing a CSG object.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/28/2013 16:37:07
////////////////////////////////////////////////////////////////////////////////
#ifndef CSGTREE_HH
#define CSGTREE_HH

#include <algorithm>
#include <vector>
#include <cassert>
#include "Geometry.hh"

typedef enum { INTERSECT = 0, UNION = 1, SUBTRACT = 2 } CSGOperation;
typedef enum { CSG_NODE_RECT = 0, CSG_NODE_ELLIPSE = 1, CSG_NODE_INTERSECT = 2,
               CSG_NODE_UNION = 3, CSG_NODE_SUBTRACT = 4 } CSGNodeType;

template<typename Vector>
class CSGTree {

public:
    class CSGNode;
    class CSGGlueNode;
    class CSGBoolNode;
    class CSGPrimitive;
    class CSGRectangleNode;
    class CSGEllipseNode;
    typedef BBox<Vector> BBox_t;
    typedef Vector                  Vector_t;
    typedef typename Vector::Scalar Real;
    typedef BoundaryPoint<Vector> _BoundaryPoint;

    CSGTree() { }

    bool isInside(const Vector &p) const {
        for (RootIt it = m_roots.begin(); it != m_roots.end(); ++it) {
            if ((*it)->isInside(p))
                return true;
        }
        return false;
    }

    // Get a node's position in its parent's list of children
    int childIndex(const CSGNode *node) const {
        assert(node);
        int idx = -1;
        if (node->parent() == NULL) {
            RootIt loc = std::find(m_roots.begin(), m_roots.end(), node);
            if (loc != m_roots.end())
                idx = loc - m_roots.begin();
        }
        else {
            idx = node->parent()->indexOfChild(node);
        }

        return idx;
    }

    BBox<Vector> boundingBox() const {
        BBox<Vector> b;
        for (RootIt it = m_roots.begin(); it != m_roots.end(); ++it) {
            if (it == m_roots.begin())
                b = (*it)->boundingBox();
            else
                b.unionBox((*it)->boundingBox());
        }
        return b;
    }

    template<typename Functor>
    Functor dfs(Functor f, CSGNode *node = NULL);

    size_t numRoots() const { return m_roots.size(); }
    const CSGNode *root(size_t i) const {
        assert(i < m_roots.size());
        return m_roots[i];
    }
    CSGNode *root(size_t i) {
        assert(i < m_roots.size());
        return m_roots[i];
    }


    std::vector<_BoundaryPoint> boundaryPoints(Real pointSpacing)
    {
        typedef std::vector<_BoundaryPoint> BndPts;
        BndPts bndPoints;
        for (size_t i = 0; i < numRoots(); ++i) {
            BndPts rootBoundaryPts = root(i)->boundaryPoints(pointSpacing);
            bndPoints.insert(bndPoints.end(), rootBoundaryPts.begin(),
                             rootBoundaryPts.end());
        }
        return bndPoints;
    }

    // Note: the tree takes ownership of node when node becomes root!
    void setRoot(CSGNode *node) {
        clearRoots();
        m_roots.push_back(node);
    }

    void clearRoots() {
        for (RootIt it = m_roots.begin(); it != m_roots.end(); ++it) {
            delete *it;
        }
        m_roots.clear();
    }

    ~CSGTree() {
        clearRoots();
    }

private:
    typedef typename std::vector<CSGNode *>::const_iterator RootIt;
    std::vector<CSGNode *> m_roots;
};

#include "CSGTree.inl"

#endif // CSGTREE_HH
