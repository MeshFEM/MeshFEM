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

template<typename Vector>
class CSGTree {

public:
    class CSGNode;
    class CSGBoolNode;
    class CSGPrimitive;
    class CSGRectangleNode;
    class CSGEllipseNode;
    typedef BBox<Vector> BBox_t;
    typedef typename Vector::Scalar Real;

    CSGTree()
    {
        CSGRectangleNode *rectA = new CSGRectangleNode(Vector(0, .0), Vector(1.0, 1.0));
        CSGRectangleNode *rectB = new CSGRectangleNode(Vector(0, 0), Vector(3.80, .25), 30);
        CSGBoolNode *rectUnion = new CSGBoolNode(UNION, rectA, rectB);
        CSGEllipseNode *ellipse = new CSGEllipseNode(Vector(0, .5), Vector(.5, .7));
        CSGBoolNode *subtract = new CSGBoolNode(SUBTRACT, rectUnion, ellipse);
        m_roots.push_back(rectA);
        // CSGEllipseNode *circle = new CSGEllipseNode(Vector(0, 0), Vector(1.5, 1.5));
        // m_roots.push_back(new CSGBoolNode(INTERSECT, subtract, circle));
    }

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

    int numRoots() const { return m_roots.size(); }
    const CSGNode *root(size_t i) const {
        assert(i < m_roots.size());
        return m_roots[i];
    }
    CSGNode *root(size_t i) {
        assert(i < m_roots.size());
        return m_roots[i];
    }

    ~CSGTree() {
        for (RootIt it = m_roots.begin(); it != m_roots.end(); ++it) {
            delete *it;
        }
    }

private:
    typedef typename std::vector<CSGNode *>::const_iterator RootIt;
    std::vector<CSGNode *> m_roots;
};

#include "CSGTree.inl"

#endif // CSGTREE_HH
