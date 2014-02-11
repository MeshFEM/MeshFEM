////////////////////////////////////////////////////////////////////////////////
// CSGFile.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements writing/reading .csg files. These are JSON files describing
//      a CSG tree.
//      Uses Boost property trees to parse but not to write! Boost's property
//      trees are untyped (all values are converted to strings) and have a hacky
//      implementation of arrays (anonymous nodes). This is okay for reading,
//      but it would produce garbage output.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/03/2013 23:17:00
////////////////////////////////////////////////////////////////////////////////
#ifndef CSGFILE_HH
#define CSGFILE_HH

#include "GlobalTypes.hh"
#include "CSGTree.hh"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cassert>

using boost::property_tree::ptree;

template<typename Vector>
void parseVector(const ptree &pt, Vector &v)
{
    int nComponentsRead = 0;
    BOOST_FOREACH(const ptree::value_type &val, pt) {
        if (!val.first.empty()) {
            nComponentsRead = -1; break;
        }
        try {
            if (nComponentsRead < v.size())
                v[nComponentsRead] = val.second.get_value<double>();
            ++nComponentsRead;
        }
        catch (...) { nComponentsRead = -1; break; }
    }

    if (nComponentsRead != v.size()) {
        throw std::runtime_error(std::string("Error parsing vector"));
    }
}

template<typename Vector>
typename CSGTree<Vector>::CSGNode *parseNode(ptree &pt)
{
    typedef           CSGTree<Vector>           _CSGTree;
    typedef typename _CSGTree::CSGNode          CSGNode;
    typedef typename _CSGTree::CSGBoolNode      CSGBoolNode;
    typedef typename _CSGTree::CSGRectangleNode CSGRectangleNode;
    typedef typename _CSGTree::CSGEllipseNode   CSGEllipseNode;
    typedef typename _CSGTree::CSGPieSliceNode  CSGPieSliceNode;
    typedef typename _CSGTree::CSGLaminateNode  CSGLaminateNode;

    std::string name = pt.get<std::string>("name");
    std::string type = pt.get<std::string>("type");

    enum {N_OP, N_RECT, N_ELLIPSE, N_PIESLICE, N_LAMINATE} node_type;
    CSGOperation op;
    if (type == "intersect")      { node_type = N_OP; op = INTERSECT; }
    else if (type == "union")     { node_type = N_OP; op = UNION; }
    else if (type == "subtract")  { node_type = N_OP; op = SUBTRACT; }
    else if (type == "rectangle") { node_type = N_RECT; }
    else if (type == "ellipse")   { node_type = N_ELLIPSE; }
    else if (type == "pieslice")  { node_type = N_PIESLICE; }
    else if (type == "laminate")  { node_type = N_LAMINATE; }
    else {
        throw std::runtime_error(std::string("Illegal CSG node type: ") + type);
    }

    CSGNode *node;
    if (node_type == N_OP) {
        ptree ltree, rtree;
        try {
            ltree = pt.get_child("left");
            rtree = pt.get_child("right");
        }
        catch (boost::property_tree::ptree_bad_path &e) {
            throw std::runtime_error(std::string("Missing left or right subtree"));
        }

        CSGNode *left, *right;
        left = parseNode<Vector>(ltree);
        try {
            right = parseNode<Vector>(rtree);
        }
        catch (...) {
            // Destroy completed left subtree if there was an error parsing the
            // right...
            delete left;
            throw;
        }

        node = new CSGBoolNode(op, left, right);
    }
    else if ((node_type == N_RECT) || (node_type == N_ELLIPSE) ||
             (node_type == N_PIESLICE) || (node_type == N_LAMINATE)) {
        Vector center, dimensions;
        parseVector(pt.get_child("center"), center);
        parseVector(pt.get_child("dimensions"), dimensions);

        double rot;
        try { rot = pt.get<double>("rotation"); }
        catch (...) { throw std::runtime_error("Error parsing rotation."); }
        
        if (node_type == N_RECT)
            node = new CSGRectangleNode(center, dimensions, rot);
        else if (node_type == N_ELLIPSE)
            node = new CSGEllipseNode(center, dimensions, rot);
        else if (node_type == N_PIESLICE)
            node = new CSGPieSliceNode(center, dimensions, rot);
        else if (node_type == N_LAMINATE)
            node = new CSGLaminateNode(center, dimensions, rot);
        else
            assert(false);
    }
    else {
        assert(false);
    }

    node->setName(name);
    return node;
}

template<typename Vector>
void parseCSGFile(const char *path, CSGTree<Vector> &csgTree)
{
    boost::property_tree::ptree pt;
    read_json(path, pt);

    typename CSGTree<Vector>::CSGNode *node = parseNode<Vector>(pt);
    csgTree.setRoot(node);
}

template<typename Vector>
void writeNode(std::ofstream &os, int indentLevel,
               const typename CSGTree<Vector>::CSGNode *node)
{
    typedef           CSGTree<Vector>           _CSGTree;
    typedef typename _CSGTree::CSGNode          CSGNode;
    typedef typename _CSGTree::CSGBoolNode      CSGBoolNode;
    typedef typename _CSGTree::CSGPrimitive     CSGPrimitive;

    std::string indent(4 * indentLevel, ' ');
    const char *type;
    bool isPrim;
    switch (node->nodeType()) {
        case CSG_NODE_INTERSECT: type = "intersect"; isPrim = false; break;
        case CSG_NODE_UNION:     type = "union";     isPrim = false; break;
        case CSG_NODE_SUBTRACT:  type = "subtract";  isPrim = false; break;
        case CSG_NODE_RECT:      type = "rectangle"; isPrim = true; break;
        case CSG_NODE_ELLIPSE:   type = "ellipse";   isPrim = true; break;
        case CSG_NODE_PIE_SLICE: type = "pieslice";  isPrim = true; break;
        case CSG_NODE_LAMINATE:  type = "laminate";  isPrim = true; break;
        default: assert(false);
    }

    os << indent << "\"name\": \"" << node->name() << "\"," << std::endl;
    os << indent << "\"type\": \""  << type << "\"," << std::endl;

    if (isPrim) {
        const CSGPrimitive *pNode = dynamic_cast<const CSGPrimitive *>(node);
        assert(pNode != NULL);
        Vector v = pNode->getCenter();
        os << indent << "\"center\": ["  << v[0] << ", " << v[1] << "],"
           << std::endl;
        v = pNode->getDimensions();
        os << indent << "\"dimensions\": ["  << v[0] << ", " << v[1] << "],"
           << std::endl;
        os << indent << "\"rotation\": " << pNode->getRotation() << std::endl;
    }
    else {
        const CSGBoolNode *bNode = dynamic_cast<const CSGBoolNode *>(node);
        assert(bNode != NULL);
        os << indent << "\"left\": {" << std::endl;
        writeNode<Vector>(os, indentLevel + 1, bNode->child(0));
        os << indent << "}," << std::endl;
        os << indent << "\"right\": {" << std::endl;
        writeNode<Vector>(os, indentLevel + 1, bNode->child(1));
        os << indent << "}" << std::endl;
    }
}

template<typename Vector>
void writeCSGFile(const char *path, const CSGTree<Vector> &csgTree)
{
    std::ofstream os(path);
    if (!os.is_open())
        throw std::runtime_error("Couldn't open csg output file.");

    os << '{' << std::endl;
    
    if (csgTree.numRoots() > 0) {
        writeNode<Vector>(os, 1, csgTree.root(0));
    }

    os << '}' << std::endl;

}

#endif // CSGFILE_HH
