////////////////////////////////////////////////////////////////////////////////
// CSGFile.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements writing/reading .csg files. These are JSON files describing
//      a CSG tree.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/03/2013 23:17:00
////////////////////////////////////////////////////////////////////////////////
#ifndef CSGFILE_HH
#define CSGFILE_HH

#include "GlobalTypes.hh"
#include "CSGTree.hh"
#include <qjson/parser.h>
#include <QFile>
#include <QString>
#include <QTextStream>
#include <QVariantMap>
#include <iostream>
#include <stdexcept>
#include <cassert>

template<typename Vector>
typename CSGTree<Vector>::CSGNode *parseNode(const QVariantMap &nodeData)
{
    typedef           CSGTree<Vector>           _CSGTree;
    typedef typename _CSGTree::CSGNode          CSGNode;
    typedef typename _CSGTree::CSGBoolNode      CSGBoolNode;
    typedef typename _CSGTree::CSGRectangleNode CSGRectangleNode;
    typedef typename _CSGTree::CSGEllipseNode   CSGEllipseNode;
    typedef typename _CSGTree::CSGPieSliceNode  CSGPieSliceNode;

    QString name = nodeData["name"].toString();
    QString type = nodeData["type"].toString();

    enum {N_OP, N_RECT, N_ELLIPSE, N_PIESLICE} node_type;
    CSGOperation op;
    if (type == "intersect")      { node_type = N_OP; op = INTERSECT; }
    else if (type == "union")     { node_type = N_OP; op = UNION; }
    else if (type == "subtract")  { node_type = N_OP; op = SUBTRACT; }
    else if (type == "rectangle") { node_type = N_RECT; }
    else if (type == "ellipse")   { node_type = N_ELLIPSE; }
    else if (type == "pieslice")  { node_type = N_PIESLICE; }
    else {
        throw std::runtime_error(std::string("Illegal CSG node type: ") +
                                 type.toStdString());
    }

    CSGNode *node;
    if (node_type == N_OP) {
        QVariantMap lSubtree = nodeData["left"].toMap();
        QVariantMap rSubtree = nodeData["right"].toMap();
        if (lSubtree.empty() || rSubtree.empty())
            throw std::runtime_error(std::string("Missing left or right subtree."));
        CSGNode *left, *right;
        left = parseNode<Vector>(lSubtree);
        try {
            right = parseNode<Vector>(rSubtree);
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
             (node_type == N_PIESLICE)) {
        Vector center, dimensions;
        bool ok;
        int nComponentsRead = 0;
        foreach (QVariant coordinate, nodeData["center"].toList()) {
            if (nComponentsRead < 2) {
                center[nComponentsRead] = coordinate.toDouble(&ok);
                if (!ok) { break; }
            }
            ++nComponentsRead;
        }
        if ((!ok) || (nComponentsRead != 2)) {
            throw std::runtime_error(std::string("Error parsing center."));
        }
        nComponentsRead = 0;
        foreach (QVariant coordinate, nodeData["dimensions"].toList()) {
            if (nComponentsRead < 2) {
                dimensions[nComponentsRead] = coordinate.toDouble(&ok);
                if (!ok) { break; }
            }
            ++nComponentsRead;
        }
        if ((!ok) || (nComponentsRead != 2)) {
            throw std::runtime_error(std::string("Error parsing dimensions."));
        }

        double rot = nodeData["rotation"].toDouble(&ok);
        if (!ok)
            throw std::runtime_error(std::string("Error parsing rotation."));
        
        if (node_type == N_RECT)
            node = new CSGRectangleNode(center, dimensions, rot);
        else if (node_type == N_ELLIPSE)
            node = new CSGEllipseNode(center, dimensions, rot);
        else if (node_type == N_PIESLICE)
            node = new CSGPieSliceNode(center, dimensions, rot);
        else
            assert(false);
    }
    else {
        assert(false);
    }

    node->setName(name.toStdString());
    return node;
}

template<typename Vector>
void parseCSGFile(const char *path, CSGTree<Vector> &csgTree)
{
    QFile file(path);
    bool success = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!success)
        throw std::runtime_error("Couldn't open file.");

    QTextStream in(&file);
    QString jsonContent = in.readAll();
    QJson::Parser parser;
    bool ok;
    QVariantMap result = parser.parse(jsonContent.toUtf8(),
                                      &ok).toMap();
    if (!ok)
        throw std::runtime_error("JSON parser failed.");
    
    typename CSGTree<Vector>::CSGNode *node = parseNode<Vector>(result);
    csgTree.setRoot(node);
}

template<typename Vector>
void writeNode(QTextStream &os, int indentLevel,
               const typename CSGTree<Vector>::CSGNode *node)
{
    typedef           CSGTree<Vector>           _CSGTree;
    typedef typename _CSGTree::CSGNode          CSGNode;
    typedef typename _CSGTree::CSGBoolNode      CSGBoolNode;
    typedef typename _CSGTree::CSGPrimitive     CSGPrimitive;

    QString indent(4 * indentLevel, ' ');
    const char *type;
    bool isPrim;
    switch (node->nodeType()) {
        case CSG_NODE_INTERSECT: type = "intersect"; isPrim = false; break;
        case CSG_NODE_UNION:     type = "union";     isPrim = false; break;
        case CSG_NODE_SUBTRACT:  type = "subtract";  isPrim = false; break;
        case CSG_NODE_RECT:      type = "rectangle"; isPrim = true; break;
        case CSG_NODE_ELLIPSE:   type = "ellipse";   isPrim = true; break;
        case CSG_NODE_PIE_SLICE: type = "pieslice";  isPrim = true; break;
        default: assert(false);
    }

    os << indent << "\"name\": \"" << node->name().c_str() << "\"," << endl;
    os << indent << "\"type\": \""  << type << "\"," << endl;

    if (isPrim) {
        const CSGPrimitive *pNode = dynamic_cast<const CSGPrimitive *>(node);
        assert(pNode != NULL);
        Vector v = pNode->getCenter();
        os << indent << "\"center\": ["  << v[0] << ", " << v[1] << "],"
           << endl;
        v = pNode->getDimensions();
        os << indent << "\"dimensions\": ["  << v[0] << ", " << v[1] << "],"
           << endl;
        os << indent << "\"rotation\": " << pNode->getRotation() << endl;
    }
    else {
        const CSGBoolNode *bNode = dynamic_cast<const CSGBoolNode *>(node);
        assert(bNode != NULL);
        os << indent << "\"left\": {" << endl;
        writeNode<Vector>(os, indentLevel + 1, bNode->child(0));
        os << indent << "}," << endl;
        os << indent << "\"right\": {" << endl;
        writeNode<Vector>(os, indentLevel + 1, bNode->child(1));
        os << indent << "}" << endl;
    }
}

template<typename Vector>
void writeCSGFile(const char *path, const CSGTree<Vector> &csgTree)
{
    QFile file(path);
    bool success = file.open(QIODevice::WriteOnly | QIODevice::Text);
    if (!success)
        throw std::runtime_error("Couldn't open csg output file.");

    QTextStream os(&file);

    os << '{' << endl;
    
    if (csgTree.numRoots() > 0) {
        writeNode<Vector>(os, 1, csgTree.root(0));
    }

    os << '}' << endl;

}

#endif // CSGFILE_HH
