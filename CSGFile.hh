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

    QString name = nodeData["name"].toString();
    QString type = nodeData["type"].toString();

    enum {N_OP, N_RECT, N_ELLIPSE} node_type;
    CSGOperation op;
    if (type == "intersect")      { node_type = N_OP; op = INTERSECT; }
    else if (type == "union")     { node_type = N_OP; op = UNION; }
    else if (type == "subtract")  { node_type = N_OP; op = SUBTRACT; }
    else if (type == "rectangle") { node_type = N_RECT; }
    else if (type == "ellipse")   { node_type = N_ELLIPSE; }
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
    else if ((node_type == N_RECT) || (node_type == N_ELLIPSE)) {
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


#endif // CSGFILE_HH
