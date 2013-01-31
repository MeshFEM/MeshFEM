////////////////////////////////////////////////////////////////////////////////
// CSGTreeModel.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements the model for the CSG Tree View
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/29/2013 16:17:36
////////////////////////////////////////////////////////////////////////////////
#include <QtGui>
#include "CSGTreeModel.hh"
#include <cassert>
#include <iostream>

QVariant CSGTreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || (role != Qt::DisplayRole))
        return QVariant();
    const CSGNode *node = getNode(index);
    return QVariant(node->name().c_str());
}

QVariant CSGTreeModel::headerData(int section, Qt::Orientation orientation,
            int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return QVariant("CSG Layers");
    return QVariant();
}

QModelIndex CSGTreeModel::index(int row, int column,
                                const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent)) {
        return QModelIndex();
    }

    CSGNode *node = NULL;
    if (!parent.isValid()) {
        node = m_tree.root(row);
    }
    else {
        CSGBoolNode *parentNode = dynamic_cast<CSGBoolNode *>(getNode(parent));
        assert(parentNode);

        node = parentNode->child(row);
    }

    if (node != NULL)
        return createIndex(row, column, node);
    return QModelIndex();
}

QModelIndex CSGTreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    CSGNode *childNode = getNode(index);
    CSGNode *parentNode = childNode->parent();

    if (parentNode == NULL)
        return QModelIndex();

    return createIndex(m_tree.childIndex(parentNode), 0, parentNode);
}

int CSGTreeModel::rowCount(const QModelIndex &parent) const
{
    if (parent.column() > 0)
        return 0;

    const CSGNode *parentItem;
    if (!parent.isValid())
        return m_tree.numRoots();
    else
        parentItem = getNode(parent);

    return parentItem->numChildren();
}

int CSGTreeModel::columnCount(const QModelIndex &parent) const
{
    return 1;
}

Qt::ItemFlags CSGTreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}
