////////////////////////////////////////////////////////////////////////////////
// CSGTreeModel.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements the model for the CSG Tree View
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/29/2013 16:07:04
////////////////////////////////////////////////////////////////////////////////
#ifndef CSGTREE_MODEL_HH
#define CSGTREE_MODEL_HH

#include <QObject>
#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>

#include "CSGTree.hh"
#include "GlobalTypes.hh"


class CSGTreeModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    typedef CSGTree_t::CSGNode CSGNode;
    typedef CSGTree_t::CSGBoolNode CSGBoolNode;

    CSGTreeModel(CSGTree_t &tree, QObject *parent = NULL)
        : QAbstractItemModel(parent), m_tree(tree) { }

    QVariant data(const QModelIndex &index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation,
            int role = Qt::DisplayRole) const;
    QModelIndex index(int row, int column, const QModelIndex &parent =
            QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
    Qt::ItemFlags flags(const QModelIndex &index) const;

    CSGNode *getNode(const QModelIndex &index) const {
        return static_cast<CSGNode *>(index.internalPointer());
    }

    QModelIndex getIndex(CSGNode *node) {
        int row = m_tree.childIndex(node);
        return createIndex(row, 0, node);
    }

    void csgTreeAboutToUpdate() {
        emit beginResetModel();
    }

    void csgTreeUpdated() {
        emit endResetModel();
    }

    ~CSGTreeModel() { }
private:
    CSGTree_t &m_tree;
};

#endif // CSGTREE_MODEL_HH
