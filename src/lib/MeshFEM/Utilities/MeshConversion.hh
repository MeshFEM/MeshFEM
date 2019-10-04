#ifndef MESH_CONVERSION_HH
#define MESH_CONVERSION_HH

#include <Eigen/Dense>
#include <MeshFEM/MeshIO.hh>

template<class Mesh>
using VType = Eigen::Matrix<typename Mesh::EmbeddingSpace::Scalar,
                            Eigen::Dynamic, Mesh::EmbeddingSpace::RowsAtCompileTime>;

template<class Mesh>
VType<Mesh> getV(const Mesh &m) {
    VType<Mesh> V(Eigen::Index(m.numVertices()), Eigen::Index(VType<Mesh>::ColsAtCompileTime));
    for (auto v : m.vertices())
        V.row(v.index()) = v.node()->p;

    return V;
}

template<class Mesh>
Eigen::Matrix<int, Eigen::Dynamic, Mesh::K + 1> getF(const Mesh &m) {
    Eigen::Matrix<int, Eigen::Dynamic, Mesh::K + 1> F(m.numElements(), Mesh::K + 1);
    for (auto e : m.elements()) {
        for (auto v : e.vertices())
            F(e.index(), v.localIndex()) = v.index();
    }
    return F;
}

inline Eigen::MatrixX3d getV(const std::vector<MeshIO::IOVertex> &vertices) {
    const size_t nv = vertices.size();
    Eigen::MatrixX3d V(nv, 3);
    for (size_t i = 0; i < nv; ++i)
        V.row(i) = vertices[i].point.transpose();
    return V;
}

inline Eigen::MatrixXi getF(const std::vector<MeshIO::IOElement> &elements) {
    const size_t K = elements.at(0).size();
    const size_t ne = elements.size();
    Eigen::MatrixXi F(ne, K);
    for (size_t i = 0; i < ne; ++i) {
        const auto &e = elements[i];
        if (e.size() != K) throw std::runtime_error("Element size mismatch");
        for (size_t j = 0; j < K; ++j)
            F(i, j) = e[j];
    }
    return F;
}

// We assume libigl's  |V|x3 or |V|x2 vertex array format, but
// if the number of columns is not 2 or 3, we try to interpret it as
// a 3x|V| or 2x|V| array.
template<class Derived>
std::vector<MeshIO::IOVertex> getMeshIOVertices(const Eigen::MatrixBase<Derived> &V) {
    std::vector<MeshIO::IOVertex> result;

    size_t nv;
    bool pointsAsRows;
    int dim;

    if ((V.cols() == 2) || (V.cols() == 3)) {
        nv = V.rows();
        dim = V.cols();
        pointsAsRows = true;
    }
    else if ((V.rows() == 2) || (V.rows() == 3)) {
        nv = V.cols();
        dim = V.rows();
        pointsAsRows = false;
    }
    else {
        throw std::runtime_error("Invalid array shape passed to getMeshIOVertices.");
    }

    using V3d = Eigen::Vector3d;

    result.reserve(nv);
    for (size_t i = 0; i < nv; ++i) {
        V3d v(V3d::Zero());
        if (pointsAsRows) v.head(dim) = V.row(i).template cast<double>();
        else              v.head(dim) = V.col(i).template cast<double>();
        result.emplace_back(v);
    }

    return result;
}

template<class Derived>
std::vector<MeshIO::IOElement> getMeshIOElements(const Eigen::MatrixBase<Derived> &F) {
    std::vector<MeshIO::IOElement> result;
    const size_t ne = F.rows();
    const size_t nc = F.cols();
    result.reserve(ne);
    for (size_t i = 0; i < ne; ++i) {
        result.emplace_back(nc);
        for (size_t c = 0; c < nc; ++c)
            result.back()[c] = F(i, c);
    }

    return result;
}


template<class Derived1, class Derived2>
std::pair<std::vector<MeshIO::IOVertex>, std::vector<MeshIO::IOElement>>
getMeshIO(const Eigen::MatrixBase<Derived1> &V, const Eigen::MatrixBase<Derived2> &F) {
    return { getMeshIOVertices(V), getMeshIOElements(F) };
}

#endif /* end of include guard: MESH_CONVERSION_HH */
