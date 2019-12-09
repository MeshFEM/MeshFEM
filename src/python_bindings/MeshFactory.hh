////////////////////////////////////////////////////////////////////////////////
// MeshFactory.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Constructs an instance of the appropriate FEMMesh template instantiation
//  based on runtime (K, degree, dimension) parameters.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/19/2019 18:35:54
////////////////////////////////////////////////////////////////////////////////
#ifndef MESHFACTORY_HH
#define MESHFACTORY_HH
#include <stdexcept>
#include <type_traits>

template<size_t K, size_t Degree, class EmbeddingSpace>
typename std::enable_if<(K <= EmbeddingSpace::RowsAtCompileTime), py::object>::type
MeshFactory(const std::vector<MeshIO::IOElement> &elements,
            const std::vector<MeshIO::IOVertex > &vertices) {
    // Note: while py::cast is not yet documented in the official documentation,
    // it accepts the return_value_policy as discussed in:
    //      https://github.com/pybind/pybind11/issues/1201
    // by setting the return value policy to take_ownership, we can avoid
    // memory leaks and double frees regardless of the holder type for FEMMesh.
    return py::cast(new FEMMesh<K, Degree, EmbeddingSpace>(elements, vertices),
                    py::return_value_policy::take_ownership);
}

template<size_t K, size_t Degree, class EmbeddingSpace>
typename std::enable_if<(K > EmbeddingSpace::RowsAtCompileTime), py::object>::type
MeshFactory(const std::vector<MeshIO::IOElement> &/* elements */,
            const std::vector<MeshIO::IOVertex > &/* vertices */) {
    throw std::runtime_error("Embedding dimension must be >= simplex dimension.");
}

template<size_t Degree, class EmbeddingSpace>
py::object MeshFactory(const std::vector<MeshIO::IOElement> &elements,
                       const std::vector<MeshIO::IOVertex > &vertices,
                       size_t simplexDimension) {
    if  (simplexDimension == 2) return MeshFactory<2, Degree, EmbeddingSpace>(elements, vertices);
    if  (simplexDimension == 3) return MeshFactory<3, Degree, EmbeddingSpace>(elements, vertices);
    else throw std::runtime_error("Unsupported simplex dimension K = " + std::to_string(simplexDimension));
}

template<class EmbeddingSpace>
py::object MeshFactory(const std::vector<MeshIO::IOElement> &elements,
                       const std::vector<MeshIO::IOVertex > &vertices,
                       size_t simplexDimension,
                       size_t degree) {
    if  (degree == 1) return MeshFactory<1, EmbeddingSpace>(elements, vertices, simplexDimension);
    if  (degree == 2) return MeshFactory<2, EmbeddingSpace>(elements, vertices, simplexDimension);
    else throw std::runtime_error("Unsupported Degree " + std::to_string(degree));
}

template<typename Real_ = double>
py::object MeshFactory(const std::vector<MeshIO::IOElement> &elements,
                       const std::vector<MeshIO::IOVertex > &vertices,
                       size_t simplexDimension,
                       size_t degree,
                       size_t embeddingDimension) {
    if (embeddingDimension == 2) return MeshFactory<Eigen::Matrix<Real_, 2, 1>>(elements, vertices, simplexDimension, degree);
    if (embeddingDimension == 3) return MeshFactory<Eigen::Matrix<Real_, 3, 1>>(elements, vertices, simplexDimension, degree);
    else throw std::runtime_error("Unsupported embedding dimension " + std::to_string(embeddingDimension));
}

#endif /* end of include guard: MESHFACTORY_HH */
