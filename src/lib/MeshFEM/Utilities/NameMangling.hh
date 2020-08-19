////////////////////////////////////////////////////////////////////////////////
// NameMangling.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Provides mangled names needed, e.g., for generating python bindings for
//  template instantiations
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef NAME_MANGLING_HH
#define NAME_MANGLING_HH

#include <string>
#include <array>
template<typename _Real>
std::string floatingPointTypeSuffix() {
    if (std::is_same<_Real,      double>::value) return "";
    if (std::is_same<_Real, long double>::value) return "_long_double";
    if (std::is_same<_Real,       float>::value) return "_float";
    throw std::runtime_error("Unrecognized floating point type");
}

template<size_t _K, size_t _Degree, class _EmbeddingSpace>
std::string getFEMName() {
    std::array<std::string, 2>  degreeNames{{"Linear", "Quadratic"}};
    std::array<std::string, 2> simplexNames{{"Tri", "Tet"}};

    std::string dimName = std::to_string(_EmbeddingSpace::RowsAtCompileTime);

    return degreeNames.at(_Degree - 1) + dimName + "D" + simplexNames.at(_K - 2);
}

template<class _Mesh>
std::string getMeshName() {
    using _Real = typename _Mesh::Real;
    return getFEMName<_Mesh::K, _Mesh::Deg, typename _Mesh::EmbeddingSpace>() + "Mesh" + floatingPointTypeSuffix<_Real>();
}

template<size_t _Dimension>
std::string
getElasticityTensorName()
{
    return "ElasticityTensor" + std::to_string(_Dimension) + "D";
}

////////////////////////////////////////////////////////////////////////////////
// More convenient unified interface based on template spcialization.
// Use forward declarations of the templates for which we specialize to avoid
// increasing compilation time.
////////////////////////////////////////////////////////////////////////////////
template<typename T>
struct NameMangler;

// FEMMesh
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
class FEMMesh;

template<size_t _K, size_t _Degree, class _EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
struct NameMangler<FEMMesh<_K, _Degree, _EmbeddingSpace, _FEMData>> {
    static std::string name() {
        return getFEMName<_K, _Degree, _EmbeddingSpace>() + "Mesh" + floatingPointTypeSuffix<typename _EmbeddingSpace::Scalar>();
    }
};

// ElasticityTensor
template<typename _Real, size_t _Dim, bool _MajorSymmetry>
class ElasticityTensor;

template<typename _Real, size_t _Dim, bool _MajorSymmetry>
struct NameMangler<ElasticityTensor<_Real, _Dim, _MajorSymmetry>> {
    static std::string name() {
        return getElasticityTensorName<_Dim>() + floatingPointTypeSuffix<_Real>();
    }
};

// SymmetricMatrix
template<size_t t_N, typename Storage>
class SymmetricMatrix;

template<size_t t_N, typename Storage>
struct NameMangler<SymmetricMatrix<t_N, Storage>> {
    static std::string name() {
        return "SymmetricMatrix" + std::to_string(t_N) + "D" +  floatingPointTypeSuffix<typename Storage::Scalar>();
    }
};

#endif /* end of include guard: NAME_MANGLING_HH */
