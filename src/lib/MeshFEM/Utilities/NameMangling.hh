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
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

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

template<typename _Energy, size_t _K, size_t _Degree, class _EmbeddingSpace>
std::string getElasticSolidName() {
    return getFEMName<_K, _Degree, _EmbeddingSpace>() + _Energy::name() + "ElasticSolid";
}

template<typename _Energy>
std::string getEnergyName() {
    return _Energy::name() + std::to_string(_Energy::Dimension) + "D";
}

template<size_t _Dimension>
std::string
getElasticityTensorName()
{
    return "ElasticityTensor" + std::to_string(_Dimension) + "D";
}

////////////////////////////////////////////////////////////////////////////////
// More convenient unified interface based on template spcialization.
////////////////////////////////////////////////////////////////////////////////
template<typename T>
struct NameMangler;

#include <MeshFEM/FEMMesh.hh> // TODO: try forward declaration
template<size_t _K, size_t _Degree, class _EmbeddingSpace>
struct NameMangler<FEMMesh<_K, _Degree, _EmbeddingSpace>> {
    static std::string name() {
        return getFEMName<_K, _Degree, _EmbeddingSpace>() + "Mesh" + floatingPointTypeSuffix<typename _EmbeddingSpace::Scalar>();
    }
};

#include <MeshFEM/ElasticSolid.hh>
template<typename _Energy, size_t _K, size_t _Degree, class _EmbeddingSpace>
struct NameMangler<ElasticSolid<_K, _Degree, _EmbeddingSpace, _Energy>> {
    static std::string name() {
        return getElasticSolidName<_Energy, _K, _Degree, _EmbeddingSpace>();
    }
};

#include <MeshFEM/ElasticSheet.hh>
template<class Psi_C>
struct NameMangler<ElasticSheet<Psi_C>> {
    static std::string name() {
        return "ElasticSheet" + getEnergyName<Psi_C>();
    }
};

template<typename _Real, size_t _Dim, bool _MajorSymmetry>
struct NameMangler<ElasticityTensor<_Real, _Dim, _MajorSymmetry>> {
    static std::string name() {
        return getElasticityTensorName<_Dim>() + floatingPointTypeSuffix<_Real>();
    }
};

template<typename _Real, size_t _N>
struct NameMangler<SymmetricMatrixValue<_Real, _N>> {
    static std::string name() {
        return "SymmetricMatrix" + std::to_string(_N) + "D" +  floatingPointTypeSuffix<_Real>();
    }
};

#endif /* ecnd of include guard: NAME_MANGLING_HH */
