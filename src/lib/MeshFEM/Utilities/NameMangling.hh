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
#include <stdexcept>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

namespace MeshFEM {

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

    std::string dimName = std::to_string(int(_EmbeddingSpace::RowsAtCompileTime));

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
    return _Energy::name() + std::to_string(_Energy::N) + "D";
}

inline std::string getElasticityTensorName(size_t N) {
    return "ElasticityTensor" + std::to_string(N) + "D";
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


// ElasticSolid
template<size_t _K, size_t _Deg, class EmbeddingSpace, class _Energy>
struct ElasticSolid;

template<typename _Energy, size_t _K, size_t _Degree, class _EmbeddingSpace>
struct NameMangler<ElasticSolid<_K, _Degree, _EmbeddingSpace, _Energy>> {
    static std::string name() {
        return getElasticSolidName<_Energy, _K, _Degree, _EmbeddingSpace>();
    }
};

// ElasticSheet
template <class _Psi_C>
struct ElasticSheet;

template<class Psi_C>
struct NameMangler<ElasticSheet<Psi_C>> {
    static std::string name() {
        return "ElasticSheet" + getEnergyName<Psi_C>();
    }
};

// ElasticityTensor
template<typename _Real, size_t _Dim, bool _MajorSymmetry>
class ElasticityTensor;

template<typename _Real, size_t _Dim, bool _MajorSymmetry>
struct NameMangler<ElasticityTensor<_Real, _Dim, _MajorSymmetry>> {
    static std::string name() {
        return getElasticityTensorName(_Dim) + floatingPointTypeSuffix<_Real>();
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

////////////////////////////////////////////////////////////////////////////////
// Determine whether a class defines a `name()` method
////////////////////////////////////////////////////////////////////////////////
template<class C, typename = void>
struct has_name_method : std::false_type {};

template<class C>
struct has_name_method<C, std::void_t<decltype(C::name())>> : std::true_type {};

// Hack to get type names without C++26 reflection
#include <type_traits>

template <typename T>
std::string get_name_of_type() {
#if defined(__clang__)
    std::string name = __PRETTY_FUNCTION__;
    auto start = name.find("T = ") + 4;
    auto end = name.find(']', start);
    return name.substr(start, end - start);
#elif defined(__GNUC__)
    std::string name = __PRETTY_FUNCTION__;
    auto start = name.find("with T = ") + 9;
    auto end = name.find(';', start);
    return name.substr(start, end - start);
#elif defined(_MSC_VER)
    std::string name = __FUNCSIG__;
    const std::string marker = "get_name_of_type<";
    auto start = name.find(marker);
    if (start == std::string::npos) throw std::runtime_error("Could not parse __FUNCSIG__");
    start += marker.size();
    auto end = name.rfind(">(void)");
    if ((end == std::string::npos) || (end < start)) throw std::runtime_error("Could not parse __FUNCSIG__");
    std::string result = name.substr(start, end - start);
    for (const char *kw : {"struct ", "class ", "enum ", "union "}) {
        const std::string keyword(kw);
        if (result.compare(0, keyword.size(), keyword) == 0) {
            result.erase(0, keyword.size());
            break;
        }
    }
    return result;
#else
    throw std::runtime_error("Unsupported compiler for type_name");
#endif
}

} // namespace MeshFEM

#endif /* ecnd of include guard: NAME_MANGLING_HH */
