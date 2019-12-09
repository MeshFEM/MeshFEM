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

#endif /* end of include guard: NAME_MANGLING_HH */
