#ifndef TEMPLATENAME_HH
#define TEMPLATENAME_HH

#include <type_traits>
#include "../EnergyDensities/LinearElasticEnergy.hh"
#include "../EnergyDensities/NeoHookeanEnergy.hh"

enum class EnergyType
{
    NEO_HOOKEAN,
    LINEAR
};

template<typename _Energy>
struct EnergyTraits
{
};

template<typename _Real, size_t _Dimension>
struct EnergyTraits<NeoHookeanEnergy<_Real, _Dimension>>
{
    static constexpr EnergyType type_v = EnergyType::NEO_HOOKEAN;
};

template<typename _Real, size_t _Dimension>
struct EnergyTraits<LinearElasticEnergy<_Real, _Dimension>>
{
    static constexpr EnergyType type_v = EnergyType::LINEAR;
};

template<typename _Energy>
struct isLinearElastic
  : public std::integral_constant<bool, EnergyTraits<_Energy>::type_v == EnergyType::LINEAR>
{
};

template<typename _Energy>
struct isNeoHookean
  : public std::integral_constant<bool, EnergyTraits<_Energy>::type_v == EnergyType::NEO_HOOKEAN>
{
};

template<typename _Energy>
std::string
getEnergyName()
{
    if (isLinearElastic<_Energy>::value)
    {
        return "LinearElastic";
    }
    if (isNeoHookean<_Energy>::value)
    {
        return "NeoHookean";
    }
    return "NotImplemented";
}

template<size_t _Degree>
std::string
getFEMName()
{
    switch (_Degree)
    {
        case 1:
            return "Linear";
        case 2:
            return "Quadratic";
        default:
            return "NotImplemented";
    }
}

template<size_t _Dimension, size_t _Degree>
std::string
getMeshTypeName()
{
    return getFEMName<_Degree>() + "FEM" + std::to_string(_Dimension) + "D";
}

template<typename _Energy, size_t _Dimension, size_t _Degree>
std::string
getElasticStructureTypeName()
{
    return getMeshTypeName<_Dimension, _Degree>() + getEnergyName<_Energy>();
}

#endif

