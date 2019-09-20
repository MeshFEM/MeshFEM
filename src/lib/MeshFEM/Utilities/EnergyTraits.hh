#ifndef ENERGYTRAITS_HH
#define ENERGYTRAITS_HH

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

#endif /* end of include guard: ENERGYTRAITS_HH */
