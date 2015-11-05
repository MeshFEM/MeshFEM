#ifndef TEMPLATE_HACKS_HH
#define TEMPLATE_HACKS_HH
#include <type_traits>

template<bool b> using static_not = std::integral_constant<bool, !b>;

// Cool trick for checking if all booleans in a boolean parameter pack are true
template <bool...> struct bool_pack;
template <bool... v> using all_true  = std::is_same<bool_pack<true, v...>, bool_pack<v..., true>>;
template <bool... v> using all_false = std::is_same<bool_pack<false, v...>, bool_pack<v..., false>>;
template <bool... v> using any_true  = static_not<std::is_same<bool_pack<true, v...>, bool_pack<v..., true>>::value>;
template <bool... v> using any_false = static_not<std::is_same<bool_pack<false, v...>, bool_pack<v..., false>>::value>;

// Are all types of the parameters passed integral?
template <typename... T>
constexpr bool all_integer_parameters() { return all_true<std::is_integral<T>::value...>::value; }

#endif /* end of include guard: TEMPLATE_HACKS_HH */
