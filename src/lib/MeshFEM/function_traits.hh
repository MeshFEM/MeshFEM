#ifndef FUNCTION_TRAITS_HH
#define FUNCTION_TRAITS_HH
#include <tuple>

////////////////////////////////////////////////////////////////////////////////
/*! Generalization of function_traits to work with lambda/functors. Allows
//  compiletime introspection of return type, arity, and argument type.
//  From:
//  http://stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda
//  Expanded to work with references to functions and mutable lambdas.
*///////////////////////////////////////////////////////////////////////////////
// For generic types, directly use the result of the signature of its 'operator()'
template <typename T>
struct function_traits
    : public function_traits<decltype(&T::operator())>
{};

// Strip off references
template <typename T>
struct function_traits<T &> : public function_traits<T> {};

// Strip off const (to work with const and non-const lambdas)
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
    : public function_traits<ReturnType(ClassType::*)(Args...)> {};

// specialize for pointers to member function
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...)>
{
    // arity is the number of arguments.
    enum { arity = sizeof...(Args) };

    typedef ReturnType result_type;

    template <size_t i>
    struct arg
    {
        typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
        // the i-th argument is equivalent to the i-th tuple element of a tuple
        // composed of those arguments.
    };
};

// Convenience metafunction to get F's return type.
template<typename F>
using return_type = typename function_traits<F>::result_type;

#endif /* end of include guard: FUNCTION_TRAITS_HH */
