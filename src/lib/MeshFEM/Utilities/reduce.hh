////////////////////////////////////////////////////////////////////////////////
// reduce.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Reduce an indexed collection of terms to a single value using a binary
//  operator (e.g. + or *).
//  Terms can be ommitted using the "SkipPredicate" (defaults to no skipping).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  10/03/2017 23:28:00
////////////////////////////////////////////////////////////////////////////////
#ifndef REDUCE_HH
#define REDUCE_HH

#include <utility>

// Generate a "SkipPredicate" for skipping the indices in list "Idxs"
template<size_t... Idxs>
struct SkipIndices {
    // Base case: no indices (used if the specialization below doesn't match)
    template<size_t j>
    struct Predicate { static constexpr bool value = false; };
};

template<size_t i, size_t... Idxs>
struct SkipIndices<i, Idxs...> {
    // Recursive case: we should skip if j matches i, or any of the indices in Idxs...
    template<size_t j>
    struct Predicate {
        static constexpr bool value = (i == j) ||
            SkipIndices<Idxs...>::template Predicate<j>::value;
    };
};

template<size_t j> using NoSkip = SkipIndices<>::Predicate<j>;

// Run Job::term<j>(args) for j in i..endIndex-1, reducing the results using
// Job::BinaryOp (like applying Python's reduce() to the list [Job::term<0>(args), Job::term<1>(args)...]).
// Terms for which SkipPredicate<j>::value == true are skipped.
// This struct implements the recursive case:
//  apply Job::BinaryOp to the first term and the reduction of the rest.
template<size_t endIndex, size_t i, class Job, template<size_t> class SkipPredicate = NoSkip>
struct Reduce {
    using result_type = decltype(Job::initializer);

    template<typename... Args>
    static result_type run(Args&&... args) {
        auto rest = Reduce<endIndex, i + 1, Job, SkipPredicate>::run(std::forward<Args>(args)...);
        if (SkipPredicate<i>::value) return rest;
        return typename Job::BinaryOp()(Job::template term<i>(std::forward<Args>(args)...), rest);
    }
};

// Base case: empty range (return initializer)
template<size_t endIndex, class Job, template<size_t> class SkipPredicate>
struct Reduce<endIndex, endIndex, Job, SkipPredicate> {
    using result_type = decltype(Job::initializer);
    template<typename... Args>
    static result_type run(Args&&... /* args */) { return Job::initializer; }
};

#endif /* end of include guard: REDUCE_HH */
