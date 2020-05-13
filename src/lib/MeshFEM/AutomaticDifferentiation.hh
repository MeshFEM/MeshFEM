#ifndef AUTOMATICDIFFERENTIATION_HH
#define AUTOMATICDIFFERENTIATION_HH

#include <unsupported/Eigen/AutoDiff>

#include "Types.hh"

using ADReal = Eigen::AutoDiffScalar<Eigen::Matrix<Real, 1, 1>>;

// Wrapper to get the underlying value of an autodiff type (or do nothing for
// primitive types)
template<typename T>
struct StripAutoDiffImpl {
    using result_type = T;
    static result_type run(const T &v) { return v; }
};

template<typename _DerType>
struct StripAutoDiffImpl<Eigen::AutoDiffScalar<_DerType>> {
    using result_type = typename Eigen::internal::traits<typename Eigen::internal::remove_all<_DerType>::type>::Scalar;
    static result_type run(const Eigen::AutoDiffScalar<_DerType> &v) { return v.value(); }
};

// Cast autodiff vectors/matrices to plain vectors/matrices.
template<typename _DerType, int... I>
struct StripAutoDiffImpl<Eigen::Matrix<Eigen::AutoDiffScalar<_DerType>, I...>> {
    using autodiff_type = Eigen::Matrix<Eigen::AutoDiffScalar<_DerType>, I...>;
    using scalar_type = typename Eigen::internal::traits<typename Eigen::internal::remove_all<_DerType>::type>::Scalar;
    using result_type = Eigen::Matrix<scalar_type, I...>;

    static result_type run(const autodiff_type &v) {
        result_type r(v.rows(), v.cols());
        for (int i = 0; i < v.rows(); ++i) {
            for (int j = 0; j < v.cols(); ++j)
                r(i, j) = v(i, j).value();
        }
        return r;
    }
};

// Strip automatic differentiation wrapper from a scalar value type (does
// nothing when applied to a non-autodiff type).
template<typename T>
typename StripAutoDiffImpl<T>::result_type
stripAutoDiff(const T &val) {
    return StripAutoDiffImpl<T>::run(val);
}

template<typename T>
constexpr bool isAutodiffType() {
    return !std::is_same<typename StripAutoDiffImpl<T>::result_type, T>::value;
}

template<typename T>
bool isAutodiffType(const T &/* val */) { return isAutodiffType<T>(); }

template<typename T>
std::string autodiffOrNotString() {
    return isAutodiffType<T>() ? "ADReal" : "Real";
}

// For casting to non autodiff types, we must strip
template<bool IsAutodiffTarget>
struct AutodiffCastImpl {
    template<typename TNew, typename TOrig>
    static TNew run(const TOrig &val) { return TNew(stripAutoDiff(val)); }
};

template<>
struct AutodiffCastImpl<true> {
    // Direct casting only works for scalar values.
    template<typename TNew, typename TOrig>
    static typename std::enable_if<std::is_arithmetic<typename StripAutoDiffImpl<TOrig>::result_type>::value, TNew>::type
    run(const TOrig &val) { return TNew(val); }

    // The only other case we support is Eigen matrices, which must be cast
    // componentwise.
    template<typename TNew, typename OrigDerived>
    static TNew run(const Eigen::MatrixBase<OrigDerived> &val) {
        using Scalar = typename TNew::Scalar;
        return val.template cast<Scalar>();
        // TNew result(val.rows(), val.cols());
        // for (int i = 0; i < val.rows(); ++i) {
        //     for (int j = 0; j < val.cols(); ++j)
        //         result(i, j) = val(i, j);
        // }

        // return result;
    }
};

template<typename TNew, typename TOrig>
TNew autodiffCast(const TOrig &orig) {
    return AutodiffCastImpl<isAutodiffType<TNew>()>::template run<TNew>(orig);
}

// std::numeric_limits is dangerous! If you use it on Eigen's autodiff types you
// will get undefined behavior.
template<typename T>
struct safe_numeric_limits
    : public std::numeric_limits<typename StripAutoDiffImpl<T>::result_type>
{
    using NonADType = typename StripAutoDiffImpl<T>::result_type;
    static_assert(std::is_arithmetic<NonADType>::value,
                  "std::numeric_limits is broken for non-arithmetic types!");
};

inline VecX_T<Real> extractDirectionalDerivative(const VecX_T<ADReal> &a) {
    const int n = a.size();
    VecX_T<Real> result(n);
    for (int i = 0; i < n; ++i)
        result[i] = a[i].derivatives()[0];
    return result;
}

#endif /* end of include guard: AUTOMATICDIFFERENTIATION_HH */
