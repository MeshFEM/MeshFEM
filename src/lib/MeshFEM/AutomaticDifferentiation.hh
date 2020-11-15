#ifndef AUTOMATICDIFFERENTIATION_HH
#define AUTOMATICDIFFERENTIATION_HH

#include "Types.hh"
#include <unsupported/Eigen/AutoDiff>

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
constexpr bool isAutoDiffType() {
    return !std::is_same<typename StripAutoDiffImpl<T>::result_type, T>::value;
}

template<typename T>
constexpr bool isAutoDiffType(const T &) { return isAutoDiffType<T>(); }

template<typename T>
std::string autodiffOrNotString() {
    return isAutoDiffType<T>() ? "ADReal" : "Real";
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
    return AutodiffCastImpl<isAutoDiffType<TNew>()>::template run<TNew>(orig);
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

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////
// If "val" does not yet have a nonempty vector of derivatives, initialize it to
// the appropriately sized vector of zeros.
template<typename T>
typename std::enable_if<isAutoDiffType<T>(), void>::type
zeroInitializeEmptyDerivatives(T &val, const size_t numVars) {
    auto &der = val.derivatives();
    if (der.rows() == 0) der.setZero(numVars);
    assert(der.rows() == numVars);
}

template<typename T>
typename std::enable_if<!isAutoDiffType<T>(), void>::type
zeroInitializeEmptyDerivatives(T &/* val */, const size_t /* numVars */) { }

template<typename T>
typename std::enable_if<isAutoDiffType<T>(), size_t>::type
derivativeSize(const T &val) { return val.derivatives().size(); }

template<typename T>
typename std::enable_if<!isAutoDiffType<T>(), size_t>::type
derivativeSize(const T &) { return 0; }

inline VecX_T<Real> extractDirectionalDerivative(const VecX_T<ADReal> &a) {
    const int n = a.size();
    VecX_T<Real> result(n);
    for (int i = 0; i < n; ++i)
        result[i] = a[i].derivatives()[0];
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Extra autodiff math functions
// Note: Eigen 3.3 changes how functions are declared and now finally implements
// tanh itself
////////////////////////////////////////////////////////////////////////////////
namespace Eigen {
#if !EIGEN_VERSION_AT_LEAST(3,3,0)
    template<typename NewDerType>
    inline AutoDiffScalar<NewDerType> MakeAutoDiffScalar(const typename NewDerType::Scalar& value, const NewDerType &der) {
      return AutoDiffScalar<NewDerType>(value,der);
    }

    #define EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(FUNC,CODE) \
    template<typename DerType> \
    inline const Eigen::AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_multiple_op<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar>, const typename Eigen::internal::remove_all<DerType>::type> > \
    FUNC(const Eigen::AutoDiffScalar<DerType>& x) { \
        using namespace Eigen; \
        typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar; \
        CODE; \
    }

    // Implement tanh for eigen autodiff
    EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(tanh,
      using std::tanh;
      using std::cosh;
      using numext::abs2;
      return MakeAutoDiffScalar(tanh(x.value()),
                        x.derivatives() * (Scalar(1)/abs2(cosh(x.value()))));
    )
#else // EIGEN Version < 3.3
    #define EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(FUNC,CODE) \
    template<typename DerType> \
    inline const Eigen::AutoDiffScalar< \
    EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(typename Eigen::internal::remove_all<DerType>::type, typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar, product) > \
    FUNC(const Eigen::AutoDiffScalar<DerType>& x) { \
        using namespace Eigen; \
        EIGEN_UNUSED typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar; \
        CODE; \
    }
#endif // Eigen Version Check

    // Implement log(cosh(x)) with derivative; useful for stable exp_smin computation
    EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(log_cosh,
        Scalar val = std::log(std::cosh(x.value()));
        return MakeAutoDiffScalar(val,
            x.derivatives() * std::tanh(x.value()));
    )
    #undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY

    // Eigen doesn't provide pow for autodiff-typed power.
    template<typename _DerType1, typename _DerType2>
    Eigen::AutoDiffScalar<typename Eigen::internal::remove_all<_DerType1>::type>
    pow(const Eigen::AutoDiffScalar<_DerType1> &x,
        const Eigen::AutoDiffScalar<_DerType2> &p)
    {
        using DerType1 = typename Eigen::internal::remove_all<_DerType1>::type;
        using DerType2 = typename Eigen::internal::remove_all<_DerType2>::type;
        using Scalar  = typename Eigen::internal::traits<DerType1>::Scalar;
        using Scalar2 = typename Eigen::internal::traits<DerType2>::Scalar;
        static_assert(std::is_same<Scalar, Scalar2>::value, "Scalar types must be same for base and exponent");

        if (x.value() < 0) throw std::runtime_error("Pow called with negative base");
        // Avoid numerical issues with zero base: derivative wrt p should be
        // zero but will be NaN.
        Scalar safe_logx = std::log(
                std::max(x.value(),
                         safe_numeric_limits<Scalar>::min() // minimum positive normalized value
        ));

        // Note: make_coherent const-casts the derivatives.
        internal::make_coherent(x.derivatives(), p.derivatives());

        return AutoDiffScalar<DerType1>(std::pow(x.value(), p.value()),
                std::pow(x.value(), p.value() - 1.0) * (
                    x.derivatives() * p.value() +
                    p.derivatives() * x.value() * safe_logx
                ));
    }
}

// Implement log_cosh for non-autodiff types.
template<typename T>
typename std::enable_if<!isAutoDiffType<T>(), T>::type
log_cosh(const T val) {
    return log(cosh(val));
}

////////////////////////////////////////////////////////////////////////////////
// Derivative debugging
////////////////////////////////////////////////////////////////////////////////
// Check for Inf/NaN in derivative fields
template<typename T>
typename std::enable_if<isAutoDiffType<T>(), bool>::type
hasInvalidDerivatives(const T &val) {
    const auto &der = val.derivatives();
    for (int i = 0; i < der.rows(); ++i)
        if (std::isnan(der[i]) || std::isinf(der[i])) return true;
    return false;
}

// Return false for non-autodiff types.
template<typename T>
typename std::enable_if<!isAutoDiffType<T>(), bool>::type
hasInvalidDerivatives(const T &/* val */) { return false; }

template<typename T>
typename std::enable_if<isAutoDiffType<T>(), void>::type
reportDerivatives(std::ostream &os, const T &val) {
    auto prec = os.precision(5);
    os << val.derivatives().transpose();
    os.precision(prec);
}

// do nothing for non-autodiff types.
template<typename T>
typename std::enable_if<!isAutoDiffType<T>(), void>::type
reportDerivatives(std::ostream &/* os */, const T &/* val */) { }

#endif /* end of include guard: AUTOMATICDIFFERENTIATION_HH */
