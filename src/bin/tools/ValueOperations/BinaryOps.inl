////////////////////////////////////////////////////////////////////////////////
// BinaryOp.inl
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements componetwise binary ops on values. To be included by Value.hh
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/28/2015 16:00:45
////////////////////////////////////////////////////////////////////////////////
struct AddOp : public BinaryOp { virtual Real operator()(Real a, Real b) const { return a + b; } };
struct SubOp : public BinaryOp { virtual Real operator()(Real a, Real b) const { return a - b; } };
struct MulOp : public BinaryOp { virtual Real operator()(Real a, Real b) const { return a * b; } };
struct DivOp : public BinaryOp { virtual Real operator()(Real a, Real b) const { return a / b; } };

// Default implementation--only used if actual implementation not found.
// Last template parameter for SFINAE
template<class T1, class T2, typename = void>
struct BinaryOpImpl {
    using ResultType = T1; // ResultType can be arbitrary since actual results aren't generated.
    static std::unique_ptr<ResultType> apply(const BinaryOp &/* op */, T1, T2) {
        throw std::runtime_error(std::string("Illegal binary operation between ") + typeid(T1).name() + " and " + typeid(T2).name());
    }
};

// Field a, Field b: Pointwise suboperation
template<class _VType1, class _VType2>
struct BinaryOpImpl<FieldValue<_VType1>, FieldValue<_VType2>> {
    using SubImpl    = BinaryOpImpl<_VType1, _VType2>;
    using ResultType = FieldValue<typename SubImpl::ResultType>;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const FieldValue<_VType1> &a, const FieldValue<_VType2> &b) {
        size_t nElems = a.size();
        if ((a.domainType != b.domainType) || (nElems != b.size()))
            throw std::runtime_error("Binary operation field domain mismatch.");
        auto result = std::make_unique<ResultType>(a.domainType, nElems);
        for (size_t i = 0; i < nElems; ++i)
            (*result)[i] = *SubImpl::apply(op, a[i], b[i]);
        return result;
    }
};

// Field a, Value Type b: Suboperation between value at each point of a and value b
template<class _VType1, class _VType2>
struct BinaryOpImpl<FieldValue<_VType1>, _VType2> {
    using SubImpl = BinaryOpImpl<_VType1, _VType2>;
    using ResultType = FieldValue<typename SubImpl::ResultType>;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const FieldValue<_VType1> &a, const _VType2 &b) {
        size_t nElems = a.size();
        auto result = std::make_unique<ResultType>(a.domainType, nElems);
        for (size_t i = 0; i < nElems; ++i)
            (*result)[i] = *SubImpl::apply(op, a[i], b);
        return result;
    }
};

// Value Type a, Field b: Suboperation between value a and value at each point of b
template<class _VType1, class _VType2>
struct BinaryOpImpl<_VType1, FieldValue<_VType2>> {
    using SubImpl = BinaryOpImpl<_VType1, _VType2>;
    using ResultType = FieldValue<typename SubImpl::ResultType>;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const _VType1 &a, const FieldValue<_VType2> &b) {
        size_t nElems = b.size();
        auto result = std::make_unique<ResultType>(b.domainType, nElems);
        for (size_t i = 0; i < nElems; ++i)
            (*result)[i] = *SubImpl::apply(op, a, b[i]);
        return result;
    }
};

// Interpolant a, Interpolant b: Suboperation between each nodal value
template<class _PointValue1, class _PointValue2>
struct BinaryOpImpl<InterpolantValue<_PointValue1>, InterpolantValue<_PointValue2>> {
    using SubImpl = BinaryOpImpl<_PointValue1, _PointValue2>;
    using ResultType = InterpolantValue<typename SubImpl::ResultType>;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const InterpolantValue<_PointValue1> &a, const InterpolantValue<_PointValue2> &b) {
        if (a.dim() != b.dim()) throw std::runtime_error("Binary operation interpolant size mismatch.");
        auto result = std::make_unique<ResultType>(a.simplexDimension());
        for (size_t i = 0; i < a.dim(); ++i)
            result->value[i] = *SubImpl::apply(op, a.value[i], b.value[i]);
        return result;
    }
};

// Interpolant a, Point Value b: Suboperation between each nodal value of a and value b
// (Disabled for Field/Interpolant second operand to prevent ambiguity)
template<class _PointValue1, class _PointValue2>
struct BinaryOpImpl<InterpolantValue<_PointValue1>, _PointValue2, typename enable_if_point_value<_PointValue2>::type> {
    using SubImpl = BinaryOpImpl<_PointValue1, _PointValue2>;
    using ResultType = InterpolantValue<typename SubImpl::ResultType>;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const InterpolantValue<_PointValue1> &a, const _PointValue2 &b) {
        auto result = std::make_unique<ResultType>(a.simplexDimension());
        for (size_t i = 0; i < a.dim(); ++i)
            result->value[i] = *SubImpl::apply(op, a.value[i], b);
        return result;
    }
};

// Point Value a, Interpolant b: Suboperation between value a and each nodal value of b
// (Disabled for Field/Interpolant first operand to prevent ambiguity)
template<class _PointValue1, class _PointValue2>
struct BinaryOpImpl<_PointValue1, InterpolantValue<_PointValue2>, typename enable_if_point_value<_PointValue1>::type> {
    using SubImpl = BinaryOpImpl<_PointValue1, _PointValue2>;
    using ResultType = InterpolantValue<typename SubImpl::ResultType>;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const _PointValue1 &a, const InterpolantValue<_PointValue2> &b) {
        auto result = std::make_unique<ResultType>(b.simplexDimension());
        for (size_t i = 0; i < b.dim(); ++i)
            result->value[i] = *SubImpl::apply(op, a, b.value[i]);
        return result;
    }
};

// Point Value a, Point Value b: Suboperation between value a and value b
// (SFINAE to restrict to subclasses of PointValue)
template<class _PointValue>
struct BinaryOpImpl<_PointValue, _PointValue, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const _PointValue &a, const _PointValue &b) {
        if (a.dim() != b.dim()) throw std::runtime_error("Binary operation dimension mismatch.");
        auto result = std::make_unique<ResultType>(a.N());
        for (size_t i = 0; i < a.dim(); ++i)
            (*result)[i] = op(a[i], b[i]);
        return result;
    }
};

// Scalar Value a, Point Value b
template<class _PointValue>
struct BinaryOpImpl<SValue, _PointValue, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const SValue &a, const _PointValue &b) {
        auto result = std::make_unique<ResultType>(b.N());
        for (size_t i = 0; i < b.dim(); ++i)
            (*result)[i] = op(a.value, b[i]);
        return result;
    }
};

// Point Value a, Scalar Value b
template<class _PointValue>
struct BinaryOpImpl<_PointValue, SValue, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const _PointValue &a, const SValue &b) {
        auto result = std::make_unique<ResultType>(a.N());
        for (size_t i = 0; i < a.dim(); ++i)
            (*result)[i] = op(a[i], b.value);
        return result;
    }
};

// Scalar Value a, Scalar Value b
template<>
struct BinaryOpImpl<SValue, SValue> {
    using ResultType = SValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr apply(const BinaryOp &op, const SValue &a, const SValue &b) {
        auto result = std::make_unique<ResultType>();
        result->value = op(a.value, b.value);
        return result;
    }
};

// Implement the inner stage of double-dispatch for component-wise binary
// operations. The outer dispatch level is handled by vtable lookup, so we only
// need to implement this inner lookup.
// Do this with a sequence of automatically generated dynamic cast type checks.
template<class T1>
[[ noreturn ]] UVPtr dispatchCWiseBinaryOpImpl(const BinaryOp &/* op */, const T1 &, CVPtr b) { throw std::runtime_error("Unknown dynamic type: " + std::string(typeid(*b).name())); }
template<class T1, class TCheck, class... Args>
UVPtr dispatchCWiseBinaryOpImpl(const BinaryOp &op, const T1 &a, CVPtr b) {
    if (auto val = dynamic_cast<const TCheck *>(b)) return BinaryOpImpl<T1, TCheck>::apply(op, a, *val);
    return dispatchCWiseBinaryOpImpl<T1, Args...>(op, a, b);
}
template<class T1>
UVPtr dispatchCWiseBinaryOp(const BinaryOp &op, const T1 &a, CVPtr b) {
    return dispatchCWiseBinaryOpImpl<T1,   SValue,   VValue,   SMValue,
                                          ISValue,  IVValue,  ISMValue,
                                          FSValue,  FVValue,  FSMValue,
                                         FISValue, FIVValue, FISMValue>(op, a, b);
}
