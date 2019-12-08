// Types not supporting sampling.
template<class T, typename>
struct SampleImpl {
    using ResultType = SValue; // Dummy needed for MultipointSampleImpl below
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const T &/* val */, const ESample &, size_t /* meshDeg */, size_t /* meshDim */) { throw std::runtime_error("Invalid operand for sample"); }
};

// Plain interpolant types
template<class _PointValue>
struct SampleImpl<InterpolantValue<_PointValue>> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const InterpolantValue<_PointValue> &/* val */, const ESample &, size_t /* meshDeg */, size_t /* meshDim */) { throw std::runtime_error("Single interpolant sampling not yet implemented."); }
};

// Nodal and per-element fields (not InterpolantValue fields)
template<class _PointValue>
struct SampleImpl<FieldValue<_PointValue>, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const FieldValue<_PointValue> &val, const ESample &s, size_t meshDeg, size_t meshDim) {
        if (val.domainType == DomainType::PER_NODE) {
            return std::make_unique<ResultType>(nodalFieldElementInterpolant(meshDeg, meshDim, val.value, s.nidx).sampleBarycentric(s.baryCoords));
        }
        else if (val.domainType == DomainType::PER_ELEMENT) { 
            if (s.eidx >= val.value.size()) throw std::runtime_error("Sampling index into per-element field out of bounds.");
            return std::make_unique<ResultType>(val.value[s.eidx]);
        }
        else { throw std::runtime_error("Invalid domain type for sampling: " + std::to_string((int) val.domainType)); };
    }
};

// Interpolant fields (per-element interpolant)
template<class _PointValue>
struct SampleImpl<FieldValue<InterpolantValue<_PointValue>>, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const FieldValue<InterpolantValue<_PointValue>> &val, const ESample &s, size_t /* meshDeg */, size_t /* meshDim */) {
        if (val.domainType == DomainType::PER_ELEMENT) { 
            if (s.eidx >= val.value.size()) throw std::runtime_error("Sampling index into per-element field out of bounds.");
            return std::make_unique<ResultType>(val.value[s.eidx].sampleBarycentric(s.baryCoords));
        }
        else { throw std::runtime_error("Invalid domain type for interpolant field sampling: " + std::to_string((int) val.domainType)); };
    }
};

template<class T>
struct MultipointSampleImpl {
    using ResultType = FieldValue<typename SampleImpl<T>::ResultType>;
    static UVPtr run(const T &val, const std::vector<ESample> &ss, size_t /* meshDeg */, size_t meshDim, DomainType dt) {
        auto result = std::make_unique<ResultType>(dt, ss.size());
        for (size_t i = 0; i < ss.size(); ++i)
            (*result)[i] = std::move(*SampleImpl<T>::run(val, ss[i], meshDim, meshDim));
        return result;
    }
};
