// Types not supporting averaging.
template<class T, typename>
struct ElmAvgImpl { static UVPtr run(const T &/* val */, const std::vector<MeshIO::IOElement> &/* elems */, size_t /* meshDeg */, size_t /* meshDim */) { throw std::runtime_error("Invalid operand for elementAverage"); } };

// Plain interpolant types.
template<class _PointValue>
struct ElmAvgImpl<InterpolantValue<_PointValue>> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const InterpolantValue<_PointValue> &val, const std::vector<MeshIO::IOElement> &/* elems */, size_t /* meshDeg */, size_t /* meshDim */) {
        return std::make_unique<ResultType>(val.average());
    }
};

// Nodal Fields (not InterpolantValue fields)
template<class _PointValue>
struct ElmAvgImpl<FieldValue<_PointValue>, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = FieldValue<_PointValue>;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const FieldValue<_PointValue> &val, const std::vector<MeshIO::IOElement> &elems, size_t meshDeg, size_t meshDim) {
        size_t nElems = elems.size();
        if (val.domainType == DomainType::PER_NODE) {
            auto result = std::make_unique<ResultType>(DomainType::PER_ELEMENT, nElems);
            // Could optimize by short-circuiting conversion from raw
            // interpolant to InterpolantValue and by caching
            // nodalFieldElementInterpolantImpl pointer.
            for (size_t i = 0; i < nElems; ++i)
                (*result)[i] = nodalFieldElementInterpolant(meshDeg, meshDim, val.value, elems[i]).average();
            return result;
        }
        else if (val.domainType == DomainType::PER_ELEMENT) { 
            // Already a constant per-element field: just verify size and copy
            if (val.size() != nElems) throw std::runtime_error("Invalid per-element field size");
            return std::make_unique<ResultType>(val);
        }
        else { throw std::runtime_error("Invalid domain type for sampling: " + std::to_string((int) val.domainType)); };
    }
};

// Interpolant Fields->PointValue Fields
template<class _PointValue>
struct ElmAvgImpl<FieldValue<InterpolantValue<_PointValue>>, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = FieldValue<_PointValue>;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const FieldValue<InterpolantValue<_PointValue>> &val, const std::vector<MeshIO::IOElement> &elems, size_t /* meshDeg */, size_t /* meshDim */) {
        size_t nElems = elems.size();
        if (val.domainType == DomainType::PER_ELEMENT) { 
            if (val.size() != nElems) throw std::runtime_error("Invalid interpolant field size");
            auto result = std::make_unique<ResultType>(val.domainType, nElems);
            for (size_t i = 0; i < nElems; ++i) (*result)[i] = val[i].average();
            return result;
        }
        else { throw std::runtime_error("Invalid interpolant field domain type: " + std::to_string((int) val.domainType)); };
    }
};
