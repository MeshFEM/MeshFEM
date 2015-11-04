// Types not supporting sampling.
template<class T, typename>
struct SampleImpl {
    static UVPtr run(const T &val, const ESample &s, size_t meshDeg, size_t meshDim) { throw std::runtime_error("Invalid operand for sample"); }
};

// Plain interpolant types
template<class _PointValue>
struct SampleImpl<InterpolantValue<_PointValue>> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const InterpolantValue<_PointValue> &val, const ESample &s, size_t meshDeg, size_t meshDim) { throw std::runtime_error("Single interpolant sampling not yet implemented."); }
};

// Construct an interpolant holding the restriction of a piecewise polynomial
// nodal field (with nodal value nvals) to the element indicated by s.
// Assumes the nodal field originates from the msh file and therefore has the
// same dimension/degree as the mesh.
// Also, we assume elements are of full dimension (K == N)
template<class PointValueType, size_t _N, size_t _Deg>
InterpolantValue<PointValueType> nodalFieldElementInterpolantImpl(const std::vector<PointValueType> &nvals, const ESample &s) {
    const size_t nnodes = Simplex::numNodes(_N, _Deg);
    if (s.nidx.size() != nnodes) throw std::logic_error("Invalid sample element size");
    for (size_t n : s.nidx) { if (n >= nvals.size()) throw std::runtime_error("A sampling element node index is outside the nodal field bounds."); }

    Interpolant<typename PointValueType::raw_type, _N, _Deg> rawInterp;
    for (size_t i = 0; i < nnodes; ++i)
        rawInterp[i] = nvals[s.nidx[i]].value;
    return InterpolantValue<PointValueType>(rawInterp); // Note: upscales for _Deg = 1
}

template<class PointValueType>
InterpolantValue<PointValueType> nodalFieldElementInterpolant(size_t meshDeg, size_t meshDim, const std::vector<PointValueType> &nvals, const ESample &s) {
    typedef InterpolantValue<PointValueType> (*ImplPtr)(const std::vector<PointValueType> &, const ESample &);
    static std::map<std::pair<size_t, size_t>, ImplPtr> impl =
       {{{2, 1}, &nodalFieldElementInterpolantImpl<PointValueType, 2, 1>},
        {{2, 2}, &nodalFieldElementInterpolantImpl<PointValueType, 2, 2>},
        {{3, 1}, &nodalFieldElementInterpolantImpl<PointValueType, 3, 1>},
        {{3, 2}, &nodalFieldElementInterpolantImpl<PointValueType, 3, 2>}};
    auto key = std::make_pair(meshDim, meshDeg);
    if (impl.count(key) == 0) {
        throw std::runtime_error("Unsupported mesh degree (" + std::to_string(meshDeg) +
                                 ") or dimension (" + std::to_string(meshDim) +
                                 ") for nodal field element interpolant construction.");
    }
    return impl.at(key)(nvals, s);
}

// Nodal Fields (not InterpolantValue fields)
template<class _PointValue>
struct SampleImpl<FieldValue<_PointValue>, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const FieldValue<_PointValue> &val, const ESample &s, size_t meshDeg, size_t meshDim) {
        if (val.domainType == DomainType::PER_NODE) {
            return std::make_unique<ResultType>(nodalFieldElementInterpolant(meshDeg, meshDim, val.value, s).sampleBarycentric(s.baryCoords));
        }
        else if (val.domainType == DomainType::PER_ELEMENT) { 
            if (s.eidx >= val.value.size()) throw std::runtime_error("Sampling index into per-element field out of bounds.");
            return std::make_unique<ResultType>(val.value[s.eidx]);
        }
        else { throw std::runtime_error("Invalid domain type for sampling: " + std::to_string((int) val.domainType)); };
    }
};

// Interpolant Fields
template<class _PointValue>
struct SampleImpl<FieldValue<InterpolantValue<_PointValue>>, typename enable_if_point_value<_PointValue>::type> {
    using ResultType = _PointValue;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const FieldValue<InterpolantValue<_PointValue>> &val, const ESample &s, size_t meshDeg, size_t meshDim) {
        if (val.domainType == DomainType::PER_ELEMENT) { 
            if (s.eidx >= val.value.size()) throw std::runtime_error("Sampling index into per-element field out of bounds.");
            return std::make_unique<ResultType>(val.value[s.eidx].sampleBarycentric(s.baryCoords));
        }
        else { throw std::runtime_error("Invalid domain type for interpolant field sampling: " + std::to_string((int) val.domainType)); };
    }
};
