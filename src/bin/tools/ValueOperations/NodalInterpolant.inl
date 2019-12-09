#include <map>

// Construct an interpolant holding the restriction of a piecewise polynomial
// nodal field (with nodal value nvals) to the element with node indices nidx.
// Assumes the nodal field originates from the msh file and therefore has the
// same dimension/degree as the mesh.
// Also, we assume elements are of full dimension (K == N)
template<class PointValueType, size_t _N, size_t _Deg>
InterpolantValue<PointValueType> nodalFieldElementInterpolantImpl(const std::vector<PointValueType> &nvals, const MeshIO::IOElement &nidx) {
    const size_t nnodes = Simplex::numNodes(_N, _Deg);
    if (nidx.size() != nnodes) throw std::logic_error("Invalid sample element size");
    for (size_t n : nidx) { if (n >= nvals.size()) throw std::runtime_error("A sampling element node index is outside the nodal field bounds."); }

    Interpolant<typename PointValueType::raw_type, _N, _Deg> rawInterp;
    for (size_t i = 0; i < nnodes; ++i)
        rawInterp[i] = nvals[nidx[i]].value;
    return InterpolantValue<PointValueType>(rawInterp); // Note: upscales for _Deg = 1
}

template<class PointValueType>
InterpolantValue<PointValueType> nodalFieldElementInterpolant(size_t meshDeg, size_t meshDim, const std::vector<PointValueType> &nvals, const MeshIO::IOElement &nidx) {
    typedef InterpolantValue<PointValueType> (*ImplPtr)(const std::vector<PointValueType> &, const MeshIO::IOElement &);
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
    return impl.at(key)(nvals, nidx);
}
