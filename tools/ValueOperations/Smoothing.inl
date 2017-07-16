#include <limits>
template<class _Val>
struct SmoothedElmFldImpl {
    using ResultType = _Val; // Dummy
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const _Val &/* val */, const std::vector<MeshIO::IOElement> &/* elems */, size_t /* meshDeg */, size_t /* meshDim */,
                     const std::vector<Real> &/* volumes */, const MeshConnectivity &/* conn */) {
        throw std::runtime_error("smoothedElementField only acts on FieldValues");
    }
};

template<class _PointValue>
struct SmoothedElmFldImpl<FieldValue<_PointValue>> {
    using ElmAverager = ElmAvgImpl<FieldValue<_PointValue>>;
    using ResultType = typename ElmAverager::ResultType;
    using URPtr = std::unique_ptr<ResultType>;
    static URPtr run(const FieldValue<_PointValue> &val, const std::vector<MeshIO::IOElement> &elems, size_t meshDeg, size_t meshDim,
                     const std::vector<Real> &volumes, const MeshConnectivity &conn) {
        // Convert/clone to pw const element field (could skip with specialization if already pw const)
        URPtr orig = ElmAverager::run(val, elems, meshDeg, meshDim);
        URPtr result(dynamic_cast<ResultType *>(orig->clone().release())); // dynamic_cast only for error checking; should always succeed.
        assert(result);
        for (size_t i = 0; i < elems.size(); ++i) {
            Real totVol = volumes.at(i);
            (*result)[i] *= totVol;
            for (size_t j = 0; j < conn.numElemNeighbors(i); ++j) {
                size_t n = conn.elemNeighbor(i, j);
                if (n == std::numeric_limits<size_t>::max()) continue; // no neighbor
                auto nval = (*orig)[n];
                Real nvol = volumes.at(n);
                nval *= nvol;
                (*result)[i] += nval, totVol += nvol;
            }
            (*result)[i] *= 1.0 / totVol;
        }

        return result;
    }
};
