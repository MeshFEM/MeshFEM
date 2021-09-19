////////////////////////////////////////////////////////////////////////////////
// FieldPostProcessing.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Postprocessing for, e.g., stress/strin fields.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/14/2020 16:14:03
////////////////////////////////////////////////////////////////////////////////
#ifndef FIELDPOSTPROCESSING_HH
#define FIELDPOSTPROCESSING_HH

#include <vector>

// Take a piecewise smooth but discontinuous interpolant field (e.g., stress)
// defined by a function `f(ei, x)`, where `ei` is an element index and `x` is
// a barycentric coordinate vector within the element and construct a C0
// piecewise linear field by averaging the distinct element corner values
// overlapping at a vertex.
// We use a simple volume-weighted averaging rule. With a more careful handling
// of, e.g., material interfaces, the resulting stress field should have a
// higher order of accuracy than the unprocessed field ("superconvergence").
template<class FEMMesh_, class F>
std::vector<return_type<F>> vertexAveragedField(const FEMMesh_ &m, const F &f) {
    static constexpr size_t K = FEMMesh_::K;
    using EvalPtK = EvalPt<K>;

    using value_type = return_type<F>;
    std::vector<value_type> result(m.numVertices(), value_type::Zero()); // TODO: make this work for scalar fields too
    std::vector<Real> averagingVolume(m.numVertices());
    EvalPtK x;
    x.fill(0.0);
    for (auto e : m.elements()) {
        Real evol = e->volume();
        for (auto v : e.vertices()) {
            averagingVolume[v.index()] += evol;
            x[v.localIndex()] = 1.0;
            result[v.index()] += evol * f(e.index(), x);
            x[v.localIndex()] = 0.0;
        }
    }
    for (auto v : m.vertices())
        result[v.index()] /= averagingVolume[v.index()];

    return result;
}

#endif /* end of include guard: FIELDPOSTPROCESSING_HH */
