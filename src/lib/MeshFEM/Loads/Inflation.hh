////////////////////////////////////////////////////////////////////////////////
// Inflation.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A pressure force inflating a closed surface mesh.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/06/2020 10:30:20
////////////////////////////////////////////////////////////////////////////////
#ifndef LOADS_INFLATION_HH
#define LOADS_INFLATION_HH

#include "Load.hh"

namespace Loads {

template<typename T>
struct NeumaierSum {
    NeumaierSum(T val = 0) : sum(val) { }

    void accumulate(T term) {
        T newSum = sum + term;
        if (std::abs(sum) >= std::abs(term))
            c += term  + (sum - newSum); // If sum is bigger, low-order digits of "term" are lost.
        else
            c += sum + (term - newSum);  // Else low-order digits of sum are lost
        sum = newSum;
    }

    T result() {  return sum + c; }

    T c = 0; // roundoff error correction
    T sum = 0;
};

template<class Object>
struct Inflation : public ObjectSpecificLoad<Object> {
    using Base = ObjectSpecificLoad<Object>;
    using Real = typename Base::Real;
    using VXd  = typename Base::VXd;
    using V3d  = Eigen::Matrix<Real, 3, 1>;

    static constexpr size_t N   = 3;
    static constexpr size_t K   = Object::K;
    static constexpr size_t Deg = Object::Deg;
    using Base::getObj;

    static_assert(K == 2, "Inflation loads are only defined for surfaces immersed in 3D");

    Inflation(std::weak_ptr<const Object> obj, Real p = 1.0)
        : Base(obj) { pressure = p; }

    Real pressure = 1.0;

    Real volume() const  {
        NeumaierSum<Real> sum;
        auto &sheet = getObj();
        const size_t ne = sheet.mesh().numElements();
        for (size_t i = 0; i < ne; ++i) {
            sum.accumulate(sheet.getCornerPositions(i).determinant());
        }
        return sum.result() / 6.0;
    }

    virtual Real energy() const override {
        return -volume() * pressure;
    }

    // Gradient with respect to the deformed state
    virtual VXd grad_x() const override {
        const auto &sheet = getObj();
        const auto &m = sheet.mesh();
        const size_t ne = m.numElements();
        VXd result = VXd::Zero(sheet.numVars());
        for (size_t i = 0; i < ne; ++i) {
            const auto &e = sheet.deformedElement(i);
            V3d contrib = (-pressure * e.volume() / 3.0) * e.normal();
            for (const auto v : m.element(i).vertices())
                result.template segment<3>(3 * v.index()) += contrib;
        }
        result.bottomRows(sheet.numVars() - 3 * m.numVertices()).setZero();
        return result;
    }

    // Gradient with respect to the rest state
    virtual VXd grad_X() const override {
        throw std::runtime_error("TODO");
    }

    virtual void hessian(SuiteSparseMatrix& H, bool /* projectionMask */ = true) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Inflation load Hessian");
        const auto &sheet = getObj();
        const auto &m = sheet.mesh();
        auto assemblePerTriContrib = [&](const size_t ti, SuiteSparseMatrix &Hout) {
            const auto &tri = m.element(ti);
            // TODO: Single loop over "other" vertices, determining v_a and v_b cyclically
            for (    const auto v_b : tri.vertices()) {
                for (const auto v_a : tri.vertices()) {
                    size_t a = 3 * v_a.index(),
                           b = 3 * v_b.index();
                    if (a >= b) continue; // strict upper triangle only (no vertex self-interaction)

                    const size_t vla = v_a.localIndex();
                    const size_t vlb = v_b.localIndex();
                    // Gradient wrt v1 of a triangle's signed volume contribution is:
                    //      d vol / d v1 = v_2 x  v_3
                    // so differentiating again with respect to v_2 or v_3
                    // gives a cross product matrix -[v_3]_x or [v_2]_x, respectively.
                    // The sign here is referred to as ordering_sign below.
                    const size_t vlother = 3 - (vla + vlb);
                    const double ordering_sign = (vlb == ((vla + 1) % 3)) ? 1.0 : -1.0;
                    // V3d contrib = (pressure * ordering_sign / 6.0) * triCornerPos.col(vlother);
                    V3d contrib = (pressure * ordering_sign / 6.0) * sheet.deformedPositions().row(tri.vertex(vlother).index());// triCornerPos.col(vlother);

                    size_t hint;
                    hint = Hout.addNZ(a + 1, b + 0,  contrib[2]);
                    // Hout.addNZ(a + 2, b + 0, -contrib[1], hint);
                    Hout.Ax[hint] -= contrib[1];

                    hint += (Hout.Ap[b + 1] - Hout.Ap[b + 0]) - 2;
                    hint = Hout.addNZ(a + 0, b + 1, -contrib[2], hint);
                    // Hout.addNZ(a + 2, b + 1,  contrib[0], hint + 1);
                    Hout.Ax[hint + 1] += contrib[0];

                    hint += (Hout.Ap[b + 2] - Hout.Ap[b + 1]) - 1;
                    hint = Hout.addNZ(a + 0, b + 2,  contrib[1], hint);
                    // Hout.addNZ(a + 1, b + 2, -contrib[0], hint);
                    Hout.Ax[hint] -= contrib[0];
                }
            }
        };
        const size_t ne = m.numTris();
        for (size_t ei = 0; ei < ne; ++ei)
            assemblePerTriContrib(ei, H);
    }

    // *Additional* nonzeros contributed by this load to the potential energy Hessian.
    // (There are none).
    virtual SuiteSparseMatrix hessianSparsityPattern(Real /* val */ = 0.0) const override {
        const size_t nv = getObj().numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
        return SuiteSparseMatrix(Hsp);
    }

    virtual ~Inflation() { }

private:
    Real m_vol;
};

}

#endif /* end of include guard: LOADS_INFLATION_HH */
