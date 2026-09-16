#pragma once

#include "3rdparty/TaylorAutodiff/TaylorFieldViews.hh"

namespace FastNewtonFlowDetail {
using namespace TaylorADFields;

template<class GNA, class GNB>
struct CrossProduct2D
    : TaylorGraphNodeNaryOperationWithViews<ScalarFieldStorage<ScalarTypeOf<GNA>>, GNA, GNB> {
    using AStorage = StorageTypeOf<GNA>;
    using BStorage = StorageTypeOf<GNB>;
    using Scalar = ScalarTypeOf<GNA>;
    using Storage = ScalarFieldStorage<Scalar>;
    using Base = TaylorGraphNodeNaryOperationWithViews<Storage, GNA, GNB>;
    using AHolder = TaylorGraphNodeHolder<GNA>;
    using BHolder = TaylorGraphNodeHolder<GNB>;
    static_assert(AStorage::M == 2 && AStorage::N == 1 && BStorage::M == 2 && BStorage::N == 1,
                  "CrossProduct2D requires two-component vector fields");

    CrossProduct2D(AHolder a, BHolder b) : Base(a, b) {
        this->upgrade(this->inferDegreeFromInputs());
    }
    static auto make(AHolder a, BHolder b) {
        return TaylorGraphNodeHolder<CrossProduct2D>::make(a, b);
    }

private:
    template<class A, class B>
    static auto cross(const A &a, const B &b) {
        return a.array().col(0) * b.array().col(1) - a.array().col(1) * b.array().col(0);
    }

    int m_actualDegreeForTarget(int target) const override {
        const auto &a = *this->template input<0>();
        const auto &b = *this->template input<1>();
        return (a.degree() < 0 || b.degree() < 0) ? -1 : std::min(target, a.degree() + b.degree());
    }

    void m_computeDegreeRange(int first, int last, const OperationMaskSlice &mask) override {
        const auto &a = *this->template input<0>();
        const auto &b = *this->template input<1>();
        for (int k = first; k <= last; ++k) {
            bool initial = true;
            for (int i = std::max(0, k - b.degree()); i <= std::min(k, a.degree()); ++i) {
                auto term = cross(a[i], b[k - i]);
                if (initial) { (*this)[k].set(mask, term); initial = false; }
                else           (*this)[k].add(mask, term);
            }
            if (initial) throw std::logic_error("Missing CrossProduct2D coefficients");
        }
    }

    void m_computeHighestDegreeCoefficientPerturbation(CoefficientPerturbations &p,
                                                      const OperationMaskSlice &mask) const override {
        const auto &a = *this->template input<0>();
        const auto &b = *this->template input<1>();
        const bool hasA = p.hasPerturbation(a), hasB = p.hasPerturbation(b);
        auto &out = p.template getPerturbation<Storage>(*this);
        if ((!hasA && !hasB) ||
            (hasA && p.template getPerturbation<AStorage>(a).degree != out.degree) ||
            (hasB && p.template getPerturbation<BStorage>(b).degree != out.degree))
            throw std::logic_error("Incompatible CrossProduct2D perturbations");
        if (hasA) out->set(mask, cross(*p.template getPerturbation<AStorage>(a), b[0]));
        if (hasB) {
            auto term = cross(a[0], *p.template getPerturbation<BStorage>(b));
            if (hasA) out->add(mask, term);
            else      out->set(mask, term);
        }
    }
};

template <class GNA, class GNB> auto cross_product_2D(TaylorGraphNodeHolder<GNA> a, TaylorGraphNodeHolder<GNB> b) {
    return CrossProduct2D<GNA, GNB>::make(a, b);
}

// Compute [||F||^2, det(F)] together, using the same matrix loads for both
// quadratic convolutions. This specialized kernel is for the owning FP64
// fields used by the two-dimensional FastNewtonFlow projection graph.
template<class GN>
struct FusedInvariants2x2
    : TaylorGraphNodeNaryOperationWithViews<AoSMatrixStorage<Eigen::Vector2d>, GN> {
    using InputStorage = StorageTypeOf<GN>;
    using Storage = AoSMatrixStorage<Eigen::Vector2d>;
    using Base = TaylorGraphNodeNaryOperationWithViews<Storage, GN>;
    using Holder = TaylorGraphNodeHolder<GN>;
    static_assert(std::is_same_v<InputStorage, AoSMatrixStorage<Eigen::Matrix2d>>,
                  "FusedInvariants2x2 requires owning column-major FP64 2x2 matrix fields");

    FusedInvariants2x2(Holder f) : Base(f) { this->upgrade(f->degree()); }
    static auto make(Holder f) { return TaylorGraphNodeHolder<FusedInvariants2x2>::make(f); }

private:
    int m_actualDegreeForTarget(int target) const override {
        int d = this->template input<0>()->degree();
        return d < 0 ? -1 : std::min(target, 2 * d);
    }

    void m_computeDegreeRange(int first, int last, const OperationMaskSlice &mask) override {
        const auto &f = *this->template input<0>();
        const int begin = *mask.begin(), end = begin + (mask.end() - mask.begin());
        for (int k = first; k <= last; ++k) {
            double *r = (*this)[k].array().data();
            bool initial = true;
            // Pair i and k-i, with a single middle term when they coincide.
            for (int i = std::max(0, k - f.degree()); i <= std::min(f.degree(), k / 2); ++i) {
                const double *a = f[i].array().data(), *b = f[k - i].array().data();
                if (i == k - i) {
                    for (int e = begin; e < end; ++e) {
                        const double *x = a + 4 * e;
                        const double norm = (x[0]*x[0] + x[1]*x[1]) + (x[2]*x[2] + x[3]*x[3]);
                        const double det = x[0]*x[3] - x[1]*x[2];
                        if (initial) { r[2*e] = norm; r[2*e + 1] = det; }
                        else         { r[2*e] += norm; r[2*e + 1] += det; }
                    }
                }
                else {
                    for (int e = begin; e < end; ++e) {
                        const double *x = a + 4 * e, *y = b + 4 * e;
                        const double norm = 2 * ((x[0]*y[0] + x[1]*y[1]) + (x[2]*y[2] + x[3]*y[3]));
                        const double det = (x[0]*y[3] - x[1]*y[2]) + (y[0]*x[3] - y[1]*x[2]);
                        if (initial) { r[2*e] = norm; r[2*e + 1] = det; }
                        else         { r[2*e] += norm; r[2*e + 1] += det; }
                    }
                }
                initial = false;
            }
            if (initial) throw std::logic_error("Missing fused-invariant coefficients");
        }
    }

    void m_computeHighestDegreeCoefficientPerturbation(CoefficientPerturbations &p,
                                                      const OperationMaskSlice &mask) const override {
        const auto &f = *this->template input<0>();
        const auto &delta = p.template getPerturbation<InputStorage>(f);
        auto &out = p.template getPerturbation<Storage>(*this);
        if (delta.degree != out.degree)
            throw std::logic_error("Incompatible fused-invariant perturbation degree");
        const auto &f0 = f[0].array();
        const auto &df = delta->array();
        auto &r = out->array();
        for (int e : mask) {
            r(e, 0) = 2 * (f0.row(e) * df.row(e)).sum();
            r(e, 1) = df(e, 0)*f0(e, 3) + f0(e, 0)*df(e, 3) - df(e, 1)*f0(e, 2) - f0(e, 1)*df(e, 2);
        }
    }
};

// Read a component of the fused invariant field without another coefficient
// history. LinearCoefficientView also maps perturbations through this column.
template<int Component>
struct InvariantComponentMap {
    static_assert(Component == 0 || Component == 1, "Invalid invariant component");
    static constexpr int shift = 0;
    template<class S> using Storage = ScalarFieldStorage<typename S::Scalar>;

    template<class S> static auto expression(const S &s, int) {
        static_assert(std::is_same_v<S, AoSMatrixStorage<Eigen::Vector2d>>,
                      "ExtractInvariant requires the owning FP64 two-component invariant field");
        return s.array().col(Component);
    }
};

template<class GN, int Component>
using ExtractInvariant = LinearCoefficientView<GN, InvariantComponentMap<Component>>;
} // namespace FastNewtonFlowDetail
