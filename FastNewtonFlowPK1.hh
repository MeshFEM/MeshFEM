#pragma once

#include "3rdparty/TaylorAutodiff/TaylorFieldViews.hh"

namespace FastNewtonFlowDetail {
using namespace TaylorADFields;

// Store only [G00, G01, G11] of G = B^T B. Pair the i and k-i
// convolution terms, whose sum is symmetric even when each term is not.
template<class GN>
struct PackedGram2x2
    : TaylorGraphNodeNaryOperationWithViews<AoSMatrixStorage<Eigen::Vector3d>, GN> {
    using InputStorage = StorageTypeOf<GN>;
    using Storage = AoSMatrixStorage<Eigen::Vector3d>;
    using Base = TaylorGraphNodeNaryOperationWithViews<Storage, GN>;
    using Holder = TaylorGraphNodeHolder<GN>;
    static_assert(std::is_same_v<InputStorage, AoSMatrixStorage<Eigen::Matrix2d>>,
                  "PackedGram2x2 requires owning column-major FP64 2x2 matrix fields");

    PackedGram2x2(Holder b) : Base(b) { this->upgrade(b->degree()); }
    static auto make(Holder b) { return TaylorGraphNodeHolder<PackedGram2x2>::make(b); }

private:
    int m_actualDegreeForTarget(int target) const override {
        int d = this->template input<0>()->degree();
        return d < 0 ? -1 : std::min(target, 2 * d);
    }

    static Eigen::Vector3d pair(const double *x, const double *y) {
        return {2 * (x[0]*y[0] + x[1]*y[1]),
                x[0]*y[2] + x[1]*y[3] + y[0]*x[2] + y[1]*x[3],
                2 * (x[2]*y[2] + x[3]*y[3])};
    }

    static Eigen::Vector3d middle(const double *x) {
        return {x[0]*x[0] + x[1]*x[1],
                x[0]*x[2] + x[1]*x[3],
                x[2]*x[2] + x[3]*x[3]};
    }

    void m_computeDegreeRange(int first, int last, const OperationMaskSlice &mask) override {
        const auto &b = *this->template input<0>();
        for (int k = first; k <= last; ++k) {
            double *r = (*this)[k].array().data();
            bool initial = true;
            for (int i = std::max(0, k - b.degree()); i <= std::min(k / 2, b.degree()); ++i) {
                const double *x = b[i].array().data(), *y = b[k - i].array().data();
                if (i == k - i) {
                    for (int e : mask) {
                        Eigen::Map<Eigen::Vector3d> out(r + 3 * e);
                        if (initial) out  = middle(x + 4 * e);
                        else         out += middle(x + 4 * e);
                    }
                }
                else {
                    for (int e : mask) {
                        Eigen::Map<Eigen::Vector3d> out(r + 3 * e);
                        if (initial) out  = pair(x + 4 * e, y + 4 * e);
                        else         out += pair(x + 4 * e, y + 4 * e);
                    }
                }
                initial = false;
            }
            if (initial) throw std::logic_error("Missing PackedGram2x2 coefficients");
        }
    }

    void m_computeHighestDegreeCoefficientPerturbation(CoefficientPerturbations &p,
                                                      const OperationMaskSlice &mask) const override {
        const auto &b = *this->template input<0>();
        const auto &db = p.template getPerturbation<InputStorage>(b);
        auto &out = p.template getPerturbation<Storage>(*this);
        if (db.degree != out.degree)
            throw std::logic_error("Incompatible PackedGram2x2 perturbation degree");
        for (int e : mask)
            Eigen::Map<Eigen::Vector3d>(out->array().data() + 3 * e) =
                pair(b[0].array().data() + 4 * e, db->array().data() + 4 * e);
    }
};

// C - G B^T with packed symmetric G and ordinary 2x2 B,C. Fuse C into
// the first convolution term, retaining the original product's single pass.
template<class GNG, class GNB, class GNC>
struct PackedGramProductDifference2x2
    : TaylorGraphNodeNaryOperationWithViews<AoSMatrixStorage<Eigen::Matrix2d>, GNG, GNB, GNC> {
    using Storage = AoSMatrixStorage<Eigen::Matrix2d>;
    using PackedStorage = AoSMatrixStorage<Eigen::Vector3d>;
    using Base = TaylorGraphNodeNaryOperationWithViews<Storage, GNG, GNB, GNC>;
    using GHolder = TaylorGraphNodeHolder<GNG>;
    using BHolder = TaylorGraphNodeHolder<GNB>;
    using CHolder = TaylorGraphNodeHolder<GNC>;
    static_assert(std::is_same_v<StorageTypeOf<GNG>, PackedStorage> &&
                  std::is_same_v<StorageTypeOf<GNB>, Storage> &&
                  std::is_same_v<StorageTypeOf<GNC>, Storage>,
                  "PackedGramProductDifference2x2 requires packed FP64 G and owning column-major FP64 2x2 B,C fields");

    PackedGramProductDifference2x2(GHolder g, BHolder b, CHolder c) : Base(g, b, c) {
        this->upgrade(this->inferDegreeFromInputs());
    }
    static auto make(GHolder g, BHolder b, CHolder c) {
        return TaylorGraphNodeHolder<PackedGramProductDifference2x2>::make(g, b, c);
    }

private:
    static Eigen::Matrix2d product(const double *g, const double *b) {
        Eigen::Matrix2d G;
        G << g[0], g[1], g[1], g[2];
        return G * Eigen::Map<const Eigen::Matrix2d>(b).transpose();
    }

    int m_actualDegreeForTarget(int target) const override {
        int g = this->template input<0>()->degree(),
            b = this->template input<1>()->degree(),
            c = this->template input<2>()->degree();
        return std::min({g, b, c}) < 0 ? -1 : std::min(target, std::max(g + b, c));
    }

    void m_computeDegreeRange(int first, int last, const OperationMaskSlice &mask) override {
        const auto &g = *this->template input<0>();
        const auto &b = *this->template input<1>();
        const auto &c = *this->template input<2>();
        for (int k = first; k <= last; ++k) {
            double *r = (*this)[k].array().data();
            const double *cp = k <= c.degree() ? c[k].array().data() : nullptr;
            bool initial = true;
            for (int i = std::max(0, k - b.degree()); i <= std::min(k, g.degree()); ++i) {
                const double *gp = g[i].array().data(), *bp = b[k - i].array().data();
                for (int e : mask) {
                    Eigen::Map<Eigen::Matrix2d> out(r + 4 * e);
                    auto term = product(gp + 3 * e, bp + 4 * e);
                    if (initial) {
                        if (cp) out = Eigen::Map<const Eigen::Matrix2d>(cp + 4 * e) - term;
                        else    out = -term;
                    }
                    else out -= term;
                }
                initial = false;
            }
            if (initial) {
                if (cp) (*this)[k].set(mask, c[k].array());
                else throw std::logic_error("Missing PackedGramProductDifference2x2 coefficients");
            }
        }
    }

    void m_computeHighestDegreeCoefficientPerturbation(CoefficientPerturbations &p,
                                                      const OperationMaskSlice &mask) const override {
        const auto &g = *this->template input<0>();
        const auto &b = *this->template input<1>();
        const auto &c = *this->template input<2>();
        auto &out = p.template getPerturbation<Storage>(*this);
        const double *dg = nullptr, *db = nullptr, *dc = nullptr;
        if (p.hasPerturbation(g)) {
            const auto &delta = p.template getPerturbation<PackedStorage>(g);
            if (delta.degree != out.degree) throw std::logic_error("Packed Gram product G perturbation degree mismatch");
            dg = delta->array().data();
        }
        if (p.hasPerturbation(b)) {
            const auto &delta = p.template getPerturbation<Storage>(b);
            if (delta.degree != out.degree) throw std::logic_error("Packed Gram product B perturbation degree mismatch");
            db = delta->array().data();
        }
        if (p.hasPerturbation(c)) {
            const auto &delta = p.template getPerturbation<Storage>(c);
            if (delta.degree != out.degree) throw std::logic_error("Packed Gram product C perturbation degree mismatch");
            dc = delta->array().data();
        }
        if (!dg && !db && !dc) throw std::logic_error("Missing PackedGramProductDifference2x2 perturbation");
        for (int e : mask) {
            Eigen::Matrix2d r = Eigen::Matrix2d::Zero();
            if (dc) r  = Eigen::Map<const Eigen::Matrix2d>(dc + 4 * e);
            if (dg) r -= product(dg + 3 * e, b[0].array().data() + 4 * e);
            if (db) r -= product(g[0].array().data() + 3 * e, db + 4 * e);
            Eigen::Map<Eigen::Matrix2d>(out->array().data() + 4 * e) = r;
        }
    }
};
} // namespace FastNewtonFlowDetail
