////////////////////////////////////////////////////////////////////////////////
// Springs.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Linear zero-restlength springs linking two material points of an elastic
//  object's deformed configuration (expressed as a linear combination of the
//  equilibrium variables) or attaching a material point to a fixed anchor
//  point in space.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  10/27/2020 22:40:22
////////////////////////////////////////////////////////////////////////////////
#ifndef SPRINGS_HH
#define SPRINGS_HH
#include "Load.hh"

namespace Loads {

// Represents the coordinates of either a material point or a fixed
// position in space (if no varIndices are specified).
// When `BlockSize > 1`, this represents a `BlockSize`-dimensional
// vector of coordinates rather than a single coordinate.
// In this case, `varIndices` holds *block variable* indices (e.g.,
// holding node indices rather than a coordinate variable index).
template<typename _CoordinateType>
struct AttachmentPointCoordinate {
    using CTraits = spmat_helper::value_traits<_CoordinateType>;
    using Real = typename CTraits::Scalar;
    using VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using VXi = Eigen::Matrix< int, Eigen::Dynamic, 1>;
    static constexpr size_t BlockSize = (CTraits::rows * CTraits::cols);

    VXi varIndices;
    VXd coefficients;
    AttachmentPointCoordinate(Eigen::Ref<const VXi> vidxs, Eigen::Ref<const VXd> coeffs)
        : varIndices(vidxs), coefficients(coeffs) { }

    AttachmentPointCoordinate(const _CoordinateType &c)
        : varIndices(0) {
        coefficients.resize(BlockSize);
        coefficients << c;
    }

    static std::vector<AttachmentPointCoordinate> fromDeformationSamplerMatrix(const SuiteSparseMatrix &dsm) {
        std::vector<AttachmentPointCoordinate> result;
        if (((dsm.m % BlockSize) != 0) ||
            ((dsm.m % BlockSize) != 0)) {
            throw std::runtime_error("dsm row and column size must be divisible by " + std::to_string(BlockSize));
        }
        result.reserve(dsm.m / BlockSize);
        // The rows of dsm give the indices/coefficients defining each attachment point coordinate.
        // We must work with the transpose so that these rows are contiguous in our compressed column format.
        auto dsm_t = dsm.transpose();
        using Index = decltype(dsm_t.n);
        using IndexVec = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
        for (Index c = 0; c < dsm_t.n; c += BlockSize) {
            Index begin = dsm_t.Ap[c], end = dsm_t.Ap[c + 1];
            // Here we effectively convert from a "scalar" deformation sampler
            // matrix down to a "block" matrix by compressing each
            // `BlockSize x BlockSize` scaled identity matrix down to a single
            // entry (i.e., the top-left value).
            result.emplace_back(Eigen::Map<const IndexVec>(&dsm_t.Ai[begin], end - begin).template cast<typename VXi::Scalar>() / BlockSize,
                                Eigen::Map<const VXd>(&dsm_t.Ax[begin], end - begin));
        }
        return result;
    }

    static std::vector<AttachmentPointCoordinate> fromTargetPositions(const Eigen::Ref<const Eigen::VectorXd> &targetPositions) {
        std::vector<AttachmentPointCoordinate> result;
        if (targetPositions.size() % BlockSize != 0) throw std::runtime_error("targetPositions.size() must be divisible by " + std::to_string(BlockSize));
        const size_t n = targetPositions.size() / BlockSize;
        result.reserve(n);
        for (size_t i = 0; i < n; ++i)
            result.emplace_back(extract(targetPositions, i));
        return result;
    }

    bool isFixedAnchor() const { return varIndices.size() == 0; }
    void validate() const {
        if (isFixedAnchor())  {
            if (size_t(coefficients.size()) != BlockSize) throw std::runtime_error("Anchor point component should have only one (block) coefficent");
        }
        else {
            if (coefficients.size() != varIndices.size()) throw std::runtime_error("Variable coefficient size mismatch");
        }
    }

    // Loop over (nodeIndex, coefficient) pairs
    template <typename F>
    void foreach_var(const F &f) const {
        const size_t nvi = varIndices.size();
        for (size_t vi = 0; vi < nvi; ++vi)
            f(varIndices[vi], coefficients[vi]);
    }

    _CoordinateType getPosition(const Eigen::Ref<const VXd> &vars) const {
        return m_getPositionImpl(vars, [](size_t vari) { return vari; });
    }

    void gradContribution(const _CoordinateType &grad_p, Eigen::Ref<VXd> grad) const {
        m_gradContributionImpl(grad_p, grad, [](size_t vari) { return vari; });
    }


    // Versions of `getPosition` and `gradContribution` that employ an index
    // remapping from local indices `varIndices` to global indices
    // `globalVarForLocalVar[varIndices]`.
    _CoordinateType getPosition(const Eigen::Ref<const VXd> &vars, const std::vector<int> &globalVarForLocalVar) const {
        return m_getPositionImpl(vars, [&](size_t vari) { return globalVarForLocalVar[vari]; });
    }

    void gradContribution(const _CoordinateType &grad_p, Eigen::Ref<VXd> grad, const std::vector<int> &globalVarForLocalVar) const {
        m_gradContributionImpl(grad_p, grad, [&](size_t vari) { return globalVarForLocalVar[vari]; });
    }

    template<class Derived> static auto  extract(const Eigen::MatrixBase<Derived> &vars, size_t i) { return spmat_helper::SegmentGetter<BlockSize, Derived>::get(vars, i); }
    template<class Derived> static auto &extract(      Eigen::MatrixBase<Derived> &vars, size_t i) { return spmat_helper::SegmentGetter<BlockSize, Derived>::get(vars, i); }

private:
    template<typename IndexRemaper>
    _CoordinateType m_getPositionImpl(const Eigen::Ref<const VXd> &vars, const IndexRemaper &iremap) const {
        if (isFixedAnchor()) return extract(coefficients, 0);
        _CoordinateType pos;
        spmat_helper::setZero(pos);
        foreach_var([&](size_t vi, Real coeff) {
            pos += extract(vars, iremap(varIndices[vi])) * coefficients[vi];
        });

        return pos;
    }

    template<typename IndexRemaper>
    void m_gradContributionImpl(const _CoordinateType &grad_p, Eigen::Ref<VXd> grad, const IndexRemaper &iremap) const {
        if (isFixedAnchor()) return; // Fixed anchor points do not contribute to the gradient
        foreach_var([&](size_t vi, Real coeff) {
            extract(grad, iremap(varIndices[vi])) += coefficients[vi] * grad_p;
        });
    }
};

template<class APC_A, class APC_B>
struct GenericSprings : public Load<Real> {
    using Base = Load<Real>;
    using VXd = typename Base::VXd;

    // Create uniaxial, axis-aligned springs connecting the attachment points
    // in `coordsA` with the corresponding attachment points in `coordsB`
    GenericSprings(std::weak_ptr<NewtonVarsBase> vars,
            const std::vector<APC_A> &coordsA,
            const std::vector<APC_B> &coordsB,
            const Eigen::Ref<const VXd> &stiffnesses)
        : m_vars(vars), m_coordsA(coordsA), m_coordsB(coordsB), m_k(stiffnesses)
    {
        if (coordsA.size() != coordsB.size()) throw std::runtime_error("Attachment point size mismatch");
        if (size_t(stiffnesses.size()) != coordsA.size()) throw std::runtime_error("Spring stiffnesses size mismatch");
        for (const auto &p : coordsA) p.validate();
        for (const auto &p : coordsB) p.validate();

        m_updateCache();
    }

    GenericSprings(std::weak_ptr<NewtonVarsBase> vars,
            const std::vector<APC_A> &coordsA,
            const std::vector<APC_B> &coordsB,
            Real stiffness)
        : GenericSprings(vars, coordsA, coordsB, Eigen::VectorXd::Constant(coordsA.size(), stiffness)) { }

    template<typename Stiffnesses>
    GenericSprings(std::weak_ptr<NewtonVarsBase> vars,
            const SuiteSparseMatrix &deformationSamplerMatrix,
            Eigen::Ref<const Eigen::VectorXd> targetPositions,
            Stiffnesses stiffness)
        : GenericSprings(vars, APC_A::fromDeformationSamplerMatrix(deformationSamplerMatrix),
                        APC_B::fromTargetPositions(targetPositions), stiffness) { }

    void setStiffnesses(Eigen::Ref<const Eigen::VectorXd> ks) { m_k = ks; m_updateCache(); }
    void setStiffnesses(Real k) { setStiffnesses(Eigen::VectorXd::Constant(m_coordsA.size(), k)); }
    VXd getStiffnesses() const { return m_k; }

    virtual Real energy() const override { return m_energy; }

    // Derivative with respect to deformed configuration
    virtual VXd grad_x() const override { return m_grad; }

    // Derivative with respect to rest configuration (for shape optimization)
    virtual VXd grad_X() const override { return VXd::Zero(m_grad.size()); }

    // Hessian with respect to deformed configuration (H_xx)
    virtual void accumulateHessian(Real weight, SuiteSparseMatrix &H, bool /* projectionMask */ = true) const override {
        const size_t ns = numSprings();

        auto addInteractions = [&](const APC_A &coords1, const APC_B &coords2, Real stiffness, bool crossTerms) {
            Real sign = crossTerms ? -1.0 : 1.0;
            for (int ii = 0; ii < coords1.varIndices.size(); ++ii) {
                for (int jj = (crossTerms ? 0 : ii); jj < coords2.varIndices.size(); ++jj) { // Visit each unordered pair once
                    int i = coords1.varIndices[ii],
                        j = coords2.varIndices[jj];
                    H.addNZ(std::min(i, j), std::max(i, j), sign * coords1.coefficients[ii] * coords2.coefficients[jj] * stiffness);
                }
            }
        };
        for (size_t s = 0; s < ns; ++s) {
            addInteractions(m_coordsA[s], m_coordsA[s], weight * m_k[s], false);
            addInteractions(m_coordsB[s], m_coordsB[s], weight * m_k[s], false);
            addInteractions(m_coordsA[s], m_coordsB[s], weight * m_k[s],  true);
        }
    }

    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const override {
        const size_t nv = m_vars->numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
        const size_t ns = numSprings();

        auto addInteractions = [&](const APC_A &coords1, const APC_B &coords2, bool crossTerms) {
            for (int ii = 0; ii < coords1.varIndices.size(); ++ii) {
                for (int jj = (crossTerms ? 0 : ii); jj < coords2.varIndices.size(); ++jj) { // Visit each unordered pair once
                    int i = coords1.varIndices[ii],
                        j = coords2.varIndices[jj];
                    Hsp.addNZ(std::min(i, j), std::max(i, j), 1.0);
                }
            }
        };
        for (size_t s = 0; s < ns; ++s) {
            addInteractions(m_coordsA[s], m_coordsA[s], false);
            addInteractions(m_coordsB[s], m_coordsB[s], false);
            addInteractions(m_coordsA[s], m_coordsB[s],  true);
        }

        SuiteSparseMatrix Hsp_csc(Hsp);
        Hsp_csc.fill(val);
        return Hsp_csc;
    }

    size_t numSprings() const { return m_coordsA.size(); }

    virtual ~GenericSprings() { }

private:
    const std::shared_ptr<NewtonVarsBase> m_vars;

    std::vector<APC_A> m_coordsA;
    std::vector<APC_B> m_coordsB;
    Eigen::VectorXd m_k;

    virtual void m_stateUpdated(typename Base::VM vmask) override {
        if (vmask == Base::VM::Defo) m_updateCache();
    }

    void m_updateCache() {
        const auto &x = m_vars->getVars();
        const size_t ns = numSprings();
        VXd posA(ns), posB(ns);
        for (size_t s = 0; s < ns; ++s) {
            APC_A::extract(posA, s) = m_coordsA[s].getPosition(x);
            APC_B::extract(posB, s) = m_coordsB[s].getPosition(x);
        }
        VXd diff = posA - posB;
        VXd forces = (m_k.array() * (posA - posB).array());
        m_energy = 0.5 * diff.dot(forces);

        m_grad.setZero(x.size());
        for (size_t s = 0; s < ns; ++s) {
            m_coordsA[s].gradContribution( APC_A::extract(forces, s), m_grad);
            m_coordsB[s].gradContribution(-APC_B::extract(forces, s), m_grad);
        }
    }

    // Cached state
    Real m_energy;
    VXd m_grad;
};

using Springs = GenericSprings<AttachmentPointCoordinate<Real>, AttachmentPointCoordinate<Real>>;

} // namespace Loads

#endif /* end of include guard: SPRINGS_HH */
