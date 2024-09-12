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
#include "../Parallelism.hh"

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
    using VXi = Eigen::VectorXi;
    static constexpr size_t BlockSize = (CTraits::rows * CTraits::cols);

    // Type of the derivative of an output coordinate with respect to a block variable
    using JacobianBlock = Eigen::Matrix<Real, BlockSize, BlockSize>;

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

    static std::vector<AttachmentPointCoordinate> fromBlockVars(const Eigen::VectorXi &blockVars) {
        std::vector<AttachmentPointCoordinate> result;
        const int nbv = blockVars.size();
        result.reserve(nbv);

        VXd coeffs = VXd::Ones(1);
        Eigen::VectorXi vidxs(1);

        for (int i = 0; i < nbv; ++i) {
            vidxs[0] = blockVars[i];
            result.emplace_back(vidxs, coeffs);
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

    // NOP: the basic `AttachmentPointCoordinate` class does not maintain state.
    void setVars(const Eigen::Ref<const VXd> &/* vars */) const { }
    void setVars(const Eigen::Ref<const VXd> &/* vars */, const std::vector<int> &/* globalVarForLocalVar */) const { }

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

    // Accumulate the contribution of this attachment point coordinate's second
    // derivative to the spring energy Hessian `H`. This is the expression
    // `dE_dp . d2p_dvar2`, where dE_dp is the gradient of the spring energy
    // with respect to the attachment point coordinate. When the attachment
    // point is a linear function of the variables (as is the case for this base
    // implementation), the second derivative is zero.
    template<class SpMat> void accumulate_contract_d2_dvar2(const _CoordinateType &/* grad_p */, SpMat &/* H */                                                    ) const { }
    template<class SpMat> void accumulate_contract_d2_dvar2(const _CoordinateType &/* grad_p */, SpMat &/* H */, const std::vector<int> &/* globalVarForLocalVar */) const { }

    template<class Derived> static decltype(auto) extract(const Eigen::MatrixBase<Derived> &vars, size_t i) { return spmat_helper::SegmentGetter<BlockSize, Derived>::get(vars, i); }
    template<class Derived> static decltype(auto) extract(      Eigen::MatrixBase<Derived> &vars, size_t i) { return spmat_helper::SegmentGetter<BlockSize, Derived>::get(vars, i); }

    auto d_dvar(size_t vi) const { return JacobianBlock::Identity() * coefficients[vi]; }

private:
    template<typename IndexRemaper>
    _CoordinateType m_getPositionImpl(const Eigen::Ref<const VXd> &vars, const IndexRemaper &iremap) const {
        if (isFixedAnchor()) return extract(coefficients, 0);
        _CoordinateType pos;
        spmat_helper::setZero(pos);
        foreach_var([&](size_t vi, Real coeff) {
            pos += extract(vars, iremap(vi)) * coeff;
        });

        return pos;
    }

    template<typename IndexRemaper>
    void m_gradContributionImpl(const _CoordinateType &grad_p, Eigen::Ref<VXd> grad, const IndexRemaper &iremap) const {
        if (isFixedAnchor()) return; // Fixed anchor points do not contribute to the gradient
        foreach_var([&](size_t vi, Real coeff) {
            extract(grad, iremap(vi)) += coeff * grad_p;
        });
    }
};

template<class APC_A, class APC_B>
struct GenericSprings : public Load<Real> {
    using Base = Load<Real>;
    using VXd = typename Base::VXd;
    static_assert(APC_A::BlockSize == APC_B::BlockSize, "APC variable block sizes must match");
    static constexpr size_t BlockSize = APC_A::BlockSize;
    using MXBd = Eigen::Matrix<Real, Eigen::Dynamic, BlockSize>;

    // Create uniaxial, axis-aligned springs connecting the attachment points
    // in `coordsA` with the corresponding attachment points in `coordsB`
    GenericSprings(std::weak_ptr<NewtonVarsBase> vars,
            const std::vector<APC_A> &coordsA,
            const std::vector<APC_B> &coordsB,
            const Eigen::Ref<const VXd> &stiffnesses)
        : Base(vars), m_coordsA(coordsA), m_coordsB(coordsB), m_k(stiffnesses)
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

        for (size_t s = 0; s < ns; ++s) {
            m_addJacobianOuterProducts</* SparsityOnly = */ false>(H, m_coordsA[s], m_coordsA[s],  weight * m_k[s]);
            m_addJacobianOuterProducts</* SparsityOnly = */ false>(H, m_coordsB[s], m_coordsB[s],  weight * m_k[s]);
            m_addJacobianOuterProducts</* SparsityOnly = */ false>(H, m_coordsA[s], m_coordsB[s], -weight * m_k[s]);
            m_addJacobianOuterProducts</* SparsityOnly = */ false>(H, m_coordsB[s], m_coordsA[s], -weight * m_k[s]);

            m_coordsA[s].accumulate_contract_d2_dvar2( APC_A::extract(m_forces, s), H);
            m_coordsB[s].accumulate_contract_d2_dvar2(-APC_B::extract(m_forces, s), H);
        }
    }

    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const override {
        const size_t nv = this->getNVars().numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
        const size_t ns = numSprings();

        for (size_t s = 0; s < ns; ++s) {
            m_addJacobianOuterProducts</* SparsityOnly = */ true>(Hsp, m_coordsA[s], m_coordsA[s], 1.0);
            m_addJacobianOuterProducts</* SparsityOnly = */ true>(Hsp, m_coordsB[s], m_coordsB[s], 1.0);
            m_addJacobianOuterProducts</* SparsityOnly = */ true>(Hsp, m_coordsA[s], m_coordsB[s], 1.0);
        }

        SuiteSparseMatrix Hsp_csc(Hsp);
        Hsp_csc.fill(val);
        return Hsp_csc;
    }

    const APC_A &attachmentPointA(size_t s) const { return m_coordsA.at(s); }
    const APC_B &attachmentPointB(size_t s) const { return m_coordsB.at(s); }

    size_t numSprings() const { return m_coordsA.size(); }

    virtual ~GenericSprings() { }

private:
    std::vector<APC_A> m_coordsA;
    std::vector<APC_B> m_coordsB;
    Eigen::VectorXd m_k;

    virtual void m_stateUpdated(typename Base::VM vmask) override {
        if (vmask == Base::VM::Defo) m_updateCache();
    }

    void m_updateCache() {
        BENCHMARK_SCOPED_TIMER_SECTION timer("GenericSprings::m_updateCache");

        const auto &x = this->getNVars().getVars();
        const size_t ns = numSprings();
        VXd posA(ns * BlockSize), posB(ns * BlockSize);
        parallel_for_range(ns, [&](size_t s) {
            // Allow the attachment point class to update
            // its cached state (if necessary).
            m_coordsA[s].setVars(x);
            m_coordsB[s].setVars(x);

            APC_A::extract(posA, s) = m_coordsA[s].getPosition(x);
            APC_B::extract(posB, s) = m_coordsB[s].getPosition(x);
        });

        VXd diff = posA - posB;

        m_forces.resize(diff.size());
        Eigen::Map<MXBd>(m_forces.data(), ns, BlockSize) = (m_k.asDiagonal() * Eigen::Map<const MXBd>(diff.data(), ns, BlockSize));
        m_energy = 0.5 * diff.dot(m_forces);

        m_grad.setZero(x.size());
        for (size_t s = 0; s < ns; ++s) {
            m_coordsA[s].gradContribution( APC_A::extract(m_forces, s), m_grad);
            m_coordsB[s].gradContribution(-APC_B::extract(m_forces, s), m_grad);
        }
    }

    // dc1_dx^T * dc2_dx
    template<bool SparsityOnly, class SpMat, class APC1, class APC2>
    void m_addJacobianOuterProducts(SpMat &H, const APC1 &c1, const APC2 &c2, Real stiffness) const {
        static constexpr size_t BS = APC1::BlockSize;
        static_assert(APC2::BlockSize == BS, "APC variable block sizes must match");
        for (int ii = 0; ii < c1.varIndices.size(); ++ii) {
            for (int jj = 0; jj < c2.varIndices.size(); ++jj) {
                int i = c1.varIndices[ii],
                    j = c2.varIndices[jj];
                if (i > j) continue;
                // TODO: do assembly using a SystemAssembler with the
                // correct block variable structure...
                if constexpr (SparsityOnly) H.addNZBlock(BS * i, BS * j, Eigen::Matrix<Real, BS, BS>::Ones());
                else                        H.addNZBlock(BS * i, BS * j, stiffness * c1.d_dvar(ii).transpose() * c2.d_dvar(jj));
            }
        }
    };

    // Cached state
    Real m_energy;
    VXd m_grad, m_forces;
};

using Springs = GenericSprings<AttachmentPointCoordinate<Real>, AttachmentPointCoordinate<Real>>;

} // namespace Loads

#endif /* end of include guard: SPRINGS_HH */
