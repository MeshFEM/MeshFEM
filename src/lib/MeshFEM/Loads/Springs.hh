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

// Represents one of the coordinates of either a material point or a fixed
// position in space (if no varIndices are specified).
template<typename _Real>
struct AttachmentPointCoordinate {
    using Real = _Real;
    using VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using VXi = Eigen::Matrix< int, Eigen::Dynamic, 1>;
    VXi varIndices;
    VXd coefficients;
    AttachmentPointCoordinate(Eigen::Ref<const VXi> vidxs, Eigen::Ref<const VXd> coeffs)
        : varIndices(vidxs), coefficients(coeffs) { }

    AttachmentPointCoordinate(Real c)
        : varIndices(0), coefficients(VXd::Constant(1, c)) { }

    static std::vector<AttachmentPointCoordinate> fromDeformationSamplerMatrix(const SuiteSparseMatrix &dsm) {
        std::vector<AttachmentPointCoordinate> result;
        result.reserve(dsm.m);
        // The rows of dsm give the indices/coefficients defining each attachment point coordinate.
        // We must work with the transpose so that these rows are contiguous in our compressed column format.
        auto dsm_t = dsm.transpose();
        using Index = decltype(dsm_t.n);
        using IndexVec = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
        for (Index c = 0; c < dsm_t.n; ++c) {
            Index begin = dsm_t.Ap[c], end = dsm_t.Ap[c + 1];
            result.emplace_back(Eigen::Map<const IndexVec>(&dsm_t.Ai[begin], end - begin).template cast<typename VXi::Scalar>(),
                                Eigen::Map<const VXd>(&dsm_t.Ax[begin], end - begin));
        }
        return result;
    }

    static std::vector<AttachmentPointCoordinate> fromTargetPositions(Eigen::Ref<const Eigen::VectorXd> targetPositions) {
        std::vector<AttachmentPointCoordinate> result;
        const size_t n = targetPositions.size();
        result.reserve(n);
        for (size_t i = 0; i < n; ++i)
            result.emplace_back(targetPositions[i]);
        return result;
    }

    bool isFixedAnchor() const { return varIndices.size() == 0; }
    void validate() const {
        if (isFixedAnchor())  {
            if (size_t(coefficients.size()) != 1) throw std::runtime_error("Anchor point component should have only one coefficent");
        }
        else {
            if (coefficients.size() != varIndices.size()) throw std::runtime_error("Variable coefficient size mismatch");
        }
    }

    Real getPosition(Eigen::Ref<const VXd> vars) const {
        if (isFixedAnchor()) return coefficients[0];
        Real pos = 0.0;
        const size_t nvi = varIndices.size();
        for (size_t vi = 0; vi < nvi; ++vi)
            pos += vars[varIndices[vi]] * coefficients[vi];
        return pos;
    }

    void gradContribution(Real stress, Eigen::Ref<VXd> grad) const {
        if (isFixedAnchor()) return; // Fixed anchor points do not contribute to the gradient
        const size_t nvi = varIndices.size();
        for (size_t vi = 0; vi < nvi; ++vi)
            grad[varIndices[vi]] += coefficients[vi] * stress;
    }
};

template<class Object>
struct Springs : public Load<Object::N, typename Object::Real> {
    static constexpr size_t N = Object::N;
    using Base = Load<Object::N, typename Object::Real>;
    using Real = typename Base::Real;
    using VXd  = typename Base::VXd;
    using MXNd = Eigen::Matrix<Real, N, 1>;
    using APC = AttachmentPointCoordinate<Real>;

    // Create uniaxial, axis-aligned springs connecting the attachment points
    // in `coordsA` with the corresponding attachment points in `coordsB`
    Springs(std::weak_ptr<const Object> obj,
            const std::vector<APC> &coordsA,
            const std::vector<APC> &coordsB,
            Eigen::Ref<const VXd> stiffnesses)
        : m_obj(obj), m_coordsA(coordsA), m_coordsB(coordsB), m_k(stiffnesses)
    {
        if (coordsA.size() != coordsB.size()) throw std::runtime_error("Attachment point size mismatch");
        if (size_t(stiffnesses.size()) != coordsA.size()) throw std::runtime_error("Spring stiffnesses size mismatch");
        for (const auto &p : coordsA) p.validate();
        for (const auto &p : coordsB) p.validate();

        m_updateCache();
        m_callbackID = getObj().registerDeformationUpdateCallback([this]() { m_updateCache(); });
    }

    Springs(std::weak_ptr<const Object> obj,
            const std::vector<APC> &coordsA,
            const std::vector<APC> &coordsB,
            Real stiffness)
        : Springs(obj, coordsA, coordsB, Eigen::VectorXd::Constant(coordsA.size(), stiffness)) { }

    template<typename Stiffnesses>
    Springs(std::weak_ptr<const Object> obj,
            const SuiteSparseMatrix &deformationSamplerMatrix,
            Eigen::Ref<const Eigen::VectorXd> targetPositions,
            Stiffnesses stiffness)
        : Springs(obj, APC::fromDeformationSamplerMatrix(deformationSamplerMatrix),
                       APC::fromTargetPositions(targetPositions), stiffness) { }

    void setStiffnesses(Eigen::Ref<const Eigen::VectorXd> ks) { m_k = ks; m_updateCache(); }
    void setStiffnesses(Real k) { setStiffnesses(Eigen::VectorXd::Constant(m_coordsA.size(), k)); }
    VXd getStiffnesses() const { return m_k; }

    virtual Real energy() const override { return m_energy; }

    // Derivative with respect to deformed configuration
    virtual VXd grad_x() const override { return m_grad; }

    // Derivative with respect to rest configuration (for shape optimization)
    virtual VXd grad_X() const override { return VXd::Zero(m_grad.size()); }

    // Hessian with respect to deformed configuration (H_xx)
    virtual void hessian(SuiteSparseMatrix &H, bool /* projectionMask */ = true) const override {
        const size_t ns = numSprings();

        auto addInteractions = [&](const APC &coords1, const APC &coords2, Real stiffness, bool crossTerms) {
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
            addInteractions(m_coordsA[s], m_coordsA[s], m_k[s], false);
            addInteractions(m_coordsB[s], m_coordsB[s], m_k[s], false);
            addInteractions(m_coordsA[s], m_coordsB[s], m_k[s],  true);
        }
    }

    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const override {
        const size_t nv = getObj().numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
        const size_t ns = numSprings();

        auto addInteractions = [&](const APC &coords1, const APC &coords2, bool crossTerms) {
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

    virtual ~Springs() {
        if (auto o = m_obj.lock())
            o->deregisterDeformationUpdateCallback(m_callbackID);
    }

private:
    std::weak_ptr<const Object> m_obj;
    std::vector<APC> m_coordsA, m_coordsB;
    Eigen::VectorXd m_k;
    int m_callbackID;

    const Object &getObj() const {
        if (auto o = m_obj.lock()) return *o;
        throw std::runtime_error("Elastic object was destroyed");
    }

    void m_updateCache() {
        const auto &x = getObj().getVars();
        const size_t ns = numSprings();
        VXd posA(ns), posB(ns);
        for (size_t s = 0; s < ns; ++s) {
             posA[s] = m_coordsA[s].getPosition(x);
             posB[s] = m_coordsB[s].getPosition(x);
        }
        VXd diff = posA - posB;
        VXd stresses = (m_k.array() * (posA - posB).array()).matrix();
        m_energy = 0.5 * diff.dot(stresses);

        m_grad.setZero(x.size());
        for (size_t s = 0; s < ns; ++s) {
            m_coordsA[s].gradContribution( stresses[s], m_grad);
            m_coordsB[s].gradContribution(-stresses[s], m_grad);
        }
    }

    // Cached state
    Real m_energy;
    VXd m_grad;
};

} // namespace Loads

#endif /* end of include guard: SPRINGS_HH */
