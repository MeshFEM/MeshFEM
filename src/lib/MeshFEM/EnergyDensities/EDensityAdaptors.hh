////////////////////////////////////////////////////////////////////////////////
// EDensityAdaptors.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Some energy densities are more conveniently expressed in terms of
//  the deformation gradient F or the Cauchy-Green deformation tensor C.
//  At the same time, particular applications may find it more convenient
//  to treat the energy density as a function of F or of C.
//  One particular example is Hessian/gradient calculation for an elastic
//  object, where an F-based interface is most convenient.
//  We provide wrappers for converting between the two interfaces.
//  Wrapping a C based interface with an F interface is simple and involves
//  low overhead. The other direction is more complicated.
//
//  In the future we plan to support efficient wrapping of both C and F-based
//  "volumetric" energy densities as membrane energies, though this requires
//  some thought to avoid recomputing singular value decompositions.
//
//  We additionally provide an `AutoHessianProjection` adaptor that implements
//  a brute-force "F-based" Hessian projection by solving an eigenvalue problem.
//  This can be applied to anisotropic energy density functions for which
//  we do not have analytic expressions for the analytic eigendecompositions.
//  For isotropic energies, it is generally straightforward to derive efficient
//  analytic formulas (see, e.g., `IsoCRLEWithHessianProjection.hh`), but the
//  overhead of the automatic approach may not be high enough to justify this
//  this implementation effort. At the very least, this can be used as a ground
//  truth for unit testing the analytical formulas.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/24/2020 01:30:00
////////////////////////////////////////////////////////////////////////////////
#ifndef EDENSITYADAPTORS_HH
#define EDENSITYADAPTORS_HH
#include <MeshFEM/EnergyDensities/Tensor.hh>
#include "EnergyTraits.hh"

// Implement an F-based interface from a C-based interface
template<class Psi_C, size_t EmbeddingDimension = Psi_C::Dimension>
struct EnergyDensityFBasedFromCBased : public Psi_C {
    using Base = Psi_C;
    static constexpr size_t Dimension = Base::Dimension;
    static constexpr size_t N         = Base::N; // "Reference space" dimension
    static constexpr size_t M         = EmbeddingDimension; // can differ from "N" for membrane energies...
    using Real    = typename Base::Real;
    using Matrix  = Eigen::Matrix<Real, M, N>;

    // Note: all Base constructors except the copy constructor initialize to
    // the identity deformation; this is compatible with our default member
    // initializer for m_F.
    using Base::Base;
    EnergyDensityFBasedFromCBased(const Base &b) : Base(b) { }

    EnergyDensityFBasedFromCBased(const EnergyDensityFBasedFromCBased &) = default;
    EnergyDensityFBasedFromCBased(const EnergyDensityFBasedFromCBased &other, const UninitializedDeformationTag &tag)
        : Base(other, tag), m_F(other.m_F) { }

    void setDeformationGradient(const Matrix &F, const EvalLevel /* elevel */ = EvalLevel::Full) {
        m_F = F;
        Base::setC(F.transpose() * F);
    }
    const Matrix &getDeformationGradient() const { return m_F; }

    Matrix denergy() const { return m_F * Base::PK2Stress(); }

    Real denergy(const Matrix &dF) const {
        return doubleContract(dF, denergy());
    }

    template<class Mat_>
    Matrix delta_denergy(const Mat_ &dF) const {
        return dF * Base::PK2Stress()
            + m_F * Base::delta_PK2Stress(symmetrized_x2(m_F.transpose() * dF));
    }

    Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        return doubleContract(delta_denergy(dF_lhs), dF_rhs);
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &dF_a, const Mat2_ &dF_b) const {
        return dF_a * Base::delta_PK2Stress(symmetrized_x2( m_F.transpose() * dF_b)) +
               dF_b * Base::delta_PK2Stress(symmetrized_x2( m_F.transpose() * dF_a)) +
               m_F * (Base::delta_PK2Stress(symmetrized_x2(dF_a.transpose() * dF_b)) +
                      Base::delta2_PK2Stress(dF_a, dF_b));
    }
private:
    using Base::setC; // Hide C-based interface to prevent deformation tensor from changing inconsistently with m_F
    Matrix m_F = Matrix::Identity(); // Note: `Matrix::Identity` works for non-square matrices (in the membrane case)

    // Hide F-based derivative interface to prevent confusion
    using Base::delta_PK2Stress;
    using Base::delta2_PK2Stress;
};

template<class Psi_F>
struct EnergyDensityCBasedFromFBased : public Psi_F {
    using Base = Psi_F;
    static constexpr size_t Dimension = Base::Dimension;
    static constexpr size_t N         = Base::N;
    using Real     = typename Base::Real;
    using Matrix   = Eigen::Matrix<Real, N, N>;
    using FInvType = std::remove_const_t<decltype(typename Base::Matrix().inverse().eval())>;

    void setC(const Matrix &C) {
        Base::setDeformationGradient(spdMatrixSqrt(C));
        m_Finv = Base::getDeformationGradient().inverse();
    }

    // Note: all Base constructors except the copy constructor initialize to
    // the identity deformation; this is compatible with our default member
    // initializer for m_Finv.
    using Base::Base;
    EnergyDensityCBasedFromFBased(const Base &b) : Base(b) { }

    EnergyDensityCBasedFromFBased(const EnergyDensityCBasedFromFBased &) = default;
    EnergyDensityCBasedFromFBased(const EnergyDensityCBasedFromFBased &other, const UninitializedDeformationTag &tag)
        : Base(other, tag), m_Finv(other.m_Finv) { }

    // d psi / d E,     E := 0.5 (C - I)
    Matrix PK2Stress() const { return m_Finv * Base::denergy(); }

    template<class Mat_>
    Matrix delta_PK2Stress(const Mat_ &dC) const {
        Matrix Finv_T_dC = m_Finv.transpose() * dC.matrix();
        return 0.5 * m_Finv * (Base::delta_denergy(Finv_T_dC) - Finv_T_dC * m_Finv * Base::denergy());
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_PK2Stress(const Mat_ &/* dC_a */, const Mat2_ &/* dC_b */) const {
        throw std::runtime_error("Unimplemented");
        return Matrix::Zero();
    }
private:
    // Hide F-based derivative interface to prevent confusion
    using Base::denergy;
    using Base::delta_denergy;
    using Base::d2energy;
    using Base::delta2_denergy;

    FInvType m_Finv = Matrix::Identity();
};

template<class Psi_F>
struct AutoHessianProjection : Psi_F {
    using Base = Psi_F;
    static constexpr size_t Dimension = Base::Dimension;
    static constexpr size_t N         = Base::N;
    using Real     = typename Base::Real;
    using Matrix   = Eigen::Matrix<Real, N, N>;
    using Vector   = Eigen::Matrix<Real, N, 1>;
    using VXd      = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using FInvType = std::remove_const_t<decltype(typename Base::Matrix().inverse().eval())>;
    using Hessian  = Eigen::Matrix<Real, N * N, N * N>;
    using ESolver  = Eigen::SelfAdjointEigenSolver<Hessian>;

    using Base::Base;
    AutoHessianProjection(const AutoHessianProjection &b, const UninitializedDeformationTag &tag)
        : Base(b, tag), projectionEnabled(b.projectionEnabled) { }

    using Base::energy;
    using Base::denergy;

    static std::string name() { return Base::name() + std::string("AutoProjected"); }

    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        Base::setDeformationGradient(F, elevel);

        // For efficiency, we only construct H and decompose H if we are
        // actually applying a Hessian projection.
        // See WARNING below!
        if (!projectionEnabled || elevel < EvalLevel::Hessian)
            return;

        // Evaluate the full Hessian by probing it on a basis with delta_denergy.
        Hessian H;
        VectorizedShapeFunctionJacobian<N, Vector> probe(0, Vector::Zero());
        for (size_t j = 0; j < N; ++j) {
            probe.g[j] = 1.0;
            for (size_t i = 0; i < N; ++i) {
                probe.c = i;
                auto delta_de = Base::delta_denergy(probe);
                // Column major flattening order to match `Matrix`!
                H.col(i + j * N) = Eigen::Map<const Eigen::Matrix<double, N * N, 1>>(delta_de.data());
            }
            probe.g[j] = 0.0;
        }

        if ((H - H.transpose()).squaredNorm() > 1e-10 * H.squaredNorm())
            throw std::runtime_error("Asymmetric probed Hessian");

        ESolver Hes(H);
        m_projectedHessian = Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(0.0).asDiagonal() * Hes.eigenvectors().transpose();
    }

    template<class Mat_>
    Matrix delta_denergy(const Mat_ &dF) const {
        if (projectionEnabled)
            return applyFlattened4thOrderTensor(m_projectedHessian, dF);
        return Base::delta_denergy(dF);
    }

    Real d2energy(const Matrix &dF_lhs, const Matrix &dF_rhs) const {
        return doubleContract(delta_denergy(dF_lhs), dF_rhs);
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &dF_a, const Mat2_ &dF_b) const {
        if (projectionEnabled) {
            // What is the use-case here?
            throw std::runtime_error("Derivatives of the projected Hessian not implemented");
        }
        return Base::delta2_denergy(dF_a, dF_b);
    }

    VXd eigenvalues() const { return ESolver(m_projectedHessian).eigenvalues(); }

    std::vector<Matrix> eigenmatrices() const {
        constexpr size_t n = N * N;
        std::vector<Matrix> result(n);

        ESolver es(m_projectedHessian);
        for (size_t i = 0; i < n; ++i)
            result[i] = Eigen::Map<const Matrix>(es.eigenvectors().col(i).data());
        return result;
    }

    const Hessian &projectedHessian() const { return m_projectedHessian; }

    // WARNING: changing this from `false` to `true` makes the result of
    // `delta2_denergy` undefined until the next call to
    // `setDeformationGradient`.
    bool projectionEnabled = true;

private:
    Hessian m_projectedHessian;
};

#endif /* end of include guard: EDENSITYADAPTORS_HH */
