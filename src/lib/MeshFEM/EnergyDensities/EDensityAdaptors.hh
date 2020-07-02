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

    void setDeformationGradient(const Matrix &F) {
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

#endif /* end of include guard: EDENSITYADAPTORS_HH */
