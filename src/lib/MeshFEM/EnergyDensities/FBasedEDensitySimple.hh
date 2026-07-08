////////////////////////////////////////////////////////////////////////////////
// FBasedEDensitySimple.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A convenience base class to ease the specification of "F-based energy
//  densities" `psi(F)` where `F` is an NxN deformation gradient.
//
//  While *any* C++ class can be used as an F-based energy if it implements the
//  required methods, writing a standalone class like this involves substantial
//  boilerplate that is unnecessary in the case where first and second
//  derivatives of `psi` are to be implemented as dense tensor evaluations.
//
//  In this case, the user can derive from `FBasedEDensitySimple` and
//  simply complete the `m_eval` method, filling in member variables
//  m_energy, m_denergy, and m_d2energy based on the deformation gradient
//  stored in `m_F`.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/28/2025 16:17:47
*///////////////////////////////////////////////////////////////////////////////
#ifndef FBASEDEDENSITYSIMPLE_HH
#define FBASEDEDENSITYSIMPLE_HH

#include "EnergyTraits.hh"
#include <Eigen/Dense>

namespace MeshFEM {

template<typename _Real, size_t _Dim>
struct FBasedEDensitySimple {
    static constexpr size_t Dimension = _Dim;
    static constexpr size_t M = _Dim;
    static constexpr size_t N = _Dim;
    using Real    = _Real;
    using Matrix  = Eigen::Matrix<Real, M, N>;
    using Hessian = Eigen::Matrix<Real, M * N, M * N>;

    static constexpr EDensityType EDType = EDensityType::FBased;

    // Note: we can't just call `setDeformationGradient` here since the derived
    // class hasn't been constructed yet (making `m_eval` pure virtual). Instead
    // we zero-initialize energy, stress and Hessian, and identity-initialize F.
    FBasedEDensitySimple() :
        m_F(Matrix::Identity()), m_energy(0), m_denergy(Matrix::Zero()), m_d2energy(Hessian::Zero())
    { }

    FBasedEDensitySimple(const FBasedEDensitySimple &other, UninitializedDeformationTag &&) { }

    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        m_F = F;
        bool projectHessian = (elevel != EvalLevel::HessianWithDisabledProjection);
        m_eval(F, elevel, projectHessian);
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const {
        throw std::runtime_error("Unimplemented.");
    }

    const Matrix &getDeformationGradient() const { return m_F; }

    Real           energy() const { return m_energy; }
    const Matrix &denergy() const { return m_denergy; }
    Real denergy(const Matrix &dF) const { return doubleContract(dF, denergy()); }

    template<typename Mat_>
    Matrix delta_denergy(const Mat_ &dF) const { return applyFlattened4thOrderTensor(m_d2energy, dF); }

    Real d2energy(const Matrix& dF_a, const Matrix& dF_b) const { return doubleContract(dF_a, delta_denergy(dF_b)); }

    const Hessian &d2energy() const { return m_d2energy; }

    Matrix PK2Stress() const { return m_F.inverse().transpose() * denergy(); }

protected:
    Real m_energy;
    Matrix m_F, m_denergy;
    Hessian m_d2energy;

    virtual void m_eval(const Matrix &F, EvalLevel elevel, bool projectHessian) = 0;
};


} // namespace MeshFEM

#endif /* end of include guard: FBASEDEDENSITYSIMPLE_HH */
