////////////////////////////////////////////////////////////////////////////////
// AutodiffEDensity.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Enable the user to specify just the energy density expression and generate
//  the necessary derivatives automatically.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/22/2025 13:02:45
*///////////////////////////////////////////////////////////////////////////////
#ifndef AUTODIFFEDENSITY_HH
#define AUTODIFFEDENSITY_HH

#include <Eigen/Dense>
#include <MeshFEM/Utilities/NameMangling.hh>
#include "EnergyTraits.hh"

#ifdef MESHFEM_WITH_TINYAD
#include <TinyAD/Scalar.hh>
#else // !MESHFEM_WITH_TINYAD
#include <MeshFEM/AutomaticDifferentiation.hh>
#endif // MESHFEM_WITH_TINYAD

template<class Psi, typename Real_, size_t Dim_, EDensityType EDType_ = EDensityType::FBased>
struct AutodiffEDensity : public Psi {
    using Real = Real_;
    static_assert((EDType_ == EDensityType::FBased) || (EDType_ == EDensityType::Membrane), "AutodiffEDensity must be either an F-based or membrane energy");
    static constexpr EDensityType EDType = EDType_;
    static_assert(!((EDType == EDensityType::Membrane) && (Dim_ != 3)), "Membrane energy density must be a function of a 3x2 matrix");
    static constexpr size_t Dimension = Dim_;
    static constexpr size_t M = Dim_;
    static constexpr size_t N = (EDType == EDensityType::Membrane) ? 2 : Dim_;
    using Matrix = Eigen::Matrix<Real, M, N>;
    using Hessian = Eigen::Matrix<Real, M * N, M * N>;

#ifdef MESHFEM_WITH_TINYAD
    using ADScalar  = TinyAD::Scalar<M * N, Real_, /* with_hessian = */ false>;
    using AD2Scalar = TinyAD::Scalar<M * N, Real_, /* with_hessian = */  true>;
#else // !MESHFEM_WITH_TINYAD
    using ADScalar  = Eigen::AutoDiffScalar<Eigen::Matrix<Real,     M * N, 1>>;
    using AD2Scalar = Eigen::AutoDiffScalar<Eigen::Matrix<ADScalar, M * N, 1>>; // Use nested Eigen Autodiff type as a hack for second derivatives
#endif // MESHFEM_WITH_TINYAD

    static std::string name() {
        if constexpr (has_name_method<Psi>::value) {
            return Psi::name();
        } else {
            return get_name_of_type<Psi>() + std::string("AD");
        }
    }

    template<typename... Args>
    AutodiffEDensity(Args &&... args) : Psi(std::forward<Args>(args)...) { setDeformationGradient(Matrix::Identity()); }
    AutodiffEDensity(const AutodiffEDensity &other, UninitializedDeformationTag &&)
        : Psi(other),
          useAbsProjection(other.useAbsProjection),
          projectionDirection(other.projectionDirection) { }

    void setDeformationGradient(const Matrix &F, const EvalLevel elevel = EvalLevel::Full) {
        m_F = F;
        if (elevel == EvalLevel::EnergyOnly) { m_energy = Psi::psi(F); }
#ifdef MESHFEM_WITH_TINYAD
        if (elevel == EvalLevel::Gradient) {
            Eigen::Matrix<ADScalar, M, N> F_AD;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < M; ++i)
                    F_AD(i, j) = ADScalar(F(i, j), i + j * M);
            }
            ADScalar psi_AD = Psi::psi(F_AD);
            m_energy = psi_AD.val;
            m_denergy = Eigen::Map<const Matrix>(psi_AD.grad.data());
        }
        if (elevel >= EvalLevel::Hessian) {
            Eigen::Matrix<AD2Scalar, M, N> F_AD2;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < M; ++i)
                    F_AD2(i, j) = AD2Scalar(F(i, j), i + j * M);
            }

            AD2Scalar psi_AD2 = Psi::psi(F_AD2);
            m_energy   = psi_AD2.val;
            m_denergy  = Eigen::Map<const Matrix>(psi_AD2.grad.data());
            m_d2energy = Eigen::Map<const Hessian>(psi_AD2.Hess.data());
#else // !MESHFEM_WITH_TINYAD
        if (elevel == EvalLevel::Gradient) {
            Eigen::Matrix<ADScalar, M, N> F_AD;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < M; ++i) {
                    F_AD(i, j).value() = F(i, j);
                    F_AD(i, j).derivatives().setUnit(i + j * M);
                }
            }
            ADScalar psi_AD = Psi::psi(F_AD);
            m_energy = psi_AD.value();
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < M; ++i)
                    m_denergy(i, j) = psi_AD.derivatives()[i + j * M];
        }
        if (elevel >= EvalLevel::Hessian) {
            Eigen::Matrix<AD2Scalar, M, N> F_AD2;
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < M; ++i) {
                    F_AD2(i, j).value() = F(i, j);
                    F_AD2(i, j).value().derivatives().setUnit(i + j * M); // Initial derivative of the value
                    F_AD2(i, j).derivatives()        .setUnit(i + j * M); // Initial value of the derivative
                    // Initial Hessian is zero
                    for (size_t l = 0; l < N; ++l)
                        for (size_t k = 0; k < M; ++k)
                            F_AD2(i, j).derivatives()[k + l * M].derivatives().setZero();
                }
            }

            AD2Scalar psi_AD2 = Psi::psi(F_AD2);
            m_energy = psi_AD2.value().value();
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < M; ++i) {
                    m_denergy(i, j) = psi_AD2.derivatives()[i + j * M].value();

                    // Extract the Hessian into our column-major-flattened storage matrix.
                    for (size_t l = 0; l < N; ++l)
                        for (size_t k = 0; k < M; ++k)
                            m_d2energy(i + j * M, k + l * M) = psi_AD2.derivatives()[i + j * M].derivatives()[k + l * M];
                }
            }
#endif // MESHFEM_WITH_TINYAD
            if (projectionEnabled && (elevel != EvalLevel::HessianWithDisabledProjection)) {
                using ESolver = Eigen::SelfAdjointEigenSolver<Hessian>;
                // TODO: short-circuit in diagonally dominant case.
                ESolver Hes(m_d2energy);
                if (projectionDirection == ProjectionDirection::Positive) {
                    // Projecting to positive semidefinite
                    if (Hes.eigenvalues()[0] < 0.0) {
                        if (useAbsProjection)
                            m_d2energy = Hes.eigenvectors() * Hes.eigenvalues().cwiseAbs().asDiagonal() * Hes.eigenvectors().transpose();
                        else
                            m_d2energy = Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(0.0).asDiagonal() * Hes.eigenvectors().transpose();
                    }
                }
                else if (projectionDirection == ProjectionDirection::Negative) {
                    // Projecting to negative semidefinite
                    if (Hes.eigenvalues()[Hes.eigenvalues().size() - 1] > 0.0) {
                        if (useAbsProjection)
                            m_d2energy = Hes.eigenvectors() * (-(Hes.eigenvalues().cwiseAbs())).asDiagonal() * Hes.eigenvectors().transpose();
                        else
                            m_d2energy = Hes.eigenvectors() * Hes.eigenvalues().cwiseMin(0.0).asDiagonal() * Hes.eigenvectors().transpose();
                    }
                }
            }
        }
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

    // WARNING: changing this from `false` to `true` makes the result of
    // `delta2_denergy` undefined until the next call to
    // `setDeformationGradient`.
    bool projectionEnabled = true;
    bool useAbsProjection = false;

    // Whether to project the per-element Hessian to the positive or negative
    // semidefinite cone. This is helpful in the context of composite
    // objectives, J(e(x)), where when J'(x) is negative we want to project
    // d^2 e / dx^2 to be negative semidefinite.
    // This member is mutable so that the const `accumulateHessian` method of a
    // derived class of `MeshEnergy` can modify it if needed.
    enum class ProjectionDirection { Positive, Negative };
    mutable ProjectionDirection projectionDirection = ProjectionDirection::Positive;

private:
    Real m_energy;
    Matrix m_F, m_denergy;
    Hessian m_d2energy;
};

// Example:
struct SymmetricDirichletPsi {
    template<class Derived>
    typename Derived::Scalar psi(const Eigen::MatrixBase<Derived> &A) { // Don't use the `auto` return type here! We must evaluate the expression template before returning...
        auto J = A.determinant();
        if (J < 0) return typename Derived::Scalar(std::numeric_limits<double>::infinity());
        return 0.5 * (A.squaredNorm() + A.inverse().squaredNorm());
    }
};

template<typename Real_, size_t Dim_>
using SymmetricDirichletDerivativeFree = AutodiffEDensity<SymmetricDirichletPsi, Real_, Dim_>;

#endif /* end of include guard: AUTODIFFEDENSITY_HH */
