////////////////////////////////////////////////////////////////////////////////
// StressToBiotStrain.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Determine a Biot strain under which the material responds with a given
//  PK1 stress `s` by solving:
//
//      e(s) := (argmin_{F \in sym(NxN)} psi(F) - e : s) - I
//
//  Note that existence and uniqueness are not guaranteed for nonlinear
//  material models. Howver, in the case of linearly elastic energy density
//  functions `psi`, this is equivalent to double contracting with the
//  compliance tensor (and subtracting the identity matrix).
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  12/21/2022 15:25:17
*///////////////////////////////////////////////////////////////////////////////
#ifndef STRESSTOBIOTSTRAIN_HH
#define STRESSTOBIOTSTRAIN_HH

#include "EnergyTraits.hh"
#include <MeshFEM/newton_optimizer/dense_newton.hh>
#include <MeshFEM/Flattening.hh>

template<class Psi>
struct StressToBiotStrainProblem {
    static_assert(Psi::EDType == EDensityType::FBased, "Only implemented for F-based energy densities.");
    static constexpr size_t N = Psi::N;
    using Real = typename Psi::Real;

    // The optimization variables are the *upper tri* of F.
    static constexpr size_t nvars = flatLen(N);
    using HessType = Eigen::Matrix<Real, nvars, nvars>;
    using MNd      = Eigen::Matrix<Real, N, N>;
    using VarType  = Eigen::Matrix<Real, nvars, 1>;

    StressToBiotStrainProblem(const Psi &psi, const MNd &PK1Stress)
        : m_psi(psi, UninitializedDeformationTag()), m_PK1Stress(PK1Stress) {
        m_psi.setDeformationGradient(MNd::Identity());
    }

    static MNd unflatten(const VarType &vars) {
        MNd result;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                result(i, j) = vars[flattenIndices<N>(i, j)];
        return result;
    }

    // Flatten the symmetrized matrix 0.5 (A + A^T)
    static VarType flattenSym(const MNd &A) {
        VarType result;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = i; j < N; ++j)
                result[flattenIndices<N>(i, j)] = 0.5 * (A(i, j) + A(j, i));
        return result;
    }

    // Flatten a potentially asymmetric matrix by summing the off-diagonal entries.
    // This ensures `flattenSym(A) . flattenSum(B) = A:B`
    static VarType flattenSum(const MNd &A) {
        VarType result;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = i; j < N; ++j)
                result[flattenIndices<N>(i, j)] = A(i, j) + A(j, i);
        return result;
    }

    void setVars(const VarType &vars) {
        m_psi.setDeformationGradient(unflatten(vars));
    }

    VarType getVars() const {
        return flattenSym(m_psi.getDeformationGradient());
    }

    Real energy() const {
        return m_psi.energy() - doubleContract(m_PK1Stress, m_psi.getDeformationGradient());
    }

    VarType gradient() const {
        return flattenSum((m_psi.denergy() - m_PK1Stress).eval());
    }

    HessType hessian() const {
        HessType result;

        MNd delta_F = MNd::Zero();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i; j < N; ++j) {
                delta_F(i, j) = 1;
                delta_F(j, i) = 1;

                MNd delta_stress = m_psi.delta_denergy(delta_F);
                result.col(flattenIndices<N>(i, j)) = flattenSum(delta_stress);

                delta_F(i, j) = 0;
                delta_F(j, i) = 0;
            }
        }
        return result;
    }

    static MNd solve(const Psi &psi, const MNd &PK1Stress, double gradTol = 1e-9, bool verbose = false) {
        StressToBiotStrainProblem prob(psi, PK1Stress);
        dense_newton(prob, /* maxIter = */ 100, gradTol, verbose);
        return prob.m_psi.getDeformationGradient() - MNd::Identity();
    }

private:
    Psi m_psi;
    MNd m_PK1Stress;
};

template<class Psi>
auto stressToBiotStrain(const Psi &psi, const typename StressToBiotStrainProblem<Psi>::MNd &PK1Stress, double gradTol = 1e-9, bool verbose = false) {
    return StressToBiotStrainProblem<Psi>::solve(psi, PK1Stress, gradTol, verbose);
}

template<class Psi>
auto stressToDefGrad(const Psi &psi, const typename StressToBiotStrainProblem<Psi>::MNd &PK1Stress, double gradTol = 1e-9, bool verbose = false) {
    return (StressToBiotStrainProblem<Psi>::solve(psi, PK1Stress, gradTol, verbose) + StressToBiotStrainProblem<Psi>::MNd::Identity()).eval();
}

#endif /* end of include guard: STRESSTOBIOTSTRAIN_HH */
