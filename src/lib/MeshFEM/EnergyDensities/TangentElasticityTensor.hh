// ////////////////////////////////////////////////////////////////////////////////
// TangentElasticityTensor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Routines for volumetric energy densities to compute the tangent elasticity
// tensor around a particular deformation. Around the identity deformation,
// this should give the elasticity tensor defining the corresponding
// linear elasticity model.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/01/2020 18:37:12
////////////////////////////////////////////////////////////////////////////////
#ifndef TANGENTELASTICITYTENSOR_HH
#define TANGENTELASTICITYTENSOR_HH

// The following `tangentElasticityTensor` routines for volumetric energy
// densities, compute the tangent elasticity tensor around a particular
// deformation. Around the identity deformation, this should give the
// elasticity tensor defining the linearized elasticity model.
template<class Psi_C, std::enable_if_t<Psi_C::EDType == EDensityType::CBased, int> = 0>
ElasticityTensor<typename Psi_C::Real, Psi_C::N>
tangentElasticityTensor(const Psi_C &psiIn, const Eigen::Matrix<typename Psi_C::Real, Psi_C::N, Psi_C::N> &C = Eigen::Matrix<typename Psi_C::Real, Psi_C::N, Psi_C::N>::Identity()) {
    using _Real = typename Psi_C::Real;
    constexpr size_t N = Psi_C::N;
    using SMatrix = SymmetricMatrixValue<_Real, N>;

    static_assert(Psi_C::EDType == EDensityType::CBased, "Psi_C must be C-based");
    Psi_C psi(psiIn, UninitializedDeformationTag());
    psi.setC(C);
    ElasticityTensor<_Real, N> result;
    for (size_t kl = 0; kl < flatLen(N); ++kl) {
        // dC = 2 * d strain (strain := 0.5 (C - I))
        result.DColAsSymMatrix(kl) = psi.delta_PK2Stress(2.0 * SMatrix::CanonicalBasis(kl));
    }
    return result;
}

template<class Psi_F, std::enable_if_t<Psi_F::EDType == EDensityType::FBased, int> = 0>
ElasticityTensor<typename Psi_F::Real, Psi_F::N>
tangentElasticityTensor(const Psi_F &psiIn, const Eigen::Matrix<typename Psi_F::Real, Psi_F::N, Psi_F::N> &F = Eigen::Matrix<typename Psi_F::Real, Psi_F::N, Psi_F::N>::Identity()) {
    using Psi_C = EnergyDensityCBasedFromFBased<Psi_F>;
    return tangentElasticityTensor<Psi_C>(Psi_C(psiIn), (F.transpose() * F).eval());
}

template<class Psi_F, std::enable_if_t<Psi_F::EDType == EDensityType::Membrane, int> = 0>
ElasticityTensor<typename Psi_F::Real, Psi_F::N>
tangentElasticityTensor(const Psi_F &/* psiIn */, const Eigen::Matrix<typename Psi_F::Real, Psi_F::N, Psi_F::N> &/* F */ = Eigen::Matrix<typename Psi_F::Real, Psi_F::N, Psi_F::N>::Identity()) {
    static_assert(Psi_F::EDType != EDensityType::Membrane, "tangentElasticityTensor not supported for membrane densities");
}

#endif /* end of include guard: TANGENTELASTICITYTENSOR_HH */
