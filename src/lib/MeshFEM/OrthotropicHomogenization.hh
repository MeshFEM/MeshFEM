////////////////////////////////////////////////////////////////////////////////
// OrthotropicHomogenization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Homogenization routines taking advantage of orthotropic symmetry by
//      analyzing only the orthotropic base cell. This cuts the matrix size in
//      four for 2D, eight for 3D.
//      These routines assume an orthotropic base material.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/27/2016 22:36:09
////////////////////////////////////////////////////////////////////////////////
#ifndef ORTHOTROPICHOMOGENIZATION_HH
#define ORTHOTROPICHOMOGENIZATION_HH

#include <MeshFEM/SparseMatrices.hh>
#include <vector>
#include <memory>
#include <stdexcept>
#include <bitset>

#include <MeshFEM/PeriodicBoundaryMatcher.hh>
#include <MeshFEM/PeriodicHomogenization.hh>

// WARNING: ONLY WORKS WITH ORTHOTROPIC BASE MATERIAL
namespace PeriodicHomogenization {
namespace Orthotropic {

////////////////////////////////////////////////////////////////////////////
/*! Solve the linear elasticity periodic homogenization cell problems for
//  each constant strain e^ij:
//       -div E : [ strain(w^ij) + e^ij ] = 0 in omega
//        n . E : [ strain(w^ij) + e^ij ] = 0 on omega's boundary
//        w^ij periodic
//        w^ij = 0 on arbitrary internal node ("pin" no rigid translation constraint)
//  @param[out]   w_ij   Fluctuation displacements (cell problem solutions)
//  @param[inout] sim    Linear elasticity simulator for omega.
//  @return     The SPD cell problem systems (one for stretches, 3 for shears)
//  Warning: this function mutates sim by removing periodic and pin
//           constraints.
*///////////////////////////////////////////////////////////////////////////
template<class _Sim>
std::vector<std::unique_ptr<SPSDSystem<Real>>>
solveCellProblems(std::vector<typename _Sim::VField> &w_ij, _Sim &sim,
                  Real cellEpsilon = 1e-7) {
    constexpr size_t N = _Sim::N;

    // Orthotropic homogenization doesn't need periodicity/NRM constraints
    // (Instead particular vars on the symmetry planes will be fixed at zero)
    sim.removePeriodicConditions();
    sim.removeNoRigidMotionConstraint();

    typename _Sim::TMatrix K, C;
    std::vector<Real> constraintRHS, fixedVarValues;
    std::vector<size_t> fixedVars;
    sim.assembleConstrainedSystem(K, C, constraintRHS, fixedVars, fixedVarValues, true);
    size_t numInitialConstraints = C.m + constraintRHS.size() + fixedVarValues.size() + fixedVarValues.size();
    if (numInitialConstraints > 0) throw std::runtime_error("Constraints unexpected.");

    K.sumRepeated();

    // There is a single system for all N stretching basis probes,
    // then one for each shearing basis probe.
    std::vector<std::unique_ptr<SPSDSystem<Real>>> probeSystems;

    const auto &mesh = sim.mesh();
    const auto &cell = mesh.boundingBox();

    using FM = PeriodicBoundaryMatcher::FaceMembership<_Sim::N>;
    std::vector<FM> nodeFaceMemberships;
    nodeFaceMemberships.reserve(mesh.numBoundaryNodes());
    for (auto bn : mesh.boundaryNodes())
        nodeFaceMemberships.emplace_back(bn.volumeNode()->p, cell, cellEpsilon);

    // Manually determine the internal faces (those on the orthotropic base cell faces).
    // These are analogous to the periodic boundary elements in the triply
    // periodic base cell case.
    auto isInternalBE = PeriodicBoundaryMatcher::determineCellFaceBoundaryElements(mesh, nodeFaceMemberships);
    for (auto be : sim.mesh().boundaryElements())
        be->isInternal = isInternalBE.at(be.index());

    // Stretching probe:
    // w^ii(x)_c = 0 on reflection plane c (plane with normal e_c)
    fixedVars.clear();
    BENCHMARK_START_TIMER("Make SPSDSystem");
    auto stretchSystem = Future::make_unique<SPSDSystem<Real>>(K);
    BENCHMARK_STOP_TIMER("Make SPSDSystem");
    for (auto bn : mesh.boundaryNodes()) {
        for (size_t c = 0; c < N; ++c)
            if (nodeFaceMemberships[bn.index()].onMinOrMaxFace(c))
                fixedVars.push_back(N * bn.volumeNode().index() + c);
    }
    fixedVarValues.assign(fixedVars.size(), 0.0);
    stretchSystem->fixVariables(fixedVars, fixedVarValues);
    probeSystems.push_back(std::move(stretchSystem));

    // Shearing probes:
    // 3D: shear probe s 0, 1, 2 ==> indices ij = 12, 20, 01
    // 2D: shear probe s 0       ==> indices ij = 01
    // For reflection planes parallel to the probe shear plane (s),
    //      w^ij(x)_c = 0 (plane with normal e_c)
    //      Note: this case only happens in 3D, where c = s
    // For reflection planes c perpendicular to the shear plane (c != s)
    //      w^ij(x)_{j != c} = 0   (two components in 3D: j = s, j!=c && j!=s)
    for (size_t s = 0; s < flatLen(N) - N; ++s) {
        BENCHMARK_START_TIMER("Make SPSDSystem");
        auto shearSystem = Future::make_unique<SPSDSystem<Real>>(K);
        BENCHMARK_STOP_TIMER("Make SPSDSystem");
        fixedVars.clear();
        // Note: nodes lying on the edges/corners may have more than one plane
        // trying to fix a particular coordinate; we could explicitly detect
        // this, but it's easier to just do a union of the fixVar set.
        // (We must ensure fixVar doesn't contain duplicates.)
        std::vector<bool> fixVar(N * mesh.numNodes(), false);
        for (auto bn : mesh.boundaryNodes()) {
            const size_t ni = bn.volumeNode().index();
            for (size_t c = 0; c < N; ++c) {
                if (nodeFaceMemberships[bn.index()].onMinOrMaxFace(c)) {
                    if (N == 3) {
                        // always fix coordinate perpendicular to shear plane (whether or not c == s)
                        fixVar.at(N * ni + s) = true;
                        // For reflections in shear plane, also fix coordinate equal to neither c nor s.
                        if (c != s) fixVar.at(N * ni + (N - (c + s))) = true;
                    }
                    else {
                        // In 2D all reflections are in the shear plane; fix the
                        // coordinate not equal to c.
                        fixVar.at(N * ni + (c == 0)) = true;
                    }
                }
            }
        }
        for (size_t i = 0; i < fixVar.size(); ++i)
            if (fixVar[i]) fixedVars.push_back(i);

        fixedVarValues.assign(fixedVars.size(), 0.0);
        shearSystem->fixVariables(fixedVars, fixedVarValues);
        probeSystems.push_back(std::move(shearSystem));
    }

    // Compute the constant strain loads
    BENCHMARK_START_TIMER("Constant Strain Load");
    std::vector<VectorField<Real, N>> l;
    l.reserve(flatLen(N));
    for (size_t ij = 0; ij < flatLen(N); ++ij) {
        auto e_ij = -_Sim::SMatrix::CanonicalBasis(ij);
        l.emplace_back(sim.constantStrainLoad(e_ij));
    }
    BENCHMARK_STOP_TIMER("Constant Strain Load");

    // Solve the cell problems.
    w_ij.reserve(flatLen(N));
    for (size_t ij = 0; ij < flatLen(N); ++ij) {
        // std::cerr << "Solving cell problem " << ij << std::endl;
        if (ij < N) w_ij.emplace_back(probeSystems.at(         0)->solve(l[ij]));
        else        w_ij.emplace_back(probeSystems.at(ij - N + 1)->solve(l[ij]));
    }

    return probeSystems;
}

constexpr inline size_t numReflectedCells(size_t N) { return 1 << N; }

// The shearing fluctuation displacements negate for reflections in the shearing
// plane. (The sign is negative when the number of reflections is odd).
// Determine this sign for shear probe "ij" and reflection "r" The bits of r
// determine which of the N coordinate reflections are applied.
template<size_t N>
Real fluctuationDisplacementSign(size_t ij, size_t r) {
    if (ij < N) return 1.0;
    std::bitset<N> isReflected(r);

    if (N == 3) {
        // Don't care about reflections orthogonal to shear plane in 3D
        size_t sPlane = ij - N;
        isReflected.reset(sPlane);
    }

    return (isReflected.count() == 1) ? -1.0 : 1.0;
}

// Compute the full periodic homogenized elasticity tensor from the orthotropic
// base cell quantity. The homogenized tensor is expressed as an integral over
// the full period cell. But for orthotropic patterns, we can instead integrate
// only over the orthotropic sub-base-cell (what happens when we call
// PeriodicHomogenization::homogenizedElasticityTensor) and then reconstruct the
// full integral by appropriate transformations.
template<size_t N>
ElasticityTensor<Real, N>
homogenizedTensorFromOrthoCellQuantity(const ElasticityTensor<Real, N> &EhO) {
    ElasticityTensor<Real, N> Eh;
    for (size_t r = 0; r < numReflectedCells(N); ++r) {
        for (size_t kl = 0; kl < flatLen(N); ++kl) {
            Real s_kl = fluctuationDisplacementSign<N>(kl, r);
            for (size_t ij = 0; ij <= kl; ++ij) {
                Real s_ij = fluctuationDisplacementSign<N>(ij, r);
                Eh.D(ij, kl) += s_ij * s_kl * EhO.D(ij, kl);
            }
        }
    }

    Eh *= 1.0 / numReflectedCells(N);
    return Eh;
}

template<class _Sim>
typename _Sim::ETensor homogenizedElasticityTensorDisplacementForm(
        const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim,
        Real baseCellVolume = 0.0) {
    auto EhOrtho = PeriodicHomogenization::homogenizedElasticityTensorDisplacementForm(
                    w_ij, sim, baseCellVolume);
    return homogenizedTensorFromOrthoCellQuantity(EhOrtho);
}

template<class _Sim>
typename _Sim::ETensor homogenizedElasticityTensor(
        const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim,
        Real baseCellVolume = 0.0) {
    auto EhOrtho = PeriodicHomogenization::homogenizedElasticityTensor(w_ij, sim, baseCellVolume);
    return homogenizedTensorFromOrthoCellQuantity(EhOrtho);
}

// Compute the exact derivative of the full homogenized elasticity tensor with
// respect to each mesh vertex position.
template<class _Sim>
OneForm<typename _Sim::ETensor, _Sim::N>
homogenizedElasticityTensorDiscreteDifferential(
        const std::vector<typename _Sim::VField> &w,
        const _Sim &sim)
{
    using ET = typename _Sim::ETensor;
    auto dEhOrtho = PeriodicHomogenization::homogenizedElasticityTensorDiscreteDifferential(w, sim);
    return compose([](const ET &e) { return homogenizedTensorFromOrthoCellQuantity(e); }, dEhOrtho);
}

} // Orthotropic
} // PeriodicHomogenization

#endif /* end of include guard: ORTHOTROPICHOMOGENIZATION_HH */
