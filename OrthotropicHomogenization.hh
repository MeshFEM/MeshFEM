////////////////////////////////////////////////////////////////////////////////
// OrthotropicHomogenization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Homogenization routines taking advantage of orthotropic symmetry by
//      analyzing only the orthotropic base cell. This cuts the matrix size in
//      four for 2D, eight for 3D.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/27/2016 22:36:09
////////////////////////////////////////////////////////////////////////////////
#ifndef ORTHOTROPICHOMOGENIZATION_HH
#define ORTHOTROPICHOMOGENIZATION_HH

#include <SparseMatrices.hh>
#include <vector>
#include <memory>
#include <stdexcept>

#include "PeriodicBoundaryMatcher.hh"

namespace PeriodicHomogenization {
namespace Orthotropic {

// WARNING: ONLY WORKS WITH ORTHOTROPIC BASE MATERIAL
template<class _Sim>
void solveCellProblems(std::vector<typename _Sim::VField> &w_ij, _Sim &sim,
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

    // Stretching probe:
    // w^ii(x)_c = 0 on reflection plane c (plane with normal e_c)
    fixedVars.clear();
    auto stretchSystem = Future::make_unique<SPSDSystem<Real>>(K);
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
    //      w^ij(x)_{j != c} = 0   (two components: j = s, j!=c && j!=s)
    for (size_t s = 0; s < flatLen(N) - N; ++s) {
        auto shearSystem = Future::make_unique<SPSDSystem<Real>>(K);
        fixedVars.clear();
        // Note: nodes lying on the edges/corners may have more than one plane
        // trying to fix a particular coordinate; we could explicitly detect
        // this, but it's easier to just do a union of the fixVar set.
        std::vector<bool> fixVar(N * mesh.numNodes(), false);
        for (auto bn : mesh.boundaryNodes()) {
            const size_t ni = bn.volumeNode().index();
            for (size_t c = 0; c < N; ++c) {
                if (nodeFaceMemberships[bn.index()].onMinOrMaxFace(c)) {
                    // coordinate perpendicular to shear plane is always fixed
                    fixVar.at(N * ni + s) = true;;
                    // in 3D, fix coordinate equal to neither c nor s.
                    if ((N == 3) && (c != s)) fixVar.at(N * ni + (N - (c + s))) = true;
                }
            }
        }
        for (size_t i = 0; i < fixVar.size(); ++i)
            if (fixVar[i]) fixedVars.push_back(i);

        fixedVarValues.assign(fixedVars.size(), 0.0);
        shearSystem->fixVariables(fixedVars, fixedVarValues);
        probeSystems.push_back(std::move(shearSystem));
    }

    std::vector<VectorField<Real, N>> l;
    l.reserve(flatLen(N));

    // Compute the constant strain loads
    for (size_t ij = 0; ij < flatLen(N); ++ij) {
        auto e_ij = -_Sim::SMatrix::CanonicalBasis(ij);
        l.push_back(sim.constantStrainLoad(e_ij));
        auto &l_ij = l[ij];
        continue;

        // On the reflection planes, we must also accumulate contributions from
        // the reflected elements:
        // ACTUALLY: NOT REALLY??
        for (size_t c = 0; c < N; ++c) {
            Eigen::Matrix<Real, N, N> R;
            R.setIdentity();
            R(c, c) = -1.0;
            // Strain in reflected element.
            auto e_reflect = e_ij.transform(R);

            typename _Sim::Mesh::ElementData::ElementLoad eLoad;
            for (auto e : mesh.elements()) {
                bool elementOnPlane = false;
                for (auto n : e.nodes()) {
                    auto bn = n.boundaryNode();
                    if (!bn) continue;
                    if (nodeFaceMemberships[bn.index()].onMinOrMaxFace(c))
                        elementOnPlane = true;
                }
                if (elementOnPlane) {
                    // Distribute load from reflected element to plane nodes.
                    e->perElementConstantStrainLoad(e_reflect, eLoad);
                    for (auto n : e.nodes()) {
                        auto bn = n.boundaryNode();
                        if (!bn) continue;
                        // Note: load must be reflected.
                        // Three reflections are effectively being applied here.
                        // Four reflections (a nop) would correspond to
                        // reflecting the FEM trial displacement function, which
                        // we don't want.
                        if (nodeFaceMemberships[bn.index()].onMinOrMaxFace(c))
                            l_ij(n.index()) += R * eLoad.col(n.localIndex());
                    }
                }
            }
        }
    }

    // MSHFieldWriter writer("adjustment_debug.msh", mesh);
    // for (size_t ij = 0; ij < flatLen(N); ++ij) {
    //     writer.addField("l_ij " + std::to_string(ij), l_ij);
    // }

    for (size_t ij = 0; ij < flatLen(N); ++ij) {
        if (ij < N) w_ij.push_back(probeSystems.at(         0)->solve(l[ij]));
        else        w_ij.push_back(probeSystems.at(ij - N + 1)->solve(l[ij]));
    }
}

template<class _Sim>
typename _Sim::ETensor homogenizedElasticityTensorDisplacementForm(
        const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim,
        Real baseCellVolume = 0.0) {
    constexpr size_t N = _Sim::N;
    const auto &mesh = sim.mesh();
    if (baseCellVolume == 0.0) baseCellVolume = mesh.boundingBox().volume();
    using SMatrix = typename _Sim::SMatrix ;
    constexpr size_t numStrains = SMatrix::flatSize();
    assert(w_ij.size() == numStrains);

    // Assume elasticity tensor is constant over the entire base cell
    const typename _Sim::ETensor &EBase = mesh.element(0)->E();

    typename _Sim::ETensor Eh;
    SMatrix nw_pq;

    // Displacement restricted to a boundary element
    Interpolant<VectorND<N>, _Sim::K - 1, _Sim::Degree> w_be;
    for (auto be : mesh.boundaryElements()) {
        typename _Sim::ETensor Econtrib;
        const auto &n = be->normal();
        for (size_t i = 0; i < w_ij.size(); ++i) {
            const auto &w = w_ij[i];
            // Copy the boundary node displacements into interpolant
            for (size_t ni = 0; ni < w_be.size(); ++ni)
                w_be[ni] = w(be.node(ni).volumeNode().index());
            auto w_be_int = w_be.integrate(be->volume());
            
            nw_pq.clear();

            constexpr size_t NReflectedCells = 1 << N;
            Real scale = 1.0 / NReflectedCells;
            for (size_t r = 0; r < NReflectedCells; ++r) {
                // Which components of the fluctuation displacements and normals are reflected?
                // TODO: change order of be/ij loops and factor this out.
                VectorND<N> normalReflect, displacementReflect;
                for (size_t c = 0; c < N; ++c) {
                    normalReflect[c]       = (r & (1 << c)) ? -1.0 : 1.0;
                    // stretching probe: displacement reflects across each plane
                    if (i < N) displacementReflect[c] = normalReflect[c];
                    else {
                        // shearing probe: displacement reflects across shearing plane (s)
                        const size_t sPlane = i - N;
                        displacementReflect[c] = (c == sPlane) ? normalReflect[c] : 1.0;
                        // displacement negates and reflects within shearing plane
                        for (size_t op = 1; op < N; ++op) {
                            const size_t otherPlane = (sPlane + op) % N;
                            if (c == otherPlane) continue; // net effect of negate + reflect is zero for the reflected component
                            // the other components get negated
                            if (r & (1 << otherPlane)) displacementReflect[c] *= -1.0;
                        }
                    }
                }

                for (size_t p = 0; p < N; ++p)
                    for (size_t q = p; q < N; ++q)
                        nw_pq(p, q) += 0.5 * scale * (displacementReflect[p] * normalReflect[q] * w_be_int[p] * n[q]
                                                    + displacementReflect[q] * normalReflect[p] * w_be_int[q] * n[p]);
            }

            Eh.DRowAsSymMatrix(i) += EBase.doubleContract(nw_pq);
        }
    }

    Eh += EBase * mesh.volume();
    Eh /= baseCellVolume;

    return Eh;
}

} // Orthotropic
} // PeriodicHomogenization

#endif /* end of include guard: ORTHOTROPICHOMOGENIZATION_HH */
