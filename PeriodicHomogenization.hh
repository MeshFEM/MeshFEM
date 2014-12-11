#ifndef PERIODICHOMOGENIZATION_HH
#define PERIODICHOMOGENIZATION_HH

#include "GaussQuadrature.hh"
#include <vector>
#include <string>

namespace PeriodicHomogenization {
    template<class _Sim>
    void solveCellProblems(std::vector<typename _Sim::VField> &w_ij, _Sim &sim)
    {
        typedef typename _Sim::VField  VField;
        typedef typename _Sim::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();

        BENCHMARK_START_TIMER("Apply Cell Conditions");
        sim.applyPeriodicConditions();
        sim.applyNoRigidMotionConstraint();
        sim.setUsePinNoRigidTranslationConstraint(true);
        BENCHMARK_STOP_TIMER("Apply Cell Conditions");

        w_ij.reserve(numStrains), w_ij.clear();
        for (size_t i = 0; i < numStrains; ++i) {
            BENCHMARK_START_TIMER("Constant Strain Load");
            VField rhs(sim.constantStrainLoad(-SMatrix::CanonicalBasis(i)));
            BENCHMARK_STOP_TIMER("Constant Strain Load");
            w_ij.push_back(sim.solve(rhs));
        }
    }

    template<class _Sim>
    typename _Sim::ETensor homogenizedElasticityTensor(
            const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim) {
        const auto &mesh = sim.mesh();
        typedef typename _Sim::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        assert(w_ij.size() == numStrains);

        // Compute homogenized elasticity tensor (stress-like version):
        // Eh_ijkl = 1/|Y| int_w [E : strain(w_ij)]_kl + E_ijkl dV
        // Where |Y| = Yvol = periodic cell (grid bounding box) volume
        //        w  = periodic base cell geometry
        typename _Sim::ETensor Eh;
        typename _Sim::Strain  strain_ij;
        for (size_t ei = 0; ei < mesh.numElements(); ++ei) {
            typename _Sim::ETensor Econtrib;
            for (size_t i = 0; i < w_ij.size(); ++i) {
                sim.elementStrain(ei, w_ij[i], strain_ij);
                Econtrib.DRowAsSymMatrix(i) =
                    mesh.element(ei)->E().doubleContract(strain_ij.average());
            }
            // Elasticity tensor is always constant on each element.
            Econtrib += mesh.element(ei)->E();
            Econtrib *= mesh.element(ei)->volume();
            Eh += Econtrib;
        }
        Eh /= mesh.boundingBox().volume();
        return Eh;

        // // The following "energy-like" version is equivalent to the more efficient
        // // "stress-like" version above:
        // // Eh_ijkl = 1/|Y| int_w <E (e(w_ij) + e_ij), e(w_kl) + e_kl> dV,
        // typename _Sim::ETensor EhE;
        // typename _Sim::Strain  strain_ij, strain_kl;
        // for (size_t ei = 0; ei < mesh.numElements(); ++ei) { 
        //     auto e = mesh.element(ei);
        //     for (size_t ij = 0; ij < numStrains; ++ij) {
        //         sim.elementStrain(ei, w_ij[ij], strain_ij);
        //         strain_ij += SMatrix::CanonicalBasis(ij);
        //         for (size_t kl = ij; kl < numStrains; ++kl) {
        //             sim.elementStrain(ei, w_ij[kl], strain_kl);
        //             strain_kl += SMatrix::CanonicalBasis(kl);
        //             EhE.D(ij, kl) +=
        //                 _Sim::template VolInt<2 * (_Sim::Degree - 1)>::integrate(
        //                     [&] (const VectorND<_Sim::numElemVertices> &p) {
        //                         return e->E().doubleContract(strain_ij(p))
        //                                      .doubleContract(strain_kl(p));
        //                     }, e->volume());
        //         }
        //     }
        // }
        // EhE /= mesh.boundingBox().volume();

        // return EhE;
    }


    // Per-boundary-element interpolant type needed to express the homogenized
    // tensor shape derivative.
    template<class _Sim>
    using BEHTensorGradInterpolant = Interpolant<typename _Sim::ETensor,
        _Sim::K - 1, 2 * (_Sim::Degree - 1)>;

    ////////////////////////////////////////////////////////////////////////////
    /*! Computes the steepest ascent direction (i.e. the theta maximizing the
    //  shape derivative DS[theta]) of each component of the homogenized
    //  elasticity tensor. This is a per-boundary-element piecewise constant
    //  (FEM degree 1) or quadratic (FEM degree 2) rank 4 tensor field.
    //  @param[in]  w       fluctuation displacements (cell problem solutions)
    //  @param[in]  sim     linear elasticity solver
    //  @return     per-boundary-element rank 4 tensor field.
    *///////////////////////////////////////////////////////////////////////////
    template<class _Sim>
    std::vector<BEHTensorGradInterpolant<_Sim>>
    homogenizedElasticityTensorGradient(
            const std::vector<typename _Sim::VField> &w, const _Sim &sim) {
        typedef typename _Sim::ETensor ETensor;
        typedef typename _Sim::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        constexpr size_t K = _Sim::K;
        constexpr size_t Deg = _Sim::Degree;
        assert(w.size() == numStrains);

        const auto &mesh = sim.mesh();
        // Shape derivative evaluated on normal velocity v_n:
        // DS(E_H)[v_n n] = int_dt <E [e_ij + e(w_ij)], e_kl + e(w_kl)> v_n dA
        // So the steepest ascent direction is to evolve with
        //      v_n(x) = <E [e_ij + e(w_ij)], e_kl + e(w_kl)> := G_ijkl(x)
        // for each non-periodic boundary point x.
        //      DS_ijkl(y) = <E [e_ij + e(w_ij)], e_kl + e(w_kl)>
        // For degree d FEM, G_ijkl is a degree 2 * (d - 1) polynomial on each
        // boundary element and is stored as a rank 4 tensor interpolant per
        // boundary element.
        constexpr size_t GDeg = 2 * (Deg - 1);
        typedef Interpolant<ETensor, K - 1, GDeg> G_t;
        std::vector<G_t> gradient(mesh.numBoundaryElements());
        typename _Sim::Strain  we_ij, we_kl;
        // Compute volume quantity
        Interpolant<ETensor, K, GDeg> G_elem;
        for (size_t elemIdx = 0; elemIdx < mesh.numElements(); ++elemIdx) { 
            auto e = mesh.element(elemIdx);
            if (!e.isBoundary()) continue;
            for (size_t ij = 0; ij < numStrains; ++ij) {
                sim.elementStrain(elemIdx, w[ij], we_ij);
                we_ij += SMatrix::CanonicalBasis(ij);
                for (size_t kl = ij; kl < numStrains; ++kl) {
                    sim.elementStrain(elemIdx, w[kl], we_kl);
                    we_kl += SMatrix::CanonicalBasis(kl);
                    auto G_ijkl = Interpolation<K, GDeg>::interpolant(
                        [&] (const VectorND<_Sim::numElemVertices> &p) {
                            return e->E().doubleContract(we_ij(p))
                                         .doubleContract(we_kl(p));
                        });
                    // Copy single entry interpolant over into interpolated rank
                    // 4 tensor's entries.
                    for (size_t n = 0; n < Simplex::numNodes(K, GDeg); ++n)
                        G_elem[n].D(ij, kl) = G_ijkl[n];
                }
            }

            // Distribute G_elem to all of this element's boundary faces/edges
            for (size_t fi = 0; fi < e.numNeighbors(); ++fi) {
                auto f = mesh.boundaryElement(e.interface(fi).boundaryEntity().index());
                if (!f) continue;
                auto &beGrad = gradient.at(f.index());
                // gradient is zero on the periodic boundary.
                if (f->isPeriodic)
                    beGrad *= 0;
                else {
                    if (GDeg == 0) beGrad[0] = G_elem[0];
                    else {
                        // Pick out nodal values from volume interpolant.
                        // TODO: optimize this to use traversal operations instead
                        // of a brute force search.
                        for (size_t bnc = 0; bnc < Simplex::numNodes(K - 1, GDeg); ++bnc) {
                            assert(bnc < f.numNodes());
                            size_t vni = f.node(bnc).volumeNode().index();
                            bool set = false;
                            for (size_t nc = 0; nc < e.numNodes(); ++nc) {
                                if (size_t(e.node(nc).index()) == vni) {
                                    beGrad[bnc] = G_elem[nc];
                                    set = true;
                                    break;
                                }
                            }
                            assert(set);
                        }
                    }
                }
            }
        }

        return gradient;
    }
}

#endif /* end of include guard: PERIODICHOMOGENIZATION_HH */
