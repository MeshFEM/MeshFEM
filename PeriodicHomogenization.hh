#ifndef PERIODICHOMOGENIZATION_HH
#define PERIODICHOMOGENIZATION_HH

#include <vector>
#include <string>

#include "GaussQuadrature.hh"
#include "InterpolantRestriction.hh"
// #include "MSHFieldWriter.hh"

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
            const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim,
            Real baseCellVolume) {
        const auto &mesh = sim.mesh();
        typedef typename _Sim::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        assert(w_ij.size() == numStrains);

        // Compute homogenized elasticity tensor (stress-like version):
        // Eh_ijkl = 1/|Y| int_w [E : strain(w_ij)]_kl + E_ijkl dV
        // Where |Y| = periodic cell (grid bounding box) volume
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
        Eh /= baseCellVolume;
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
        // EhE /= baseCellVolume;

        // return EhE;
    }

    // Assuming that the base elasticity tensor is constant over the entire base
    // cell, we can rewrite the homogenized elasticity tensor stress integral
    // formula in terms of displacements (using Green's theorem):
    // Eh_ijkl = 1/|Y| int_w [E : strain(w_ij)]_kl + E_ijkl dy
    //         = 1/|Y| int_dw E_ijpq frac{1}{2} (w^{kl}_p n_q + w^{kl}_q n_p) dA(y) + E * volFrac
    //         = 1/|Y| E_ijpq nw_pq + E * volFrac
    // Where   |Y|  = periodic cell (grid bounding box) volume
    //          w   = periodic base cell geometry
    //        nw_pq = 0.5 * int_dw [w^{kl}]_p n_q + [w^{kl}]_q n_p dA(y)
    template<class _Sim>
    typename _Sim::ETensor homogenizedElasticityTensorDisplacementForm(
            const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim,
            Real baseCellVolume) {
        const auto &mesh = sim.mesh();
        typedef typename _Sim::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        assert(w_ij.size() == numStrains);

        // Elasticity tensor must be constant over the entire base cell
        const typename _Sim::ETensor &EBase = mesh.element(0)->E();

        typename _Sim::ETensor Eh;
        SMatrix nw_pq;

        // Displacement restricted to a boundary element
        Interpolant<VectorND<_Sim::N>, _Sim::K - 1, _Sim::Degree> w_be;
        for (size_t bei = 0; bei < mesh.numBoundaryElements(); ++bei) {
            auto be = mesh.boundaryElement(bei);
            typename _Sim::ETensor Econtrib;
            const auto &n = be->normal();
            for (size_t i = 0; i < w_ij.size(); ++i) {
                const auto &w = w_ij[i];
                // Copy the boundary node values into interpolant
                for (size_t ni = 0; ni < w_be.size(); ++ni)
                    w_be[ni] = w(be.node(ni).volumeNode().index());
                auto w_be_int = w_be.integrate(be->volume());

                for (size_t p = 0; p < _Sim::N; ++p)
                    for (size_t q = p; q < _Sim::N; ++q)
                        nw_pq(p, q) = 0.5 * (w_be_int[p] * n[q] + w_be_int[q] * n[p]);
                Eh.DRowAsSymMatrix(i) += EBase.doubleContract(nw_pq);
            }
        }

        Eh += EBase * mesh.volume();
        Eh /= baseCellVolume;

        return Eh;
    }

    // Assumes the base cell is the axis-aligned mesh bounding box
    // (not true, e.g., for rotated base cells).
    template<class _Sim>
    typename _Sim::ETensor homogenizedElasticityTensor(
            const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim) {
        return homogenizedElasticityTensor(w_ij, sim, sim.mesh().boundingBox().volume());
    }

    // Displacement form...
    // Assumes the base cell is the axis-aligned mesh bounding box
    // (not true, e.g., for rotated base cells).
    template<class _Sim>
    typename _Sim::ETensor homogenizedElasticityTensorDisplacementForm(
            const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim) {
        return homogenizedElasticityTensorDisplacementForm(w_ij, sim, sim.mesh().boundingBox().volume());
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Compute the macroscopic-strain-to-microscopic strain tensors.
    //  These are rank 4 tensors that have minor but not major symmetries:
    //      G_ijkl(x) = [e(w^kl)(x) + e^kl]_ij
    //  @return     A vector of per-element tensors. The tensor at index i is
    //              the average of G_ijkl(x) over element i.
    *///////////////////////////////////////////////////////////////////////////
    template<class _Sim>
    std::vector<ElasticityTensor<Real, _Sim::N, false>>
    macroStrainToMicroStrainTensors(const std::vector<typename _Sim::VField> &w, const _Sim &sim) {
        size_t numElems = sim.mesh().numElements();
        std::vector<ElasticityTensor<Real, _Sim::N, false>> G(numElems);
        typename _Sim::Strain  strain_ij;
        for (size_t e = 0; e < numElems; ++e) {
            for (size_t ij = 0; ij < w.size(); ++ij) {
                sim.elementStrain(e, w[ij], strain_ij);
                G[e].DColAsSymMatrix(ij) = strain_ij.average();
                G[e].DColAsSymMatrix(ij) += _Sim::SMatrix::CanonicalBasis(ij);
            }
        }
        return G;
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
        Real bboxVol = mesh.boundingBox().volume();

        // Shape derivative evaluated on normal velocity v_n:
        // DS(E_H)[v_n n] = 1/|Y| int_dt <E [e_ij + e(w_ij)], e_kl + e(w_kl)> v_n dA
        // So the steepest ascent direction is to evolve with
        //      v_n(x) = 1/|Y| <E [e_ij + e(w_ij)], e_kl + e(w_kl)> := G_ijkl(x)
        // for each non-periodic boundary point x.
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
                    G_ijkl /= bboxVol;
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
                // Zero gradient on the periodic boundary. ETensor default
                // constructor zero-inits, so beGrad should currently be zero.
                if (f->isPeriodic) continue;
                restrictInterpolant(e, f, G_elem, beGrad);
            }
        }

        return gradient;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Shape derivative of fluctuation displacements evaluated on a particular
    // velocity field. This is the "direct" approach not using the adjoint
    // method:
    // Solve cell problems with load
    //      - int_bdry (v dot n) (strain(phi) : C : [strain(w^kl) + e^kl]) dA
    ////////////////////////////////////////////////////////////////////////////
    template<class _Sim, class NormalShapeVelocity>
    void fluctuationDisplacementShapeDerivatives(const _Sim &sim,
            const std::vector<typename _Sim::VField> &w,
            const NormalShapeVelocity &vn,
            std::vector<typename _Sim::VField> &dot_w) {
        BENCHMARK_START_TIMER("Fluctuation Shape Derivatives");

        constexpr size_t Deg = _Sim::Degree;
        constexpr size_t   K = _Sim::K;
        typename _Sim::Strain  strain_kl;

        const auto &mesh = sim.mesh();

        using SMatrix = typename _Sim::SMatrix;
        std::vector<Interpolant<SMatrix, K - 1, Deg - 1>> bdry_stresses;
        bdry_stresses.resize(mesh.numBoundaryElements());

        // static size_t it = 0; 
        // MSHFieldWriter writer("debug_fd_sd_" + std::to_string(it) + ".msh", sim.mesh());
        // ++it;

        dot_w.clear(), dot_w.reserve(w.size());
        for (size_t kl = 0; kl < w.size(); ++kl) {
            for (auto e : mesh.elements()) {
                if (!e.isBoundary()) continue;
                const auto &C = e->E();
                sim.elementStrain(e.index(), w[kl], strain_kl);
                strain_kl += SMatrix::CanonicalBasis(kl);

                for (size_t fi = 0; fi < e.numNeighbors(); ++fi) {
                    auto f = mesh.boundaryElement(e.interface(fi).boundaryEntity().index());
                    if (!f) continue;
                    auto &bdry_stress_kl = bdry_stresses.at(f.index());
                    if (f->isPeriodic) bdry_stress_kl = 0;
                    else               restrictInterpolant(e, f, strain_kl, bdry_stress_kl);
                    for (size_t n = 0; n < bdry_stress_kl.size(); ++n)
                        bdry_stress_kl[n] = C.doubleContract(bdry_stress_kl[n]);
                }
            }

            auto loadChange = sim.changeInDivTensorLoad(vn, bdry_stresses, true);
            dot_w.push_back(sim.solve(loadChange));

            // typename _Sim::VField outField;
            // // Subtract off average displacements so that fields are comparable
            // // across meshes.
            // outField = w[kl];
            // outField -= outField.mean();
            // writer.addField("w " + std::to_string(kl), outField);
            // outField = dot_w[kl];
            // outField -= outField.mean();
            // writer.addField("dot w " + std::to_string(kl), outField);
        }

        BENCHMARK_STOP_TIMER("Fluctuation Shape Derivatives");
    }
}

#endif /* end of include guard: PERIODICHOMOGENIZATION_HH */
