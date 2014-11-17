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

    ////////////////////////////////////////////////////////////////////////////
    /*! Computes the steepest ascent direction (i.e. the theta maximizing the
    //  shape derivative DS[theta]) of each component of the homogenized
    //  elasticity tensor. This is a per-boundary-element rank 4 tensor field.
    //  @param[in]  w       fluctuation displacements (cell problem solutions)
    //  @param[in]  sim     linear elasticity solver
    //  @return     per-boundary-element rank 4 tensor field.
    *///////////////////////////////////////////////////////////////////////////
    template<class _Sim>
    std::vector<typename _Sim::ETensor> homogenizedTensorGradient(
            const std::vector<typename _Sim::VField> &w, const _Sim &sim) {
        typedef typename _Sim::ETensor ETensor;
        typedef typename _Sim::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        assert(w.size() == numStrains);

        const auto &mesh = sim.mesh();
        // Shape derivative evaluated on normal velocity v_n:
        // DS(E_H)[v_n n] = int_dt <E [e_ij + e(w_ij)], e_kl + e(w_kl)> v_n dA
        // So the steepest ascent direction is to evolve with
        //      v_n(x) = <E [e_ij + e(w_ij)], e_kl + e(w_kl)> := G_ijkl(x)
        // for each non-periodic boundary point x.
        //      DS_ijkl(y) = <E [e_ij + e(w_ij)], e_kl + e(w_kl)>
        // For linear FEM, G_ijkl is constant on each element, so is stored as a
        // tensor per boundary edge.
        // NOTE: for higher order FEM, we will probably have to settle for a
        // function that computes an inner product with G instead of returning
        // a representation of G itself (unless what we are taking an inner
        // product with is constant, in which case we can return average of G
        // over the boundary element, which is actually probably the case.)
        std::vector<ETensor> gradient(mesh.numBoundaryElements());
        SMatrix we_ij, we_kl;
        for (size_t elemIdx = 0; elemIdx < mesh.numElements(); ++elemIdx) { 
            auto e = mesh.element(elemIdx);
            if (!e.isBoundary()) continue;
            ETensor G_elem;
            for (size_t ij = 0; ij < numStrains; ++ij) {
                sim.elementStrain(elemIdx, w[ij], we_ij);
                we_ij += SMatrix::CanonicalBasis(ij);
                for (size_t kl = ij; kl < numStrains; ++kl) {
                    sim.elementStrain(elemIdx, w[kl], we_kl);
                    we_kl += SMatrix::CanonicalBasis(kl);
                    G_elem.D(ij, kl) = e->E().doubleContract(we_ij)
                                             .doubleContract(we_kl);
                }
            }

            // Distribute G_elem to all of this element's boundary faces/edges
            for (size_t f = 0; f < mesh.element(elemIdx).numNeighbors(); ++f) {
                auto h = mesh.element(elemIdx).interface(f).boundaryEntity();
                if (h && !h->isPeriodic)
                    gradient.at(h.index()) = G_elem;
            }
        }

        return gradient;
    }
}

#endif /* end of include guard: PERIODICHOMOGENIZATION_HH */
