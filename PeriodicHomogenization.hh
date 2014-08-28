#ifndef PERIODICHOMOGENIZATION_HH
#define PERIODICHOMOGENIZATION_HH

#include "LinearElasticity.hh"
#include <vector>
#include <string>

namespace PeriodicHomogenization {
    template<class _Simulator, class _FieldWriter>
    void solveCellProblems(std::vector<typename _Simulator::VField> &w_ij,
                           _Simulator &sim, _FieldWriter *writer = NULL)
    {
        typedef typename _Simulator::VField  VField;
        typedef typename _Simulator::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();

        sim.applyPeriodicConditions();
        sim.applyNoRigidMotionConstraint();

        w_ij.reserve(numStrains), w_ij.clear();
        for (size_t i = 0; i < numStrains; ++i) {
            VField rhs(sim.constantStrainLoad(-SMatrix::CanonicalBasis(i)));
            if (writer) {
                writer->addField(std::string("rhs ") + std::to_string(i),
                        sim.extractNodalField(rhs), _FieldWriter::PER_NODE);
            }
            w_ij.push_back(sim.solve(rhs));
        }
    }

    template<class _Simulator>
    typename _Simulator::ETensor homogenizedElasticityTensor(
            const std::vector<typename _Simulator::VField> &w_ij,
            const _Simulator &sim)
    {
        const auto &mesh = sim.mesh();
        typedef typename _Simulator::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        assert(w_ij.size() == numStrains);

        // Compute homogenized elasticity tensor (stress-like version):
        // Eh_ijkl = 1/|Y| int_w [E : strain(w_ij)]_kl + E_ijkl dV
        // Where |Y| = Yvol = periodic cell (grid bounding box) volume
        //        w  = periodic base cell geometry
        typename _Simulator::ETensor Eh;
        for (size_t ei = 0; ei < mesh.numElements(); ++ei) {
            typename _Simulator::ETensor Econtrib;
            for (size_t i = 0; i < w_ij.size(); ++i)
                sim.elementStress(ei, w_ij[i], Econtrib.DRowAsSymMatrix(i));
            Econtrib += mesh.element(ei)->E();
            Econtrib *= mesh.element(ei)->volume();
            Eh += Econtrib;
        }
        Eh /= mesh.boundingBox().volume();

        // // The following "energy-like" version is equivalent to the more efficient
        // // "stress-like" version above:
        // // Eh_ijkl = 1/|Y| int_w <E (e(w_ij) + e_ij), e(w_kl) + e_kl> dV,
        // typename _Simulator::ETensor EhE;
        // SMatrix we_ij, we_kl;
        // for (size_t ei = 0; ei < mesh.numElements(); ++ei) { 
        //     auto e = mesh.element(ei);
        //     for (size_t ij = 0; ij < numStrains; ++ij) {
        //         sim.elementStrain(ei, w_ij[ij], we_ij);
        //         we_ij += SMatrix::CanonicalBasis(ij);
        //         for (size_t kl = ij; kl < numStrains; ++kl) {
        //             sim.elementStrain(ei, w_ij[kl], we_kl);
        //             we_kl += SMatrix::CanonicalBasis(kl);
        //             EhE.D(ij, kl) += e->volume() * (e->E().
        //                     doubleContract(we_ij).doubleContract(we_kl));
        //         }
        //     }
        // }
        // EhE /= mesh.boundingBox().volume();

        return Eh;
    }

    template<class _Simulator>
    typename _Simulator::SField homogenizedElasticityTensorShapeDerivative(
            const typename _Simulator::ETensor &target,
            const std::vector<typename _Simulator::VField> &w_ij,
            const _Simulator &sim)
    {
        typedef typename _Simulator::ETensor ETensor;
        typedef typename _Simulator::SField  SField;
        typedef typename _Simulator::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        assert(w_ij.size() == numStrains);

        const auto &mesh = sim.mesh();
        ETensor diff = target - homogenizedElasticityTensor(w_ij, sim);
        // Shape derivative evaluated on normal velocity v_n:
        // diff_ijkl int_dt -<E [e_ij + e(w_ij)], e_kl + e(w_kl)> v_n dA
        // So the steepest descent is to evolve with
        //      v_n = diff_ijkl <E [e_ij + e(w_ij)], e_kl + e(w_kl)>
        //         := diff_ijkl DS_ijkl where
        //      DS_ijkl(y) = <E [e_ij + e(w_ij)], e_kl + e(w_kl)>
        // DS is constant on each element
        SField descentVelocity(mesh.numBoundaryElements());
        ETensor DS;
        SMatrix we_ij, we_kl;
        for (size_t ei = 0; ei < mesh.numElements(); ++ei) { 
            auto e = mesh.element(ei);
            if (!e.isBoundary()) continue;
            for (size_t ij = 0; ij < numStrains; ++ij) {
                sim.elementStrain(ei, w_ij[ij], we_ij);
                we_ij += SMatrix::CanonicalBasis(ij);
                for (size_t kl = ij; kl < numStrains; ++kl) {
                    sim.elementStrain(ei, w_ij[kl], we_kl);
                    we_kl += SMatrix::CanonicalBasis(kl);
                    DS.D(ij, kl) = e->E().doubleContract(we_ij)
                                         .doubleContract(we_kl);
                }
            }
            Real vn = diff.quadrupleContract(DS);

            // distribute vn to all of this element's boundary faces/edges
            for (size_t f = 0; f < mesh.element(ei).numNeighbors(); ++f) {
                auto h = mesh.element(ei).interface(f);
                if (h.isBoundary()) {
                    auto bh = h.boundaryEntity();
                    assert(bh);
                    descentVelocity[bh.index()] = vn;
                }
            }
        }

        return descentVelocity;
    }
}

#endif /* end of include guard: PERIODICHOMOGENIZATION_HH */
