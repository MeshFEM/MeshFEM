#ifndef PERIODICHOMOGENIZATION_HH
#define PERIODICHOMOGENIZATION_HH

#include "LinearElasticity.hh"
#include "MSHFieldWriter.hh"
#include <vector>
#include <string>

namespace PeriodicHomogenization3D {
    using namespace LinearElasticity3D;

    template<class _Simulator>
    void solveCellProblems(std::vector<VField> &w_ij, _Simulator &sim,
            MSHFieldWriter *mshWriter = NULL) {
        sim.applyPeriodicConditions();
        sim.applyNoRigidMotionConstraint();

        w_ij.reserve(6), w_ij.clear();
        for (size_t i = 0; i < 6; ++i) {
            VField rhs(sim.constantStrainLoad(-SMatrix::CanonicalBasis(i)));
            if (mshWriter) {
                mshWriter->addField(std::string("rhs ") + std::to_string(i),
                        sim.extractNodalField(rhs), MSHFieldWriter::PER_NODE);
            }
            w_ij.push_back(sim.solve(rhs));
        }
    }

    template<class _Simulator>
    ETensor homogenizedElasticityTensor(const std::vector<VField> &w_ij,
                                        const _Simulator &sim) {
        const auto &mesh = sim.mesh();

        // Compute homogenized elasticity tensor (stress-like version):
        // Eh_ijkl = 1/|Y| int_w [E : strain(w_ij)]_kl + E_ijkl dV
        // Where |Y| = Yvol = periodic cell (grid bounding box) volume
        //        w  = periodic base cell geometry
        ETensor Eh;
        for (size_t ei = 0; ei < mesh.numElements(); ++ei) {
            ETensor Econtrib;
            for (size_t i = 0; i < 6; ++i)
                sim.elementStress(ei, w_ij[i], Econtrib.DRowAsSymMatrix(i));
            Econtrib += mesh.element(ei)->elasticityTensor();
            Econtrib *= mesh.element(ei)->volume();
            Eh += Econtrib;
        }
        Eh /= mesh.boundingBox().volume();

        // // The following "energy-like" version is equivalent to the more efficient
        // // "stress-like" version above:
        // // // Eh_ijkl = 1/|Y| int_w <E (e(w_ij) + e_ij), e(w_kl) + e_kl> dV,
        // // // Where the integrand can be written as:
        // // //  <E e(w_ij), e(w_kl)> + [stress(w_ij)]_kl + [stress(w_kl)]_ij +
        // // //      rho * E_ijkl
        // // ETensor EhE(rho * m_E);
        // // CornerVec cornerIndices;
        // // PerElementOrthotropicStiffnessIntegrand Ke(m_E, m_model);
        // // typedef ElasticityTensor<Real, 3>  ETensor;
        // // for (size_t e = 0; e < m_elementGrid.numElements(); ++e) {
        // //     _BBox b = m_elementGrid.elementBoundingBox(e);
        // //     m_elementGrid.elementCorners(e, cornerIndices);
        // //     bool exact = m_exactFullElements && m_elementGrid.elementIsFull(e);
        // //     Ke.configure(b.dimensions(), exact);
        // //     if (!exact) m_quadrature.integrate(Ke, b);
        // //     Real vol = m_elementData[e].volume();

        // //     for (size_t ij = 0; ij < 6; ++ij) {
        // //         for (size_t kl = ij; kl < 6; ++kl) {
        // //             CornerVField we_ij, we_kl;
        // //             m_extractCornerVField(w_ij[ij], cornerIndices, we_ij);
        // //             m_extractCornerVField(w_ij[kl], cornerIndices, we_kl);
        // //             Real elemContrib = Ke.bilinearForm(we_ij, we_kl);

        // //             FlattenedRank2Tensor stress;
        // //             m_elementData[e].displacementToStress(w_ij[ij], cornerIndices,
        // //                     m_E, stress);
        // //             elemContrib += stress[kl] * vol;
        // //             m_elementData[e].displacementToStress(w_ij[kl], cornerIndices,
        // //                     m_E, stress);
        // //             elemContrib += stress[ij] * vol;

        // //             EhE.D(ij, kl) += elemContrib / Yvol;
        // //         }
        // //     }
        // // }
        // if (timer) timer->stop("Compute Homogenized Tensor");

        return Eh;
    }

}

#endif /* end of include guard: PERIODICHOMOGENIZATION_HH */
