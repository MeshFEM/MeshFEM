#include "MeshFEM/GlobalBenchmark.hh"

#define NORMAL_INFERENCE_PROBLEM_VERBOSITY 1

template <class Psi_2x2>
void ElasticSheet<Psi_2x2>::setIdentityDeformation() {
    const auto &m = mesh();

    // Set the deformed positions to the rest positions.
    m_deformedPositions.resize(m_numVertices, 3);
    for (const auto v : m.vertices())
        m_deformedPositions.row(v.index()) = v.node()->p.transpose();
    m_updateElementEmbedding();

    initializeMidedgeNormals(/* inferCreaseAngles = */ true, /* minimizeBending = */ true);

    // Dispatch deformation update notifications.
    // We don't use our derived `m_defoConfigUpdated` implementation because
    // all necessary internal state has already been updated.
    Base::m_defoConfigUpdated();
}

// Quadratic minimization to infer midedge normals (thetas):
// minimize the squared Frobenius norm of the second fundamental form.
// (This objective is proportional to the bending energy stored in an isotropic
// plate with Young's modulus 1 and Poisson's ratio 0.)
// For convenience we use our Newton solver even though it should always
// converge in a single iteration.
//
// In order to make the normals/curvature computed independent of the reference
// configuration, we pose the inference energy on the deformed mesh (pushing
// the second fundamental form forward to the deformed configuration). We
// calculate this pushed-forward fundamental form directly and then verify its
// pullback agrees with II.
#include <MeshFEM/newton_optimizer/NewtonProblem.hh>
#include <MeshFEM/SystemAssembler.hh>
template<class ESheet>
struct NormalInferenceProblem : public NewtonProblem {
    using Assembler = ScalarSystemAssembler;
    Assembler m_assembler;

    using M3d = typename ESheet::M3d;
    NormalInferenceProblem(ESheet &sheet) : m_sheet(sheet), m_assembler(sheet.numThetas()) {
        // Build the theta-theta block of the full Elastic sheet Hessian sparsity pattern...
        m_hessianSparsity = m_assembler.sparsityPattern(sheet.mesh().numElements(),
                                                        [&](size_t ei) { return varsForElement(ei); });

        m_updateDeformedII();
    }

    std::array<size_t, 3> varsForElement(size_t ei) const {
        return { m_sheet.edgeForHalfEdge(3 * ei + 0),
                 m_sheet.edgeForHalfEdge(3 * ei + 1),
                 m_sheet.edgeForHalfEdge(3 * ei + 2) };
    }

    virtual size_t numVars() const override { return m_sheet.numThetas(); }
    virtual void setVars(const VXd &vars) override { m_sheet.setThetas(vars.cast<typename ESheet::Real>()); m_updateDeformedII(); }
    virtual VXd getVars() const override { return m_sheet.getThetas().template cast<double>(); }

    virtual Real objective() const override {
        Real result = 0.0;
        const size_t ne = m_deformedII.size();
        for (size_t ei = 0; ei < ne; ++ei)
            result += 0.5 * m_sheet.deformedArea(ei) * m_deformedII[ei].squaredNorm();
        return result;
    }

    virtual VXd gradient(bool /* freshIterate */ = false) const override {
        VXd g(VXd::Zero(numVars()));
        for (const auto e : m_sheet.mesh().elements()) {
            const size_t ei = e.index();
            const auto &II = m_deformedII[ei];
            const auto &dtg = m_sheet.deformedTriGeometry(ei);
            const Real A = dtg.A;
            const Real dE_dpsi = A;
            for (const auto he : e.halfEdges()) {
                const size_t edgeIdx = m_sheet.edgeForHalfEdge(he.index());
                const size_t lhi = he.localIndex();
                const Real sign = he.isPrimary() ? 1.0 : -1.0;
                const Real len = dtg.edgeLens[lhi];

                const auto glambda = (dtg.unitEdgePerpendiculars.col(lhi) / dtg.h[lhi]).eval();
                const Real dE_d_A_gamma_div_len = (4 * dE_dpsi) * (II * glambda).dot(glambda); // Derivative of the energy with respect to the coefficient of `glambda \otimes glambda` in the shape operator.
                // The derivative with respect to the theta variables is simple
                g[edgeIdx] += ((sign * (dtg.A / len))) * dE_d_A_gamma_div_len;
            }
        }

        return g;
    }

protected:
    virtual NewtonHessian m_getHessianSparsityPattern() const override { return m_hessianSparsity; }
    virtual bool m_updateSparsityPattern() const override { return false; }

    virtual void m_evalHessian(NewtonHessian &result, bool /* projectionMask */) const override {
        result.setZero();

        m_assembler.assembleHessian(result,
            m_sheet.mesh().numElements(),
            [&](size_t ei) -> M3d {
                M3d H_e;
                auto e = m_sheet.mesh().element(ei);
                const auto &dtg = m_sheet.deformedTriGeometry(ei);
                const Real A = dtg.A;
                const Real dE_dpsi = A;
                for (const auto he : e.halfEdges()) {
                    const size_t lhi = he.localIndex();
                    const Real sign = he.isPrimary() ? 1.0 : -1.0;
                    const Real len = dtg.edgeLens[lhi];

                    const auto &glambda = (dtg.unitEdgePerpendiculars.col(lhi) / dtg.h[lhi]).eval();
                    for (const auto he_b : e.halfEdges()) {
                        const size_t lhi_b = he_b.localIndex();
                        if (lhi_b < lhi) continue; // Only fill in upper triangle of H_e
                        const Real len_b = dtg.edgeLens[lhi_b];

                        const auto &glambda_b = (dtg.unitEdgePerpendiculars.col(lhi_b) / dtg.h[lhi_b]).eval();

                        const Real sign_b = he_b.isPrimary() ? 1.0 : -1.0;
                        const Real d2E_d2_A_gamma_div_len_ab = 4 * (4 * dE_dpsi) * std::pow(glambda.dot(glambda_b), 2);

                        // Shape operator/gamma are linear in theta, so (delta_b d_A_gamma_div_len_d_xa) term vanishes.
                        const Real delta_b_dE_d_A_gamma_div_len = ((sign_b * (A / len_b))) * d2E_d2_A_gamma_div_len_ab;

                        H_e(lhi, lhi_b) = (sign * (A / len)) * delta_b_dE_d_A_gamma_div_len;
                    }
                }
                return H_e;
            },
            [&](size_t ei) { return varsForElement(ei); }
        );
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        result.setIdentity(true);
    }

    void m_updateDeformedII() {
        const auto &m = m_sheet.mesh();
        m_deformedII.resize(m.numElements());
        const auto &gammas = m_sheet.getGammas();

        for (const auto e : m.elements()) {
            const size_t ei = e.index();
            auto &II_d = m_deformedII[ei];
            II_d.setZero();
            const auto &dtg = m_sheet.deformedTriGeometry(ei);
            for (const auto he : e.halfEdges()) {
                auto glambda = (dtg.unitEdgePerpendiculars.col(he.localIndex()) / dtg.h[he.localIndex()]).eval();
                Real len = m_sheet.deformedEdgeVector(he).norm();
                II_d += ((4 * gammas[he.index()] * (dtg.A / len)) * glambda) * glambda.transpose();
            }

            M3d F = (e->gradBarycentric() * m_sheet.getCornerPositions(ei)).transpose();
            M3d II = m_sheet.getII_3D(ei);
            if ((II - F.transpose() * II_d * F).squaredNorm() / II.squaredNorm() > 1e-18)
                throw std::runtime_error("Second fundamental form pushforward mismatch.");
        }
    }

    ESheet &m_sheet;
    mutable NewtonHessian m_hessianSparsity;
    std::vector<M3d> m_deformedII;
};

template <class Psi_2x2>
void ElasticSheet<Psi_2x2>::initializeMidedgeNormals(bool inferCreaseAngles, bool minimizeBending) {
    const auto &m = mesh();

    if (inferCreaseAngles) {
        m_creaseAngles.resize(numCreases());
        // Initialize the crease angle variables as the (signed) dihedral angles of
        // the rest mesh. This is usually want we want and is necessary for a
        // piecewise flat sheet to be initialized with flat rest triangles (i.e., m_restII = 0).
        for (size_t i = 0; i < numCreases(); ++i) {
            auto he = m.halfEdge(halfEdgeForCreaseAngle(i));
            m_creaseAngles[i] = atan2(he.tri()->normal().cross(he.opposite().tri()->normal()).dot((he.tip().node()->p - he.tail().node()->p).normalized()),
                                      he.tri()->normal()  .dot(he.opposite().tri()->normal()));
        }
    }

    // Initialize the reference frames.
    // We pick the averaged edge normals as the initial d1 frame vector and midedge normal.
    m_referenceFrame.resize(numEdges());
    m.visitEdges([this](CHEHandle he, size_t edgeIndex) {
        V3d t  = (deformedEdgeVector(he)).normalized().transpose();
        V3d d1 = deformedTriNormal(he.tri().index());
        if (!he.isBoundary()) d1 += deformedTriNormal(he.opposite().tri().index());
        d1 = d1.normalized();

        if (std::abs(t.dot(d1)) > 1e-14) {
            std::cout << "Perpendicularity error for edge " << edgeIndex << std::endl;
            std::cout << "tri n = " << deformedTriNormal(he.tri().index()).transpose() << std::endl;
            if (!he.isBoundary()) std::cout << "opp n = " << deformedTriNormal(he.opposite().tri().index()).transpose() << std::endl;
            std::cout << "averaged d1 = " << d1.transpose() << std::endl;
            std::cout << "t = " << t.transpose() << std::endl;
            throw std::logic_error("Non-perpendicular averaged edge normal: " + std::to_string(t.dot(d1)));
        }

        m_referenceFrame[edgeIndex] << t, d1, t.cross(d1); // Generate the third vector of the right-handed frame.
    });

    // Measure the angle around the edge tangent from reference director d1 to the triangle normal.
    // (ccw with tip pointing toward us)
    m_alphas.resize(m.numHalfEdges());
    for (const auto he : m.halfEdges()) {
        const auto &frame = m_referenceFrame[edgeForHalfEdge(he.index())];

        const auto &n = deformedTriNormal(he.tri().index());
        m_alphas[he.index()] = angle<Real>(/* axis */ frame.col(0), frame.col(1), n);
        if (std::abs(m_alphas[he.index()]) > M_PI / 2) { // Shouldn't happen except for sharp creases
            std::cout << "WARNING: Large alpha: " << m_alphas[he.index()] << std::endl;
            std::cout << frame << std::endl;
            std::cout << "Tri normal: " << n.transpose() << std::endl;

            V3d n_avg = n;
            if (he.opposite().tri())
                n_avg += deformedTriNormal(he.opposite().tri().index());
            n_avg = n_avg.normalized();
            std::cout << "Averaged edge normal: " << n_avg.transpose() << std::endl << std::endl;
            std::cout << "For he, edge: " << he.index() << ", " << edgeForHalfEdge(he.index()) << std::endl;
        }
    }

    // Apply the current frame/alphas as the source values.
    updateSourceFrame();

    // Initialize with midedge normals coinciding with the averaged edge normals.
    // Side effect: updates the cached shape operator and midedge normals.
    m_thetas.setZero(numThetas());
    m_updateDeformedElements(/* positionsUpdated = */ false);

    // Finally, infer the "best" midedge normals by minimizing the bending energy with respect to theta.
    if (minimizeBending) {
        if (!m_normalInferenceOptimizer) {
            auto problem = std::make_unique<NormalInferenceProblem<ElasticSheet>>(*this);
            m_normalInferenceOptimizer = std::make_unique<NewtonOptimizer>(std::move(problem));
            m_normalInferenceOptimizer->options.factorizer = get_default_cholesky_provider();
            m_normalInferenceOptimizer->options.verbose = NORMAL_INFERENCE_PROBLEM_VERBOSITY;
        }
        m_normalInferenceOptimizer->optimize();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Elastic Energy
////////////////////////////////////////////////////////////////////////////////
template <class Psi_2x2>
typename ElasticSheet<Psi_2x2>::Real ElasticSheet<Psi_2x2>::energy(const EnergyType etype) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSheet.energy");
    return summation_parallel([this, etype](size_t ei) { return m_shellElements[ei].energy(etype); },
                              mesh().numElements());
}

////////////////////////////////////////////////////////////////////////////////
// Elastic Energy Gradient
////////////////////////////////////////////////////////////////////////////////
template <class Psi_2x2>
template <class Result>
void ElasticSheet<Psi_2x2>::accumulateGradGamma(Real weight, size_t ei, size_t lhi, bool updatedSource, Result &&result) const {
    typename PBE::CPosMap gradCornerPos(result.data()); // Corner positions in each row
    const auto &de = deformedTriGeometry(ei);

    if (!updatedSource) {
        // Parallel transport
        // Here we are differentiating with respect to the halfedge vector; in the
        // case of the non-primary halfedge, we effectively differentiate the negated
        // angle with respect to the negated edge vector, so the sign cancels out!
        const size_t edgeIdx = edgeForHalfEdge(mesh().element(ei).halfEdge(lhi).index());
        const auto &srcFrame = m_sourceReferenceFrame[edgeIdx];
        const auto &curFrame =       m_referenceFrame[edgeIdx];
        const auto &t  = curFrame.col(0), &ts  = srcFrame.col(0),
                   &d1 = curFrame.col(1),
                   &d2 = curFrame.col(2), &ds2 = srcFrame.col(2);

        const Real inv_chi_hat = 1.0 / (1.0 + ts.dot(t));
        // Derivative of `alpha` with respect to the unit edge tangent.
        V3d neg_dalpha_dt = (ds2.dot(t) * ts.cross(d2) + d1.dot(ts) * ds2) * inv_chi_hat
                          - (ds2.dot(t) * d1.dot(ts) * inv_chi_hat * inv_chi_hat) * ts
                          + d2.cross(ds2);
        // Note that alpha decreases (gamma increases) when d1 rotates ccw.
        // Derivative of energy with respect to the edge vector.
        // Incorporates the (1 / ||e_i||) (I - t_i t_i^T) term.
        V3d dcoeff_dedge = (weight / de.edgeLens[lhi]) * (neg_dalpha_dt - t.dot(neg_dalpha_dt) * t);
        gradCornerPos.row((lhi + 2) % 3) += dcoeff_dedge; // local tip
        gradCornerPos.row((lhi + 1) % 3) -= dcoeff_dedge; // local tail
    }

    // Gamma increases when normal rotates cw.
    gradCornerPos += (-weight / (de.edgeVecDotProducts(lhi, lhi) * de.h[lhi])) * de.edgeVecDotProducts.col(lhi) * de.normal.transpose();
}

template <class Psi_2x2>
void ElasticSheet<Psi_2x2>::accumulateGradient(Real weight, VXd &g, bool updatedSource, VariableMask vars, const EnergyType etype) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSheet.gradient");
    if (vars != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");
    const auto &m = mesh();
    auto accumulate_per_element_contrib = [this, updatedSource, etype, weight, &m](size_t ei, VXd &g_out) {
        auto g_e = m_shellElements[ei].gradient(weight, etype);

        // Chain rule accounting for gamma changes due to parallel transport and
        // rotating triangle normals.
        for (size_t lhi = 0; lhi < 3; ++lhi) {
            Real dE_dgamma_i = g_e[PBE::GammaOffset + lhi];
            accumulateGradGamma(dE_dgamma_i, ei, lhi, updatedSource, g_e);
        }

        const auto &e = m.element(ei);
        g_out.template segment<3>(3 * e.vertex(0).index()) += g_e.template segment<3>(0);
        g_out.template segment<3>(3 * e.vertex(1).index()) += g_e.template segment<3>(3);
        g_out.template segment<3>(3 * e.vertex(2).index()) += g_e.template segment<3>(6);

        const size_t to = thetaOffset();
        const size_t co = creaseAngleOffset();
        for (const auto he : e.halfEdges()) {
            const Real sign = he.isPrimary() ? 1.0 : -1.0; // ∂ gamma / ∂ theta
            g_out[to + edgeForHalfEdge(he.index())] += sign * g_e[9 + he.localIndex()];
            int ci = creaseForHalfEdge(he.index());
            // Crease chain rule: note that ∂ gamma / ∂ crease_angle = -0.5
            if (ci >= 0) g_out[co + ci] -= 0.5 * g_e[9 + he.localIndex()];
        }
    };

    assemble_parallel(accumulate_per_element_contrib, g, m.numElements());
}

////////////////////////////////////////////////////////////////////////////////
// Elastic Energy Hessian
////////////////////////////////////////////////////////////////////////////////
// Get the Hessian of per-element elastic energy with respect to the
// triangle-local x and normal rotation variables.
// Note that the "local normal rotation variables" are defined to be `gamma_i`,
// and differ in sign from the global theta variables for non-primary halfedges.
template <class Psi_2x2>
typename ElasticSheet<Psi_2x2>::PerElementHessian
ElasticSheet<Psi_2x2>::elementHessian(size_t ei, const EnergyType etype, bool projectionMask) const {
    const bool membraneProjection = projectionMask && (m_hessianProjectionType == HessianProjectionType::MembraneFBased);
    PerElementHessian H_elem = m_shellElements[ei].hessian(1.0, membraneProjection, etype);

    // Chain rule for bending energy term.
    if (!m_disableBending && ((etype == EnergyType::Bending) || (etype == EnergyType::Full))) {
        Eigen::Matrix<Real, 3, 9, Eigen::RowMajor> dGamma_dx;
        dGamma_dx.setZero();
#if 1
        for (size_t lhi = 0; lhi < 3; ++lhi)
            accumulateGradGamma(1.0, ei, lhi, /* updatedSource = */ true, dGamma_dx.row(lhi));
#else
        const auto &de = deformedTriGeometry(ei);
        for (size_t lhi = 0; lhi < 3; ++lhi) {
            // Gamma increases when normal rotates cw.
            typename PBE::CPosMap(dGamma_dx.row(lhi).data())
                          = (-1.0 / (de.edgeVecDotProducts(lhi, lhi) * de.h[lhi])) * de.edgeVecDotProducts.col(lhi) * de.normal.transpose();
        }
#endif

        // Chain rule accounting for gamma changes due to parallel transport and
        // normal rotation. Letting H denote the result of PBE::Hessian:
        // d2E/dxdɣ = H_xɣ + (dɣ/dx)^T H_ɣɣ
        // First, compute the (dɣ/dx)^T H_ɣɣ  term, but defer its addition until
        // after the d2E/dx2 block is updated (since that needs the original H_xɣ term).
        M3d H_gg = H_elem.template bottomRightCorner<3, 3>().template selfadjointView<Eigen::Upper>(); // Using the `selfadjointView` in the product below is slow...
        Eigen::Matrix<Real, 9, 3> xGammaBlockContrib = dGamma_dx.transpose() * H_gg; // This is reused below...

        // d2E/dx2 = H_xx + 2 sym(H_xɣ dɣ/dx) + (dɣ/dx)^T H_ɣɣ (dɣ/dx) + g . d2ɣ/dx2
        Eigen::Matrix<Real, 9, 9> xxBlockContrib = (H_elem.template topRightCorner<9, 3>() + 0.5 * xGammaBlockContrib).eval() * dGamma_dx;
        H_elem.template  topLeftCorner<9, 9>().template triangularView<Eigen::Upper>() += xxBlockContrib + xxBlockContrib.transpose();
        H_elem.template topRightCorner<9, 3>() += xGammaBlockContrib;

        // Gamma Hessian term
        auto g_gamma = m_shellElements[ei].plate.grad_gamma();
        const auto &de = deformedTriGeometry(ei);

        // Precompute the n ⨂  ehatp_k terms used repeatedly below.
        std::array<M3d, 3> n_otimes_ehatp;
        n_otimes_ehatp[0] = de.normal * de.unitEdgePerpendiculars.col(0).transpose();
        n_otimes_ehatp[1] = de.normal * de.unitEdgePerpendiculars.col(1).transpose();
        n_otimes_ehatp[2] = de.normal * de.unitEdgePerpendiculars.col(2).transpose();

        for (size_t lhi = 0; lhi < 3; ++lhi) {
            // Accumulate dE_dɣ d2gamma_i_dx2 directly to the Hessian
            Real d2E_dgamma_i = g_gamma[lhi];
            V3d ei_dot_edge = de.edgeVecDotProducts.row(lhi).transpose();

            // Note that the derivative of the parallel transport term is purely
            // skew symmetric and only acts to cancel the skew symmetric part of
            // the normal rotation term. Therefore we omit it and simply compute
            // the symmetric part of the normal rotation term.

            // Symmetrized n ⨂  ehatp_i term (the only asymmetric subterm)
            Real liSq = ei_dot_edge[lhi];
            {
                M3d contrib = (d2E_dgamma_i / liSq) * n_otimes_ehatp[lhi];
                M3d symmetrized_contrib = 0.5 * (contrib + contrib.transpose());

                const size_t tip  = (lhi + 2) % 3;
                const size_t tail = (lhi + 1) % 3;

                H_elem.template block<3, 3>(3 *  tip, 3 *  tip) +=  symmetrized_contrib;
                H_elem.template block<3, 3>(3 * tail, 3 * tail) +=  symmetrized_contrib;
                if (tip < tail) H_elem.template block<3, 3>(3 *  tip, 3 * tail) -= symmetrized_contrib;
                else            H_elem.template block<3, 3>(3 * tail, 3 *  tip) -= symmetrized_contrib;
            }

            Real coeff = d2E_dgamma_i / (liSq * de.h[lhi]);
            H_elem.template block<3, 3>(3 * 0, 3 * 1) += (coeff * ei_dot_edge[1] / de.h[0]) * n_otimes_ehatp[0]
                                                      +  (coeff * ei_dot_edge[0] / de.h[1]) * n_otimes_ehatp[1].transpose();
            H_elem.template block<3, 3>(3 * 0, 3 * 2) += (coeff * ei_dot_edge[2] / de.h[0]) * n_otimes_ehatp[0]
                                                      +  (coeff * ei_dot_edge[0] / de.h[2]) * n_otimes_ehatp[2].transpose();
            H_elem.template block<3, 3>(3 * 1, 3 * 2) += (coeff * ei_dot_edge[2] / de.h[1]) * n_otimes_ehatp[1]
                                                      +  (coeff * ei_dot_edge[1] / de.h[2]) * n_otimes_ehatp[2].transpose();

            H_elem.template block<3, 3>(3 * 0, 3 * 0).template triangularView<Eigen::Upper>() += (coeff * ei_dot_edge[0] / de.h[0]) * (n_otimes_ehatp[0] + n_otimes_ehatp[0].transpose());
            H_elem.template block<3, 3>(3 * 1, 3 * 1).template triangularView<Eigen::Upper>() += (coeff * ei_dot_edge[1] / de.h[1]) * (n_otimes_ehatp[1] + n_otimes_ehatp[1].transpose());
            H_elem.template block<3, 3>(3 * 2, 3 * 2).template triangularView<Eigen::Upper>() += (coeff * ei_dot_edge[2] / de.h[2]) * (n_otimes_ehatp[2] + n_otimes_ehatp[2].transpose());
        }
    }

    return H_elem;
}

template <class Psi_2x2>
struct ElasticSheet<Psi_2x2>::CustomHEAData {
    CustomHEAData(const ElasticSheet &es, Real weight, size_t ei, const EnergyType etype, bool projectionMask) {
        H_e = es.elementHessian(ei, etype, projectionMask);
        if (weight != 1.0) H_e *= weight;
        const auto &m = es.mesh();
        auto e = m.element(ei);
        for (auto v : e.vertices())
            evars[v.localIndex()] = v.index();
        size_t numCreases = 0;
        for (auto he : e.halfEdges()) {
            halfedgeIsPrimary[he.localIndex()] = he.isPrimary();
            evars[3 + he.localIndex()] = m.numVertices() + es.edgeForHalfEdge(he.index());
            int ci = es.creaseForHalfEdge(he.index());
            if (ci < 0) continue;
            localHalfedgeForLocalCrease[numCreases] = he.localIndex();
            evars[6 + numCreases++] = m.numVertices() + es.numEdges() + ci;
        }
        evars.numVars = 6 + numCreases;
    }

    MatMaxN_T<Real, 3> block(size_t a, size_t b, size_t /* bsa */, size_t /* bsb */) const { return block(a, b); }

    MatMaxN_T<Real, 3> block(size_t a, size_t b) const {
        // x-x block
        if (b < 9) return H_e.template block<3, 3>(a, b);

        // *-theta cols
        if (b < 12) {
            Real coeff = halfedgeIsPrimary[b - 9] ? 1.0 : -1.0;
            if (a < 9) return coeff * H_e.template block<3, 1>(a, b);
            MatMaxN_T<Real, 3> result(1, 1);
            result(0, 0) = (halfedgeIsPrimary[a - 9] ? coeff : -coeff) * H_e(a, b); 
            return result;
        }

        // *-crease_angle cols
        size_t localCrease_b = b - 12;
        b = 9 + localHalfedgeForLocalCrease[localCrease_b];
        Real coeff = -0.5; // dgamma / d crease_angle = -0.5

        if (a < 9) return coeff * H_e.template block<3, 1>(a, b);
        if (a >= 12) {
            size_t localCrease_a = a - 12;
            a = 9 + localHalfedgeForLocalCrease[localCrease_a];
            coeff *= -0.5; // dgamma / d crease_angle = -0.5

            if (a > b) {
                // The index rewriting above can reference the lower triangle of
                // the (theta-theta) block--redirect to the upper triangle.
                std::swap(a, b);
            }
        }
        else if (a >= 9) {
            coeff *= halfedgeIsPrimary[a - 9] ? 1.0 : -1.0;
        }

        MatMaxN_T<Real, 3> result(1, 1);
        result(0, 0) = coeff * H_e(a, b); 
        return result;
    }

    // Number of block variables in the typical case.
    static constexpr size_t TypicalNumVars() { return 6; }

    PerElementHessian H_e;
    EBlockVars evars;
    Eigen::Vector3i localHalfedgeForLocalCrease;
    std::array<bool, 3> halfedgeIsPrimary;
};

template <class Psi_2x2>
void ElasticSheet<Psi_2x2>::accumulateHessian(Real weight, NewtonHessian &H, const EnergyType etype, bool projectionMask, VariableMask vars) const {
    if (vars != VariableMask::Defo) throw std::runtime_error("Unimplemented VariableMask");
    BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSheet.hessian");

    assembler().assembleHessian(H, mesh().numElements(), [this, etype, projectionMask, weight](size_t ei) { return CustomHEAData(*this, weight, ei, etype, projectionMask); });
}

////////////////////////////////////////////////////////////////////////////////
// Geometric quantities
////////////////////////////////////////////////////////////////////////////////
template <class Psi_2x2>
typename ElasticSheet<Psi_2x2>::MX2d ElasticSheet<Psi_2x2>::getPrincipalCurvatures() const {
    const auto &m = mesh();
    MX2d result(m.numElements(), 2);
    for (const auto e : m.elements()) {
        // Principal curvatures are the eigenvalues of the (asymmetric) shape operator h g^{-1},
        // where h and g are the first and second fundamental forms, respectively.
        // Sign conventions vary, but we take the (somewhat less common) convention that
        // a sphere's princinpal curvatures are positive.
        const size_t ei = e.index();
        M32d FB = getFB(ei);
        M2d S = plateElement(ei).II * (FB.transpose() * FB).inverse();

        Eigen::EigenSolver<M2d> esolver(S);
        auto eigs = esolver.eigenvalues();
        if (eigs.imag().norm() / eigs.real().norm() > 1e-10) throw std::runtime_error("Non-real curvatures");
        result.row(ei) = eigs.real();
        if (result(ei, 0) > result(ei, 1)) std::swap(result(ei, 0), result(ei, 1));
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Internal state management
////////////////////////////////////////////////////////////////////////////////
template <class Psi_2x2>
void ElasticSheet<Psi_2x2>::m_adaptReferenceFrame() {
    if ((m_sourceReferenceFrame.size() != numEdges())
           || (m_referenceFrame.size() != numEdges())) {
        throw std::logic_error("Invalid reference frame sizes");
    }

    // Use the source alphas to resolve the 2 * Pi ambiguity in alpha
    // definition by enforcing temporal continuity.
    // (Choose 2 pi offset to minimize change from source alpha)
    // Temporal coherence: choose 2 Pi offset to minimize change from previous theta.
    auto setCoherentAngle = [this](size_t hei, Real alpha) {
        m_alphas[hei] = alpha + (2 * M_PI) * std::round(stripAutoDiff((m_sourceAlphas[hei] - alpha) / (2 * M_PI)));
    };

    const auto &m = mesh();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numEdges()),
                      [&](const tbb::blocked_range<size_t> &r) {
        for (size_t edgeIndex = r.begin(); edgeIndex < r.end(); ++edgeIndex) {
            auto he = m.halfEdge(halfEdgeForEdge(edgeIndex));
            M3d f_ref;
            f_ref.col(0) = (deformedEdgeVector(he)).normalized().transpose();
            const M3d &f_src = m_sourceReferenceFrame[edgeIndex];
            // if (edgeIndex == 0) {
            //     std::cout << "Parallel transporting from " << f_src.col(0).transpose()
            //                                      << " to " << f_ref.col(0).transpose() << std::endl;
            //     std::cout << "vector: " << f_src.col(1).transpose() << std::endl;
            //     std::cout << "result: " << parallelTransportNormalized<Real>(f_src.col(0), f_ref.col(0), f_src.col(1)).transpose() << std::endl;
            // }
            f_ref.col(1) = parallelTransportNormalized<Real>(f_src.col(0), f_ref.col(0), f_src.col(1));
            f_ref.col(2) = parallelTransportNormalized<Real>(f_src.col(0), f_ref.col(0), f_src.col(2));

            auto hop = he.opposite();
            // Measure the ccw angle around the edge tangent from reference director d1 to the triangle normal.
            if (hop.tri()) { setCoherentAngle(hop.index(), angle<Real>(f_ref.col(0), f_ref.col(1), deformedTriNormal(hop.tri().index()))); }
                           { setCoherentAngle( he.index(), angle<Real>(f_ref.col(0), f_ref.col(1), deformedTriNormal(he .tri().index()))); }
            m_referenceFrame[edgeIndex] = f_ref;
       }
    });
}

template <class Psi_2x2>
void ElasticSheet<Psi_2x2>::m_updateElementEmbedding() {
    const auto &m = mesh();
    const size_t ne = m.numElements();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m.numElements()),
                      [&](const tbb::blocked_range<size_t> &r) {
        for (size_t ei = r.begin(); ei < r.end(); ++ei) {
            const auto &e = m.element(ei);
            m_shellElements[ei].embed(
                (CornerPositions() << m_deformedPositions.row(e.vertex(0).index()),
                                      m_deformedPositions.row(e.vertex(1).index()),
                                      m_deformedPositions.row(e.vertex(2).index())).finished()
            );
        }
    });
}

template <class Psi_2x2>
void ElasticSheet<Psi_2x2>::m_updateDeformedElements(bool positionsUpdated) {
    if (positionsUpdated) {
        m_updateElementEmbedding();
        m_adaptReferenceFrame();
    }

    const size_t ne = mesh().numElements();
    for (size_t ei = 0; ei < ne; ++ei)
        m_shellElements[ei].setGammas(getTriGammas(ei));
}
