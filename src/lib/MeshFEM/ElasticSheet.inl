#include "newton_optimizer/newton_optimizer.hh"

#define NORMAL_INFERENCE_PROBLEM_VERBOSITY 0

template <class Psi_C>
void ElasticSheet<Psi_C>::setIdentityDeformation() {
    const auto &m = mesh();

    // Set the deformed positions to the rest positions.
    m_deformedPositions.resize(m_numVertices, 3);
    for (const auto &v : m.vertices())
        m_deformedPositions.row(v.index()) = v.node()->p.transpose();
    m_updateDeformedElements();

    initializeMidedgeNormals();
}

// Quadratic minimization to infer midedge normals (thetas):
// minimize the squared Frobenius norm of the second fundamental form.
// (This objective is proportional to the bending energy stored in an isotropic
// plate with Young's modulus 1 and Poisson's ratio 0.)
// For convenience we use our Newton sovler even though it should always
// converge in a single iteration.
//
// In order to make the normals/curvature computed independent of the reference
// configuration, we pose the inference energy on the deformed mesh (pushing
// the second fundamental form forward to the deformed configuration). We
// calculate this pushed-forward fundamental form directly and then verify its
// pullback agrees with II.
template<class ESheet>
struct NormalInferenceProblem : public NewtonProblem {
    using M3d = typename ESheet::M3d;
    NormalInferenceProblem(ESheet &sheet) : m_sheet(sheet) {
        m_hessianSparsity = sheet.hessianSparsityPattern();
        const size_t to = sheet.thetaOffset();
        m_hessianSparsity.rowColRemoval([to](size_t i) { return i < to; });

        m_updateDeformedII();
    }

    virtual size_t numVars() const override { return m_sheet.numThetas(); }
    virtual void setVars(const VXd &vars) override { m_sheet.setThetas(vars.cast<typename ESheet::Real>()); m_updateDeformedII(); }
    virtual const VXd getVars() const override { return m_sheet.getThetas().template cast<double>(); }

    virtual Real energy() const override {
        Real result = 0.0;
        const size_t ne = m_deformedII.size();
        for (size_t ei = 0; ei < ne; ++ei)
            result += 0.5 * m_sheet.deformedElement(ei).volume() * m_deformedII[ei].squaredNorm();
        return result;
    }

    virtual VXd gradient(bool freshIterate = false) const override {
        VXd g(VXd::Zero(numVars()));
        for (const auto &e : m_sheet.mesh().elements()) {
            const size_t ei = e.index();
            const auto &II = m_deformedII[ei];
            const auto &de = m_sheet.deformedElement(ei);
            const Real A = de.volume();
            const Real dE_dpsi = A;
            for (const auto &he : e.halfEdges()) {
                const size_t edgeIdx = m_sheet.edgeForHalfEdge(he.index());
                const Real sign = he.isPrimary() ? 1.0 : -1.0;
                const Real len = m_sheet.deformedEdgeVector(he).norm();

                const auto &glambda = de.gradBarycentric().col(he.localIndex());
                const Real dE_d_A_gamma_div_len = (4 * dE_dpsi) * (II * glambda).dot(glambda); // Derivative of the energy with respect to the coefficient of `glambda \otimes glambda` in the shape operator.
                // The derivative with respect to the theta variables is simple
                g[edgeIdx] += ((sign * (A / len))) * dE_d_A_gamma_div_len;
            }
        }

        return g;
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }

protected:
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool /* projectionMask */) const override {
        result.setZero();
        for (const auto &e : m_sheet.mesh().elements()) {
            const size_t ei = e.index();
            const auto &II = m_deformedII[ei];
            const auto &de = m_sheet.deformedElement(ei);
            const Real A = de.volume();
            const Real dE_dpsi = A;
            for (const auto &he : e.halfEdges()) {
                const size_t edgeIdx = m_sheet.edgeForHalfEdge(he.index());
                const Real sign = he.isPrimary() ? 1.0 : -1.0;
                const Real len = m_sheet.deformedEdgeVector(he).norm();

                const auto &glambda = de.gradBarycentric().col(he.localIndex());
                const Real dE_d_A_gamma_div_len = (4 * dE_dpsi) * (II * glambda).dot(glambda); // Derivative of the energy with respect to the coefficient of `glambda \otimes glambda` in the shape operator.

                for (const auto &he_b : e.halfEdges()) {
                    const size_t edgeIdx_b = m_sheet.edgeForHalfEdge(he_b.index());
                    if (edgeIdx > edgeIdx_b) continue;

                    const auto &glambda_b = de.gradBarycentric().col(he_b.localIndex());

                    const Real sign_b = he_b.isPrimary() ? 1.0 : -1.0;
                    const Real len_b = m_sheet.deformedEdgeVector(he_b).norm();
                    const Real d2E_d2_A_gamma_div_len_ab = 4 * (4 * dE_dpsi) * std::pow(glambda.dot(glambda_b), 2);

                    // (Shape operator/gamma are linear in theta, so (delta_b d_A_gamma_div_len_d_xa) term vanishes.
                    const Real delta_b_dE_d_A_gamma_div_len = ((sign_b * (A / len_b))) * d2E_d2_A_gamma_div_len_ab;

                    result.addNZ(edgeIdx, edgeIdx_b, (sign * (A / len)) * delta_b_dE_d_A_gamma_div_len);
                }
            }
        }
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        result.setIdentity(true);
    }

    void m_updateDeformedII() {
        const auto &m = m_sheet.mesh();
        m_deformedII.resize(m.numElements());
        const auto &II = m_sheet.getII();
        const auto &gammas = m_sheet.getGammas();

        for (const auto &e : m.elements()) {
            const size_t ei = e.index();
            auto &II_d = m_deformedII[ei];
            II_d.setZero();
            const auto &de = m_sheet.deformedElement(e.index());
            for (const auto &he : e.halfEdges()) {
                auto glambda = de.gradBarycentric().col(he.localIndex());
                Real len = m_sheet.deformedEdgeVector(he).norm();
                II_d += ((4 * gammas[he.index()] * (de.volume() / len)) * glambda) * glambda.transpose();
            }

            M3d F = m_sheet.getCornerPositions(ei) * e->gradBarycentric().transpose();
            if ((II[ei] - F.transpose() * II_d * F).squaredNorm() / II[ei].squaredNorm() > 1e-18)
                throw std::runtime_error("Second fundamental form pushforward mismatch.");
        }
    }

    ESheet &m_sheet;
    mutable SuiteSparseMatrix m_hessianSparsity;
    std::vector<M3d> m_deformedII;
};

template <class Psi_C>
void ElasticSheet<Psi_C>::initializeMidedgeNormals(bool minimizeBending) {
    const auto &m = mesh();

    // Initialize the reference frames.
    // We pick the averaged edge normals as the initial d1 frame vector and midedge normal.
    m_referenceFrame.resize(m_numEdges);
    m.visitEdges([this, &m](CHEHandle he, size_t edgeIndex) {
        V3d t  = (deformedEdgeVector(he)).normalized().transpose();
        V3d d1 = m_deformedElements[he.tri().index()].normal();
        if (!he.isBoundary()) d1 += m_deformedElements[he.opposite().tri().index()].normal();
        d1 = d1.normalized();

        if (std::abs(t.dot(d1)) > 1e-14) throw std::logic_error("Non-perpendicular averaged edge normal: " + std::to_string(t.dot(d1)));

        m_referenceFrame[edgeIndex] << t, d1, t.cross(d1); // Generate the third vector of the right-handed frame.
    });

    // Measure the angle around the edge tangent from reference director d1 to the triangle normal.
    // (ccw with tip pointing toward us)
    m_alphas.resize(m.numHalfEdges());
    for (const auto &he : m.halfEdges()) {
        const auto &frame = m_referenceFrame[m_edgeForHalfEdge[he.index()]];

        const auto &n = m_deformedElements[he.tri().index()].normal();
        m_alphas[he.index()] = angle<Real>(frame.col(0), frame.col(1), n);
        if (std::abs(m_alphas[he.index()]) > M_PI / 2) { // Shouldn't happen except for sharp creases
            std::cout << "WARNING: Large alpha: " << m_alphas[he.index()] << std::endl;
            std::cout << frame << std::endl;
            std::cout << "Tri normal: " << n.transpose() << std::endl;

            V3d n_avg = n;
            if (he.opposite().tri())
                n_avg += m_deformedElements[he.opposite().tri().index()].normal();
            n_avg = n_avg.normalized();
            std::cout << "Averaged edge normal: " << n_avg.transpose() << std::endl << std::endl;
            std::cout << "For he, edge: " << he.index() << ", " << m_edgeForHalfEdge[he.index()] << std::endl;
        }
    }

    // Apply the current frame/alphas as the source values.
    updateSourceFrame();

    // Initialize with midedge normals coinciding with the averaged edge normals.
    // Side effect: updates the cached shape operator and midedge normals.
    setThetas(VXd::Zero(m_numEdges));

    // Finally, infer the "best" midedge normals by minimizing the bending energy with respect to theta.
    if (minimizeBending) {
        auto problem = std::make_unique<NormalInferenceProblem<ElasticSheet>>(*this);
        auto opt = std::make_unique<NewtonOptimizer>(std::move(problem));
        opt->options.verbose = NORMAL_INFERENCE_PROBLEM_VERBOSITY;
        opt->optimize();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Elastic Energy
////////////////////////////////////////////////////////////////////////////////
template <class Psi_C>
typename ElasticSheet<Psi_C>::Real ElasticSheet<Psi_C>::elementEnergy(size_t ei, const EnergyType etype) const {
    const auto &m = mesh();

    Psi_C psi(getEnergyDensity(ei), UninitializedDeformationTag());
    const auto &e = m.element(ei);
    const M32d &B = m_B[ei];
    Real result = 0.0;

    // Membrane energy contribution
    if ((etype == EnergyType::Membrane) || (etype == EnergyType::Full)) {
        M32d FB = getCornerPositions(ei) * (e->gradBarycentric().transpose() * B);
        psi.setC(FB.transpose() * FB);
        result += m_h * psi.energy();
    }

    // Bending energy contribution
    // (Only an approximation unless Psi_C is actually St Venant Kirchhoff...)
    if (!m_disableBending && ((etype == EnergyType::Bending) || (etype == EnergyType::Full))) {
        // We obtain a 2x2 second fundamental form in the reference configuration
        // using our orthonormal basis for the undeformed triangle.
        psi.setC(2 * (B.transpose() * (m_II[ei] - m_restII[ei]) * B) + M2d::Identity());
        result += (std::pow(m_h, 3) / 12.0) * psi.energy();
    }

    return result * e->volume();
}

template <class Psi_C>
typename ElasticSheet<Psi_C>::Real ElasticSheet<Psi_C>::energy(const EnergyType etype) const {
    return summation_parallel<Real>([this, etype](size_t ei) { return elementEnergy(ei, etype); },
                                    mesh().numElements());
}

////////////////////////////////////////////////////////////////////////////////
// Elastic Energy Gradient
////////////////////////////////////////////////////////////////////////////////
template <class Psi_C>
typename ElasticSheet<Psi_C>::ElementGradient ElasticSheet<Psi_C>::elementGradient(size_t ei, bool updatedSource, const EnergyType etype) const {
    ElementGradient g_e(ElementGradient::Zero());

    Psi_C psi(getEnergyDensity(ei), UninitializedDeformationTag());
    const auto &m = mesh();
    const auto &e = m.element(ei);
    const M32d &B = m_B[ei];

    // Membrane energy contribution
    if ((etype == EnergyType::Membrane) || (etype == EnergyType::Full)) {
        M32d FB = getCornerPositions(ei) * (e->gradBarycentric().transpose() * B);
        psi.setC(FB.transpose() * FB);

        // Derivative of `h * A * psi` with respect to C
        M2d two_dpsi_hA = psi.PK2Stress() * (e->volume() * m_h);

        // C = B^T F^T F B
        // delta C = B^T deltaF^T F B + B^T F^T deltaF B
        // delta psi = dpsi_hA : delta C = 2 dpsi_hA : B^T F^T deltaF B
        //           = FB * 2 dpsi_hA * B^T : deltaF
        M3d d_psi_dF = FB * two_dpsi_hA * B.transpose();

        // d_psi_dF : (e_c otimes grad lambda_v)
        //      = e_c . d_psi_dF * (grad lambda_v)
        //      = (d_psi_dF * (grad lambda_v))_c
        Eigen::Map<M3d>(g_e.data()) = d_psi_dF * e->gradBarycentric();
    }

    // Bending energy contribution
    if (!m_disableBending && ((etype == EnergyType::Bending) || (etype == EnergyType::Full))) {
        psi.setC(2 * (B.transpose() * (m_II[ei] - m_restII[ei]) * B) + M2d::Identity());
        const Real dE_dpsi = (e->volume() * std::pow(m_h, 3) / 12.0);
        const M2d stress = psi.PK2Stress();
        constexpr size_t to = 3 * numNodesPerElement;
        const Real A = m_deformedElements[ei].volume();

        for (const auto &he : e.halfEdges()) {
            const V2d Bt_glambda_ref = B.transpose() * e->gradBarycentric().col(he.localIndex());
            const Real sign = he.isPrimary() ? 1.0 : -1.0;
            const Real len = deformedEdgeVector(he).norm();
            const Real dE_d_A_gamma_div_len = (4 * dE_dpsi) * (stress * Bt_glambda_ref).dot(Bt_glambda_ref); // Derivative of the energy with respect to the coefficient of `glambda \otimes glambda` in the shape operator.
            // The derivative with respect to the theta variables is simple.
            g_e[to + he.localIndex()] = ((sign * (A / len))) * dE_d_A_gamma_div_len;

            // Derivative with respect to the corner positions.
            Eigen::Map<M3d>(g_e.data()) += dE_d_A_gamma_div_len * d_A_gamma_div_len_d_x(he, updatedSource);;
        }
    }

    return g_e;
}

template <class Psi_C>
typename ElasticSheet<Psi_C>::VXd ElasticSheet<Psi_C>::gradient(bool updatedSource, const EnergyType etype) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Gradient");
    const auto &m = mesh();
    auto accumulate_per_element_contrib = [this, updatedSource, etype, &m](size_t ei, VXd &g_out) {
        auto g_e = elementGradient(ei, updatedSource, etype);
        const auto &e = m.element(ei);

        g_out.template segment<3>(3 * e.vertex(0).index()) += g_e.template segment<3>(0);
        g_out.template segment<3>(3 * e.vertex(1).index()) += g_e.template segment<3>(3);
        g_out.template segment<3>(3 * e.vertex(2).index()) += g_e.template segment<3>(6);

        const size_t to = thetaOffset();
        for (const auto &he : e.halfEdges())
            g_out[to + m_edgeForHalfEdge[he.index()]] += g_e[9 + he.localIndex()];
    };

    VXd g(VXd::Zero(numVars()));
    assemble_parallel(accumulate_per_element_contrib, g, m.numElements());

    return g;
}

template <class Psi_C>
template <class SHEHandle>
typename ElasticSheet<Psi_C>::M3d ElasticSheet<Psi_C>::d_A_gamma_div_len_d_x(const SHEHandle &he, bool updatedSource) const {
    const Real sign = he.isPrimary() ? 1.0 : -1.0;
    const V3d eVec = deformedEdgeVector(he);
    const Real len = eVec.norm();
    const Real gamma = m_thetas[m_edgeForHalfEdge[he.index()]] - m_alphas[he.index()];
    const auto &deformedElement = m_deformedElements[he.tri().index()];
    const Real A = deformedElement.volume();
    const Real A_div_len = A / len;
    const M3d A_div_len_gradLambdas = A_div_len * deformedElement.gradBarycentric();

    // The derivative with respect to vertex positions is more complicated:
    // these change both alpha (by changing both the triangle normal and the reference frame)
    // and the quantity (A / len).
    M3d gradCornerPos(   gamma * A_div_len_gradLambdas);  // Derivative of area term
    V3d dcoeff_dedge = -(gamma * A_div_len / len) * (eVec / len); // Change of len term

    // Parallel transport term (reference frame rotation)
    // The reference frame only twists around the edge if the source frame is not updated!
    if (!updatedSource) {
        const size_t edgeIdx = m_edgeForHalfEdge[he.index()];
        const auto &srcFrame = m_sourceReferenceFrame[edgeIdx];
        const auto &curFrame =       m_referenceFrame[edgeIdx];
        const auto &t   = curFrame.col(0), &ts  = srcFrame.col(0),
                   &d1  = curFrame.col(1), &ds1 = srcFrame.col(1),
                   &d2  = curFrame.col(2), &ds2 = srcFrame.col(2);

        const Real inv_chi_hat = 1.0 / (1.0 + ts.dot(t));
        V3d neg_dalpha_dt = (ds2.dot(t) * ts.cross(d2) + d1.dot(ts) * ds2) * inv_chi_hat
                          - (ds2.dot(t) * d1.dot(ts) * inv_chi_hat * inv_chi_hat) * ts
                          + d2.cross(ds2);
        // When d1 rotates ccw, alpha decreases ==> gamma increases
        dcoeff_dedge += sign * (A / (len * len)) * (neg_dalpha_dt - t.dot(neg_dalpha_dt) * t);
    }
    gradCornerPos.col((he.localIndex() + 2) % 3) += dcoeff_dedge; // local tip
    gradCornerPos.col((he.localIndex() + 1) % 3) -= dcoeff_dedge; // local tail

    // Normal rotation term: change in alpha due to the rotating normal
    // The derivative of the normal with respect to vertex i is (-glambda_i otimes n)
#if 0
    // Unsimplified version
    V3d eperp_hat = sign * deformedElement.gradBarycentric().col(he.localIndex()).normalized(); // Unit vector perpendicular to both normal and he; used to measure derivative of -angle around he.
    //     increasing alpha decreases gamma              Normal rotation in the positive direction around he (alpha increase)
    //            |                                             ________________________________________________________
    //            v                                           /                                                         \,
    gradCornerPos -= ((A / len) * deformedElement.normal()) * (eperp_hat.transpose() * deformedElement.gradBarycentric());
#else
    // Equivalent, easier-to-differentiate version
    gradCornerPos -= (2 * sign * deformedElement.normal()) * (A_div_len_gradLambdas.col(he.localIndex()).transpose()
                                                           *  A_div_len_gradLambdas);
#endif
    gradCornerPos *= sign;

    // gradCornerPos = deformedElement.gradBarycentric();

    return gradCornerPos;
}

template <class Psi_C>
template <class SHEHandle>
typename ElasticSheet<Psi_C>::M3d ElasticSheet<Psi_C>::d2_A_gamma_div_len_d_x_dtheta(const SHEHandle &he) const {
    // (Assumes an updated source frame since this is only called from `hessian`)
    const V3d eVec = deformedEdgeVector(he);
    const Real len = eVec.norm();
    const auto &deformedElement = m_deformedElements[he.tri().index()];
    const Real A = deformedElement.volume();
    const Real sign = he.isPrimary() ? 1.0 : -1.0;

    // The derivative with respect to vertex positions is more complicated:
    // these change both alpha (by changing both the triangle normal and the reference frame)
    // and the quantity (A / len).
    M3d dgradCornerPos_dtheta(  (A / len) * deformedElement.gradBarycentric());  // Derivative of area term
    V3d dcoeff_dedge = -(A / (len * len)) * (eVec / len); // Change of len term

    dgradCornerPos_dtheta.col((he.localIndex() + 2) % 3) += dcoeff_dedge; // local tip
    dgradCornerPos_dtheta.col((he.localIndex() + 1) % 3) -= dcoeff_dedge; // local tail
    dgradCornerPos_dtheta *= sign;

    return dgradCornerPos_dtheta;
}

// TODO reduce duplicated work by returning a 3rd order tensor of the derivatives wrt all three components of v_b?
template <class Psi_C>
template <class SHEHandle, class SVHandle>
typename ElasticSheet<Psi_C>::M3d ElasticSheet<Psi_C>::delta_d_A_gamma_div_len_d_x(const SHEHandle &he, const SVHandle &v_b, const size_t c_b) const {
    // (Assumes an updated source frame since this is only called from `hessian`)
    const Real sign = he.isPrimary() ? 1.0 : -1.0;
    const V3d eVec = deformedEdgeVector(he);
    const Real len = eVec.norm();
    const Real gamma = m_thetas[m_edgeForHalfEdge[he.index()]] - m_alphas[he.index()];
    const auto &deformedElement = m_deformedElements[he.tri().index()];
    const M3d &gradLambdas = deformedElement.gradBarycentric();
    const V3d &n = deformedElement.normal();
    const Real A = deformedElement.volume();

    // Intermediate quantities (values and their derivatives with respect to component c_b of v_b's deformed position.)
    const Real A_div_len = A / len;
    const M3d A_div_len_gradLambdas = A_div_len * gradLambdas;

    const Real d_eVec_dv_b = (v_b.index() == he.tip().index()) ? 1.0 : ((v_b.index() == he.tail().index()) ? -1.0 : 0.0);
    const Real deltaLen = d_eVec_dv_b * eVec[c_b] / len;

    const Real delta_A_div_len = A_div_len_gradLambdas(c_b, v_b.localIndex()) - (A_div_len / len) * deltaLen;
    const V3d delta_n = -gradLambdas.col(v_b.localIndex()) * n[c_b];

    const M3d delta_A_div_len_gradLambdas = delta_A_div_len * gradLambdas
                                          - gradLambdas.col(v_b.localIndex()) * A_div_len_gradLambdas.row(c_b)
                                          - n * (delta_n.transpose() * A_div_len_gradLambdas);

    // Gamma increases (alpha decreases) as the triangle normal rotates towards eperp
    const Real delta_gamma = sign * 2 * A_div_len_gradLambdas.col(he.localIndex()).dot(delta_n);

    // delta_gamma = 0.0;

    M3d delta_gradCornerPos(delta_gamma * A_div_len_gradLambdas + gamma * delta_A_div_len_gradLambdas);  // Derivative of area term

    V3d delta_dcoeff_dedge = - (delta_gamma * A_div_len / (len * len)) * eVec
                       - (gamma * delta_A_div_len / (len * len)) * eVec
                   + 2 * (gamma * (A_div_len / (len * len * len)) * deltaLen) * eVec;
    delta_dcoeff_dedge[c_b] -=       (gamma * (A_div_len / (len * len))) * d_eVec_dv_b;

    // Parallel transport term Hessian (assuming updated source)
    if (d_eVec_dv_b != 0.0) {
        const size_t edgeIdx = m_edgeForHalfEdge[he.index()];
        const auto &t = m_referenceFrame[edgeIdx].col(0);
        // Note: edge signs cancel!
        delta_dcoeff_dedge -= d_eVec_dv_b * (A_div_len / (2 * len * len)) * t.cross(M3d::Identity().col(c_b));
    }

    delta_gradCornerPos.col((he.localIndex() + 2) % 3) += delta_dcoeff_dedge; // local tip
    delta_gradCornerPos.col((he.localIndex() + 1) % 3) -= delta_dcoeff_dedge; // local tail

    delta_gradCornerPos -= (2 * sign * delta_n) * (      A_div_len_gradLambdas.col(he.localIndex()).transpose() *       A_div_len_gradLambdas);
    delta_gradCornerPos -= (2 * sign *       n) * (delta_A_div_len_gradLambdas.col(he.localIndex()).transpose() *       A_div_len_gradLambdas);
    delta_gradCornerPos -= (2 * sign *       n) * (      A_div_len_gradLambdas.col(he.localIndex()).transpose() * delta_A_div_len_gradLambdas);
    delta_gradCornerPos *= sign;

    return delta_gradCornerPos;
}

////////////////////////////////////////////////////////////////////////////////
// Elastic Energy Hessian
////////////////////////////////////////////////////////////////////////////////
template <class Psi_C>
void ElasticSheet<Psi_C>::hessian(SuiteSparseMatrix &H, const EnergyType etype, bool /* projectionMask */) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Hessian");
    const auto &m = mesh();
    auto assembler_per_element_contrib = [&m, this, etype](size_t ei, SuiteSparseMatrix& Hout) {
        const auto &e = m.element(ei);
        Psi_C psi(getEnergyDensity(ei), UninitializedDeformationTag());
        const M32d &B = m_B[ei];

        Eigen::Matrix<Real, 9, 9> Hxx;
        Hxx.setZero();

        // Membrane energy contribution
        if ((etype == EnergyType::Membrane) || (etype == EnergyType::Full)) {
            M32d FB = getCornerPositions(ei) * (e->gradBarycentric().transpose() * B);
            psi.setC(FB.transpose() * FB);

            for (const auto &v_b : e.vertices()) {
                VSFJ deltaF_b(0, e->gradBarycentric().col(v_b.localIndex()));
                for (size_t c_b = 0; c_b < 3; ++c_b) {
                    deltaF_b.c = c_b;
                    size_t var_b = 3 * v_b.localIndex() + c_b;
                    M2d deltaC_b = FB.transpose() * deltaF_b * B;
                    deltaC_b = (deltaC_b + deltaC_b.transpose()).eval();

                    //        d_psi_dF = FB * two_dpsi_hA * B.transpose();
                    M3d delta_d_psi_dF = (((deltaF_b * B) * psi.PK2Stress() + FB * psi.delta_PK2Stress(deltaC_b)) * B.transpose()) * (e->volume() * m_h);

                    for (const auto &v_a : e.vertices()) {
                        if (v_a.index() > v_b.index()) continue;
                        VSFJ deltaF_a(0, e->gradBarycentric().col(v_a.localIndex()));
                        for (size_t c_a = 0; c_a < 3; ++c_a) {
                            if ((v_a.index() == v_b.index()) && c_a > c_b) continue;
                            size_t var_a = 3 * v_a.localIndex() + c_a;
                            deltaF_a.c = c_a;
                            Hxx(var_a, var_b) += doubleContract(delta_d_psi_dF, deltaF_a);
                        }
                    }
                }
            }
        }

        // Bending energy contribution
        // This can be heavily optimized...
        if (!m_disableBending && ((etype == EnergyType::Bending) || (etype == EnergyType::Full))) {
            psi.setC(2 * (B.transpose() * (m_II[ei] - m_restII[ei]) * B) + M2d::Identity());
            const Real dE_dpsi = (e->volume() * std::pow(m_h, 3) / 12.0);
            const M2d stress = 1.0 * psi.PK2Stress();

            const size_t to = thetaOffset();
            const auto &deformedElement = m_deformedElements[ei];

            for (const auto &he : e.halfEdges()) {
                const V2d Bt_glambda_ref = B.transpose() * e->gradBarycentric().col(he.localIndex());
                const size_t edgeIdx = m_edgeForHalfEdge[he.index()];
                const Real sign = he.isPrimary() ? 1.0 : -1.0;
                const V3d eVec = deformedEdgeVector(he);
                const Real len = eVec.norm();
                const Real A = deformedElement.volume();
                const Real dE_d_A_gamma_div_len = (4 * dE_dpsi) * (stress * Bt_glambda_ref).dot(Bt_glambda_ref); // Derivative of the energy with respect to the coefficient of `glambda \otimes glambda` in the shape operator.
                const M3d d_A_gamma_div_len_d_xa = d_A_gamma_div_len_d_x(he, true);

                M3d d2_E_d_A_gamma_div_len_dx(M3d::Zero());

                for (const auto &he_b : e.halfEdges()) {
                    const V2d Bt_glambda_ref_b = B.transpose() * e->gradBarycentric().col(he_b.localIndex());
                    const Real sign_b = he_b.isPrimary() ? 1.0 : -1.0;
                    const Real len_b = deformedEdgeVector(he_b).norm();
                    const size_t edgeIdx_b = m_edgeForHalfEdge[he_b.index()];
                    // TODO: optimize this by factoring delta_PK2Stress call outside the loop and caching it -- taking advantage of major symmetry).
                    const Real d2E_d2_A_gamma_div_len_ab = 4 * (4 * 2 * dE_dpsi) * Bt_glambda_ref.dot(psi.delta_PK2Stress(Bt_glambda_ref_b * Bt_glambda_ref_b.transpose()) * Bt_glambda_ref);

                    {
                        // theta-theta block
                        //      (Shape operator/gamma are linear in theta, so (delta_b d_A_gamma_div_len_d_xa) term vanishes.
                        const Real delta_b_dE_d_A_gamma_div_len = ((sign_b * (A / len_b))) * d2E_d2_A_gamma_div_len_ab;

                        if (edgeIdx <= edgeIdx_b)
                            Hout.addNZ(to + edgeIdx, to + edgeIdx_b, (sign * (A / len)) * delta_b_dE_d_A_gamma_div_len);

                        // x-theta block
                        M3d delta_gradCornerPos = delta_b_dE_d_A_gamma_div_len * d_A_gamma_div_len_d_xa;
                        if (he_b == he) // d_A_gamma_div_len_d_x for "he" is constant wrt. the other edges' thetas.
                            delta_gradCornerPos += dE_d_A_gamma_div_len * d2_A_gamma_div_len_d_x_dtheta(he);

                        Hout.addNZ(3 * e.vertex(0).index(), to + edgeIdx_b, delta_gradCornerPos.col(0));
                        Hout.addNZ(3 * e.vertex(1).index(), to + edgeIdx_b, delta_gradCornerPos.col(1));
                        Hout.addNZ(3 * e.vertex(2).index(), to + edgeIdx_b, delta_gradCornerPos.col(2));
                    }

                    // Precompute quantities needed for x-x block
                    // (Effect of the full changing shape operator due to perturbing x).
                    // TODO: hoist this out of the loop too...
                    d2_E_d_A_gamma_div_len_dx += d2E_d2_A_gamma_div_len_ab * d_A_gamma_div_len_d_x(he_b, true);
                }

                // x-x block
                for (const auto &v_b : e.vertices()) {
                    for (size_t c_b = 0; c_b < 3; ++c_b) {
                        M3d delta_gradCornerPos = d2_E_d_A_gamma_div_len_dx(c_b, v_b.localIndex()) * d_A_gamma_div_len_d_xa
                                                +   dE_d_A_gamma_div_len * delta_d_A_gamma_div_len_d_x(he, v_b, c_b);
                        Hxx.col(3 * v_b.localIndex() + c_b) += Eigen::Map<Eigen::Matrix<Real, 9, 1>>(delta_gradCornerPos.data());

                    }
                }
            }

        }
#if 0
        // Symmetry test: the full Hxx must be constructed to run this (disable lower triangle skip in membrane term).
        if ((Hxx - Hxx.transpose()).squaredNorm() / Hxx.squaredNorm() > 1e-10) {
            std::cout << "Asymmetric hessian contrib:" << std::endl;
            std::cout << Hxx;
            std::cout << std::endl;
            throw std::runtime_error("Asymmetric hessian contrib");
        }
#endif
        for (const auto &v_b : e.vertices()) {
            for (size_t c_b = 0; c_b < 3; ++c_b) {
                const size_t var_b = 3 * v_b.index() + c_b;
                for (const auto &v_a : e.vertices()) {
                    if (v_a.index() > v_b.index()) continue;
                    Hout.addNZ(3 * v_a.index(), var_b, Hxx.col(3 * v_b.localIndex() + c_b).segment(3 * v_a.localIndex(), (v_a.index() == v_b.index()) ? c_b + 1 : 3));
                }
            }
        }
    };

    assemble_parallel(assembler_per_element_contrib, H, m.numElements());
}

template <class Psi_C>
SuiteSparseMatrix ElasticSheet<Psi_C>::hessianSparsityPattern(Real val) const {
    SuiteSparseMatrix Hsp(numVars(), numVars());
    Hsp.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    Hsp.Ap.reserve(numVars() + 1);

    auto &Ap = Hsp.Ap;
    auto &Ai = Hsp.Ai;

    const auto &m = mesh();

    // Build the sparsity pattern in compressed form one column (variable) at a time.
    Hsp.Ap.push_back(0);

    auto finalizeCol = [&]() {
        const size_t colStart = Ap.back();
        const size_t colEnd = Ai.size();
        Ap.push_back(colEnd);
        std::sort(Ai.begin() + colStart, Ai.begin() + colEnd);
    };

    // Each vertex DoF interacts with the vertices in its one-ring
    for (const auto &v : m.vertices()) {
        const size_t vi = v.index();

        for (size_t v_comp = 0; v_comp < 3; ++v_comp) {
            // Self-interaction (upper triangle)
            for (size_t c = 0; c <= v_comp; ++c)
                Ai.push_back(3 * vi + c);
            for (const auto &he : v.incidentHalfEdges()) {
                const size_t ui = he.tail().index();
                if (ui < vi) {
                    for (size_t c = 0; c < 3; ++c)
                        Ai.push_back(3 * ui + c);
                }
            }
            finalizeCol();
        }
    }

    // Each midedge normal angle interacts with the vertices and midedge normals in its attached triangles.
    const size_t to = thetaOffset();
    m.visitEdges([&](CHEHandle he, size_t edgeIndex) {
        // Self-interaction
        Ai.push_back(to + edgeIndex);

        // Vertices within this triangle
        for (const auto v : he.tri().vertices()) {
            for (size_t c = 0; c < 3; ++c)
                Ai.push_back(3 * v.index() + c);
        }

        // Single opposite vertex in opposite triangle
        // (don't add edge endpoints twice)
        if (he.opposite().tri()) {
            size_t vopp = he.opposite().next().tip().index();
            for (size_t c = 0; c < 3; ++c)
                Ai.push_back(3 * vopp + c);
        }

        // Interaction with other variables in stencil
        he.visitIncidentElements([&](size_t ti) {
            const auto t = m.tri(ti);
            // *Other* theta variables
            for (const auto &he : t.halfEdges()) {
                size_t otherEdgeIdx = m_edgeForHalfEdge[he.index()];
                if (otherEdgeIdx < edgeIndex)
                    Ai.push_back(to + otherEdgeIdx);
            }
        });
        finalizeCol();
    });

    Hsp.nz = Ai.size();
    Hsp.Ax.assign(Hsp.nz, val);

    return Hsp;
}

////////////////////////////////////////////////////////////////////////////////
// Geometric quantities
////////////////////////////////////////////////////////////////////////////////
template <class Psi_C>
typename ElasticSheet<Psi_C>::MX2d ElasticSheet<Psi_C>::getPrincipalCurvatures() const {
    const auto &m = mesh();
    MX2d result(m.numElements(), 2);
    for (const auto &e : m.elements()) {
        // Principal curvatures are the eigenvalues of the (asymmetric) shape operator h g^{-1},
        // where h and g are the first and second fundamental forms, respectively.
        // Sign conventions vary, but we take the (somewhat less common) convention that
        // a sphere's princinpal curvatures are positive.
        const size_t ei = e.index();
        M32d FB = getCornerPositions(ei) * (e->gradBarycentric().transpose() * m_B[ei]);
        M2d S = (m_B[ei].transpose() * m_II[ei] * m_B[ei]) * (FB.transpose() * FB).inverse();

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
template <class Psi_C>
void ElasticSheet<Psi_C>::m_adaptReferenceFrame() {
    if ((m_sourceReferenceFrame.size() != m_numEdges)
           || (m_referenceFrame.size() != m_numEdges)) {
        throw std::logic_error("Invalid reference frame sizes");
    }

    mesh().visitEdges([this](HEHandle he, size_t edgeIndex) {
        M3d &f_ref = m_referenceFrame[edgeIndex];
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
        if (hop.tri()) m_alphas[hop.index()] = angle<Real>(f_ref.col(0), f_ref.col(1), m_deformedElements[hop.tri().index()].normal());
                       m_alphas[ he.index()] = angle<Real>(f_ref.col(0), f_ref.col(1), m_deformedElements[he .tri().index()].normal());
    });

    // Use the source alphas to resolve the 2 * Pi ambiguity in alpha
    // definition by enforcing temporal continuity.
    // (Choose 2 pi offset to minimize change from source alpha)
    // Temporal coherence: choose 2 Pi offset to minimize change from previous theta.
    m_alphas += (2 * M_PI) * stripAutoDiff((m_sourceAlphas - m_alphas) / (2 * M_PI)).array().round().matrix();

    m_updateMidedgeNormals();
    m_updateShapeOperators();
}

template <class Psi_C>
void ElasticSheet<Psi_C>::m_updateMidedgeNormals() {
    m_midedgeNormals.resize(m_numEdges, 3);
    for (size_t i = 0; i < m_numEdges; ++i) {
        m_midedgeNormals.row(i) = std::cos(m_thetas[i]) * m_referenceFrame[i].col(1) +
                                  std::sin(m_thetas[i]) * m_referenceFrame[i].col(2);
    }
}

template <class Psi_C>
void ElasticSheet<Psi_C>::m_updateDeformedElements() {
    const auto &m = mesh();
    m_deformedElements.resize(m.numElements());
    for (const auto &e : m.elements()) {
        m_deformedElements[e.index()].embed(m_deformedPositions.row(e.vertex(0).index()).transpose(),
                                            m_deformedPositions.row(e.vertex(1).index()).transpose(),
                                            m_deformedPositions.row(e.vertex(2).index()).transpose());
    }
}

template <class Psi_C>
void ElasticSheet<Psi_C>::m_updateShapeOperators() {
    const auto &m = mesh();
    m_II.resize(m.numTris());
    auto gammas = getGammas();

    for (const auto &e : m.elements()) {
        auto &result = m_II[e.index()];
        result.setZero();
        const auto &deformedElement = m_deformedElements[e.index()];
        for (const auto &he : e.halfEdges()) {
            auto glambda_ref = e->gradBarycentric().col(he.localIndex());
            Real len = deformedEdgeVector(he).norm();
            result += ((4 * gammas[he.index()] * (deformedElement.volume() / len)) * glambda_ref) * glambda_ref.transpose();
        }
    }
}

template <class Psi_C>
void ElasticSheet<Psi_C>::m_updateB() {
    // Generate an orthonormal basis for the tangent plane of each triangle.
    const auto &m = mesh();
    const size_t nt = m.numTris();

    // First, check if we actually have a plate in the z = 0 plane; in this
    // case we use the global 2D coordinate system's axis vectors as our
    // orthonormal basis to ease specification of anisotropic materials.
    if (std::abs(m.boundingBox().dimensions()[2]) < 1e-16) {
        M32d globalB(M32d::Identity());
        m_B.assign(nt, M32d::Identity().eval());
        return;
    }

    m_B.resize(nt);
    for (auto tri : m.elements()) {
        V3d b0 = (tri.node(1)->p - tri.node(0)->p).normalized();
        V3d b1 = tri->normal().cross(b0);
        const size_t ti = tri.index();
        m_B[ti].col(0) = b0;
        m_B[ti].col(1) = b1;
    }
}
