////////////////////////////////////////////////////////////////////////////////
// RotationStrainExtrapolation.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Base interface for extrapolators used to predict updated variables along a
//  linesearch ray x(α) starting from a base point x0 in direction d.
//
//  This mirrors the Python extrapolator structure used in the research code:
//      - linesearch_begin(x0, d)   : precompute/cache per-ray quantities
//      - linesearch_eval(alpha)    : evaluate extrapolated point at alpha
//      - operator()(x0, coeffs, alphas) convenience wrapper
//
//  The derived classes (e.g. RSNewtonFlowExtrapolator) should implement the
//  two virtual methods and can optionally override the batch-eval behavior.
//
//  Author:  Johnson Hu, xinzhuohu@gmail.com
//  Company: University of California, Davis
//  Created: 02/24/2026 22:09:55
*///////////////////////////////////////////////////////////////////////////////
#ifndef ROTATIONSTRAINEXTRAPOLATION_HH
#define ROTATIONSTRAINEXTRAPOLATION_HH
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/Laplacian.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEM/Types.hh>
#include "PoissonGradientIntegration.hh"

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>

namespace rotation_strain_extrapolation {

static constexpr size_t N = 2;

using V2d   = Eigen::Vector2d;
using V3d   = Eigen::Vector3d;
using VXd   = Eigen::VectorXd;
using MNd   = Eigen::Matrix<Real, N, N>;

using MXd   = Eigen::MatrixXd;
using MXNd  = Eigen::Matrix<Real, Eigen::Dynamic, N, Eigen::RowMajor>;

using UVMat = MXNd;
using UVMatMapConst = Eigen::Map<const UVMat>;
using UVMatMap = Eigen::Map<UVMat>;
using VecMapConst = Eigen::Map<const VXd>;
using VecMap = Eigen::Map<VXd>;

// C++ counterpart of python/Stretch2Relax/extra_utils.py:getLaplacianFactorizer.
// `fixedVars` are pinned scalar DoF indices used to remove the nullspace.
template<class Mesh>
std::unique_ptr<CholeskyFactorizerBase>
getLaplacianFactorizer(const Mesh &m, const std::vector<size_t> &fixedVars = {}) {
    auto L = Laplacian::construct(m); // upper triangle by construction
    SuiteSparseMatrix Lsparse(std::move(L));
    Lsparse.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;

    auto Linv = make_cholesky_factorizer(get_default_cholesky_provider());
    Linv->factorize(Lsparse, fixedVars);
    return Linv;
}

// C++ counterpart of python/Stretch2Relax/extra_utils.py:extrapolateDeformGrad.
// Supported modes intentionally mirror the currently requested scope:
//   - "Eulerian"
//   - "Linear"
// and throw for "Lagrangian".
void extrapolateDeformGrad(const std::vector<MNd> &F,
                      Real alpha,
                      const std::vector<MNd> &d_grad,
                      std::vector<MNd> &F_extra,
                      const std::vector<MNd> &F_inv,
                      const std::string &method = "Eulerian")
{
    BENCHMARK_SCOPED_TIMER_SECTION timer("extrapolateDeformGrad");
    if (F.size() != d_grad.size())
        throw std::runtime_error("extrapolateDeformGrad: F and d_grad must have the same number of elements.");

    const size_t ne = F.size();
    // for (size_t ei = 0; ei < ne; ++ei) {
    //     if ((F[ei].rows() != 2) || (F[ei].cols() != 2))
    //         throw std::runtime_error("extrapolateDeformGrad: each F[ei] must be 2x2.");
    //     if ((d_grad[ei].rows() != 2) || (d_grad[ei].cols() != 2))
    //         throw std::runtime_error("extrapolateDeformGrad: each d_grad[ei] must be 2x2.");
    // }

        if (F_inv.size() != ne)
            throw std::runtime_error("extrapolateDeformGrad: F_inv must have the same number of elements as F.");
        // for (size_t ei = 0; ei < ne; ++ei) {
        //     if (((*F_inv)[ei].rows() != 2) || ((*F_inv)[ei].cols() != 2))
        //         throw std::runtime_error("extrapolateDeformGrad: each F_inv[ei] must be 2x2.");
        // }

    if (method == "Linear") {
        for (size_t ei = 0; ei < ne; ++ei)
            F_extra[ei] = F[ei] + alpha * d_grad[ei];
        return;
    }

    if (method == "Eulerian") {
        parallel_for_range(ne, [&](size_t ei) {
            const MNd Fi = F[ei];
            const MNd dFi = d_grad[ei];
            const MNd &Finvi = F_inv[ei];

            const MNd DFinv = dFi * Finvi;
            const MNd R_tilde_zero = 0.5 * alpha * (DFinv - DFinv.transpose());
            const MNd S_tilde_zero = 0.5 * alpha * (DFinv + DFinv.transpose());

            MNd S_extra = S_tilde_zero;
            S_extra(0, 0) += 1.0;
            S_extra(1, 1) += 1.0;

            const Real theta = R_tilde_zero(1, 0);
            const Real c = std::cos(theta), s = std::sin(theta);
            MNd R_extra;
            R_extra << c, -s,
                       s,  c;

            const MNd F_tilde = R_extra * S_extra;
            F_extra[ei] = F_tilde * Fi;
        });
        return;
    }
    
    if (method == "Lagrangian")
        throw std::runtime_error("extrapolateDeformGrad: method 'Lagrangian' is not implemented in the C++ path yet.");

    throw std::runtime_error("extrapolateDeformGrad: method '" + method + "' not implemented.");
}

// C++ counterpart of python/Stretch2Relax/extra_utils.py:getUVnewSolvePoission.
// `F_extra` stores one 2x2 matrix per element; each matrix row is the per-element
// gradient used to build a Poisson RHS for u and v respectively.
template<class Mesh>
UVMat getUVnewSolvePoisson(const Mesh &m,
                            const std::vector<MNd> &F_extra,
                            const CholeskyFactorizerBase &LFactorizer,
                            std::optional<size_t> fixedVind = std::nullopt,
                            std::optional<V2d> fixedUV = std::nullopt)
{
    BENCHMARK_START_TIMER_SECTION("getUVnewSolvePoisson");
    BENCHMARK_START_TIMER_SECTION("Static Assert Checks");
    static_assert(Mesh::EmbeddingDimension == 2,
                  "getUVnewSolvePoission currently expects a 2D embedding.");

    if (F_extra.size() != m.numElements())
        throw std::runtime_error("getUVnewSolvePoission: F_extra must have one 2x2 matrix per mesh element.");
    BENCHMARK_STOP_TIMER_SECTION("Static Assert Checks");

    MXNd F_extra_u(m.numElements(), 2), F_extra_v(m.numElements(), 2);
    BENCHMARK_START_TIMER_SECTION("F_extra_u and F_extra_v extraction");
    parallel_for_range(F_extra.size(), [&](size_t ei) {
        const MNd &Fe = F_extra[ei];
        if ((Fe.rows() != 2) || (Fe.cols() != 2))
            throw std::runtime_error("getUVnewSolvePoission: each F_extra[ei] must be a 2x2 matrix.");
        F_extra_u.row(ei) = Fe.row(0);
        F_extra_v.row(ei) = Fe.row(1);
    });
    BENCHMARK_STOP_TIMER_SECTION("F_extra_u and F_extra_v extraction");

    BENCHMARK_START_TIMER_SECTION("Poisson RHS extraction");
    VXd rhs_u = poisson_gradient_integration::rhs(m, F_extra_u);
    VXd rhs_v = poisson_gradient_integration::rhs(m, F_extra_v);
    BENCHMARK_STOP_TIMER_SECTION("Poisson RHS extraction");

    BENCHMARK_START_TIMER_SECTION("Poisson Solver");    
    VXd u_sol = LFactorizer.solve(rhs_u);
    VXd v_sol = LFactorizer.solve(rhs_v);
    BENCHMARK_STOP_TIMER_SECTION("Poisson Solver");

    BENCHMARK_START_TIMER_SECTION("UV Matrix Construction");
    UVMat uv_new(u_sol.size(), 2);
    uv_new.col(0) = u_sol;
    uv_new.col(1) = v_sol;
    BENCHMARK_STOP_TIMER_SECTION("UV Matrix Construction");

    BENCHMARK_START_TIMER_SECTION("Fixed Vertex Shift");
    if (fixedVind.has_value()) {
        if (!fixedUV.has_value())
            throw std::runtime_error("getUVnewSolvePoission: fixedUV must be provided when fixedVind is specified.");
        if (*fixedVind >= static_cast<size_t>(uv_new.rows()))
            throw std::runtime_error("getUVnewSolvePoission: fixedVind is out of range.");

        const V2d shift = *fixedUV - uv_new.row(*fixedVind).transpose();
        uv_new.rowwise() += shift.transpose();
    }
    BENCHMARK_STOP_TIMER_SECTION("Fixed Vertex Shift");
    BENCHMARK_STOP_TIMER_SECTION("getUVnewSolvePoisson");
    return uv_new;
}

template<typename Real>
struct Extrapolator {
public:
    
    static_assert(std::is_floating_point<Real>::value,
                  "Extrapolator Scalar must be a floating-point type.");
    
    virtual ~Extrapolator() = default;

    // Precompute and cache quantities for extrapolating away from x0 along direction `d`
    virtual void linesearch_begin(const VXd &x0, const VXd &d) = 0;

    // Extrapolate the current base point: `x0 + alpha d`, must return a UVMat
    virtual UVMat linesearch_eval(Real alpha) const = 0;

    // Batch-evaluate multiple alphas after linesearch_begin
    virtual std::vector<UVMat> eval_uvs(const std::vector<Real> &alphas) const {
        std::vector<UVMat> uvs_out;
        uvs_out.reserve(alphas.size());
        for (Real a : alphas)  uvs_out.emplace_back(linesearch_eval(a));
        return uvs_out;
    }

    // Convenience wrapper matching the Python convention where coeffs[0] is the direction.
    virtual std::vector<UVMat>
    operator()(const VXd &x0,
               const std::vector<VXd> &coeffs,
               const std::vector<Real> &alphas)
    {
        if (coeffs.empty())
            throw std::runtime_error("Extrapolator: coeffs is empty (expected coeffs[0] as direction).");
        linesearch_begin(x0, coeffs[0]);
        return eval_uvs(alphas);
    }

    // -------------------------------------------------------------------------
    // Tiny adapters: flatten/unflatten between N×2 UV and 2N vector.
    //
    // Convention: row-interleaved layout:
    //     x = [u0, v0, u1, v1, ..., u_{N-1}, v_{N-1}]^T
    // This is exactly the in-memory order of UVMat (RowMajor).
    // -------------------------------------------------------------------------

    // Zero-copy read-only view: UV(Nx2) -> VXd view (2N)
    static VecMapConst flatten_view(const UVMat &UV) {
        return VecMapConst(UV.data(), UV.size());
    }
    // Zero-copy writable view: UV(Nx2) -> VXd view (2N)
    static VecMap flatten_view(UVMat &UV) {
        return VecMap(UV.data(), UV.size());
    }
    static VXd flatten(const UVMat &UV) {
        VXd x(UV.size());
        x = flatten_view(UV);
        return x;
    }

    // unflatten: VXd x (2N) --> UV (Nx2)
    static UVMatMapConst unflatten_view(const VXd &x){
        if (x.size() % 2 != 0) throw std::runtime_error("unflatten_view: x.size() must be even (2N).");
        const size_t N = x.size() / 2;
        return UVMatMapConst(x.data(), N, 2);
    }
    static UVMatMap unflatten_view(VXd &x){
        if (x.size() % 2 != 0) throw std::runtime_error("unflatten_view: x.size() must be even (2N).");
        const size_t N = x.size() / 2;
        return UVMatMap(x.data(), N, 2);
    }
    static UVMat unflatten(const VXd &x){
        UVMat UV = unflatten_view(x); // Copy from Map
        return UV;
    }


protected:
    Extrapolator() = default;

};

// Simple and Straightforward LinearExtrapolator
////    linesearch_eval(alpha) = x0 + alpha * d
template<typename Real>
struct LinearExtrapolator : public Extrapolator<Real> {
public:
    using Base = Extrapolator<Real>;

    void linesearch_begin(const VXd &x0, const VXd &d) override {
        m_x0 = x0;
        m_d = d;
        m_lsBegin = true;
    }

    UVMat linesearch_eval(Real alpha) const override {
        if (!m_lsBegin)
            throw std::runtime_error("LinearExtrapolator: linesearch_begin must be called before linesearch_eval.");
        VXd x = m_x0 + alpha * m_d;
        return Base::unflatten(x);
    }

private:
    VXd m_x0, m_d;
    bool m_lsBegin = false;

};

template<typename Real, class Mesh>
struct RSNewtonFlowExtrapolator : public Extrapolator<Real> {
public:
    using Base = Extrapolator<Real>;

    RSNewtonFlowExtrapolator(const Mesh &m, const std::string &method = "Eulerian")
        : m_mesh(m), m_method(method), m_Linv(getLaplacianFactorizer(m, {0})) { }

    void linesearch_begin(const VXd &x0, const VXd &d) override {
        if (x0.size() != d.size())
            throw std::runtime_error("RSNewtonFlowExtrapolator: x0 and d must have the same size.");
        if (x0.size() % 2 != 0)
            throw std::runtime_error("RSNewtonFlowExtrapolator: x0 and d must have even size (2N).");

        const size_t ne = m_mesh.numElements();
        m_F.resize(ne);
        m_Finv.resize(ne);
        m_F_ex.resize(ne);

        // can be parallelized
        m_d_grad.resize(ne);
        parallel_for_range(ne, [&](size_t ei) {
            m_F[ei] = elementJacobian(ei, x0);
            m_Finv[ei] = m_F[ei].inverse();
            m_d_grad[ei] = elementJacobian(ei, d);
        });

        const auto x0_uv = Base::unflatten_view(x0);
        m_c0 = x0_uv.colwise().mean().transpose();

        // const auto d_uv = Base::unflatten_view(d);
        // const VXd d_u = d_uv.col(0);
        // const VXd d_v = d_uv.col(1);
        // const MXNd u_grad = scalarGradient(mesh(), d_u);
        // const MXNd v_grad = scalarGradient(mesh(), d_v);

        // m_d_grad.resize(ne);
        // for (size_t ei = 0; ei < ne; ++ei) {
        //     MNd dF;
        //     dF.row(0) = u_grad.row(ei);
        //     dF.row(1) = v_grad.row(ei);
        //     m_d_grad[ei] = dF;
        // }

        m_lsBegin = true;
    }

    UVMat linesearch_eval(Real alpha) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("linesearch_eval_call_in_Cpp");
        if (!m_lsBegin)
            throw std::runtime_error("RSNewtonFlowExtrapolator: linesearch_begin must be called before linesearch_eval.");

        extrapolateDeformGrad(m_F, alpha, m_d_grad, m_F_ex, m_Finv, m_method);
        
        UVMat uv_ex = getUVnewSolvePoisson(mesh(), m_F_ex, *m_Linv);

        const V2d shift = m_c0 - uv_ex.colwise().mean().transpose();
        uv_ex.rowwise() += shift.transpose();

        return uv_ex;
    }

    const Mesh &mesh() const { return m_mesh; }

    // Compute the Jacobian of a nodal vector field `x` at the center of element `ei`.
    MNd elementJacobian(size_t ei, const VXd &x) const {
        const auto &m = mesh();
        const auto &e = m.element(ei);
        // Note: the following assumes elements are piecewise linear
        MNd result = MNd::Zero();
        auto gphis = e->gradBarycentric();
        for (auto v : e.vertices())
            result += x.template segment<N>(N * v.index()) * gphis.col(v.localIndex()).transpose();
        return result;
    }

private:

    static MXNd scalarGradient(const Mesh &mesh, const VXd &scalarField) {
        if (size_t(scalarField.size()) != mesh.numNodes())
            throw std::runtime_error("RSNewtonFlowExtrapolator: scalarField size mismatch in gradient assembly.");

        MXNd g(mesh.numElements(), Mesh::EmbeddingDimension);
        g.setZero();
        for (const auto e : mesh.elements()) {
            for (const auto n : e.nodes()) {
                g.row(e.index()) += scalarField[n.index()] * e->gradPhi(n.localIndex()).average();
            }
        }
        return g;
    }
    const Mesh &m_mesh;
    std::string m_method;
    std::unique_ptr<CholeskyFactorizerBase> m_Linv;

    std::vector<MNd> m_F, m_Finv, m_d_grad;
    mutable std::vector<MNd> m_F_ex;
    V2d m_c0 = V2d::Zero();
    bool m_lsBegin = false;
};




} // namespace rotation_strain_extrapolation


#endif /* ROTATIONSTRAINEXTRAPOLATION_HH */