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
#include <MeshFEM/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEM/Types.hh>
#include "NewtonFlow.hh"
#include "PoissonGradientIntegration.hh"

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>

namespace rotation_strain_extrapolation {

using V2d   = Eigen::Vector2d;
using V3d   = Eigen::Vector3d;
using VXd   = Eigen::VectorXd;
using M2d   = Eigen::Matrix<Real, 2, 2>;
using MNd   = Eigen::MatrixXd;

using UVMat = Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>; // Nx2
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
    static_assert(Mesh::EmbeddingDimension == 2,
                  "getUVnewSolvePoission currently expects a 2D embedding.");

    if (F_extra.size() != m.numElements())
        throw std::runtime_error("getUVnewSolvePoission: F_extra must have one 2x2 matrix per mesh element.");

    MNd F_extra_u(m.numElements(), 2), F_extra_v(m.numElements(), 2);
    for (size_t ei = 0; ei < m.numElements(); ++ei) {
        const MNd &Fe = F_extra[ei];
        if ((Fe.rows() != 2) || (Fe.cols() != 2))
            throw std::runtime_error("getUVnewSolvePoission: each F_extra[ei] must be a 2x2 matrix.");
        F_extra_u.row(ei) = Fe.row(0);
        F_extra_v.row(ei) = Fe.row(1);
    }

    VXd rhs_u = poisson_gradient_integration::rhs(m, F_extra_u);
    VXd rhs_v = poisson_gradient_integration::rhs(m, F_extra_v);

    VXd u_sol = LFactorizer.solve(rhs_u);
    VXd v_sol = LFactorizer.solve(rhs_v);

    UVMat uv_new(u_sol.size(), 2);
    uv_new.col(0) = u_sol;
    uv_new.col(1) = v_sol;

    if (fixedVind.has_value()) {
        if (!fixedUV.has_value())
            throw std::runtime_error("getUVnewSolvePoission: fixedUV must be provided when fixedVind is specified.");
        if (*fixedVind >= static_cast<size_t>(uv_new.rows()))
            throw std::runtime_error("getUVnewSolvePoission: fixedVind is out of range.");

        const V2d shift = *fixedUV - uv_new.row(*fixedVind).transpose();
        uv_new.rowwise() += shift.transpose();
    }

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




} // namespace rotation_strain_extrapolation


#endif /* ROTATIONSTRAINEXTRAPOLATION_HH */