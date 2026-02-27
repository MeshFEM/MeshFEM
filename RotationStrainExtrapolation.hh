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
#include <MeshFEM/Types.hh>
#include "NewtonFlow.hh"
#include "PoissonGradientIntegration.hh"

#include <Eigen/Dense>
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