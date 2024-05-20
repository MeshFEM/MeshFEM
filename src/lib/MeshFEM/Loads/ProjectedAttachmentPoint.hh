////////////////////////////////////////////////////////////////////////////////
// ProjectedAttachmentPoint.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A spring attachment point class that implements a closest-point projection
//  onto a surface or curve. This is useful for creating springs that pull
//  points toward their closest point on a target surface or curve.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  05/17/2024 17:35:44
*///////////////////////////////////////////////////////////////////////////////
#ifndef PROJECTEDATTACHMENTPOINT_HH
#define PROJECTEDATTACHMENTPOINT_HH

#include "Springs.hh"
#include <MeshFEM/ClosestPointProjection.hh>

namespace Loads {

template<typename _CoordinateType>
struct ProjectedAttachmentPoint {
    using APC = AttachmentPointCoordinate<_CoordinateType>;
    using VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using JacobianBlock = typename APC::JacobianBlock;
    using CP = ClosestPointProjection<_CoordinateType>;
    static constexpr size_t BlockSize = APC::BlockSize;

    ProjectedAttachmentPoint(const APC &attachmentPoint, std::shared_ptr<CP> closestPtProjector)
        : preprojectionAttachmentPoint(attachmentPoint),
          varIndices(preprojectionAttachmentPoint.varIndices),
          coefficients(preprojectionAttachmentPoint.coefficients),
          m_projector(closestPtProjector)
    { }

    static std::vector<ProjectedAttachmentPoint> fromDeformationSamplerMatrix(const SuiteSparseMatrix &dsm, std::shared_ptr<CP> closestPtProjector) {
        std::vector<ProjectedAttachmentPoint> result;
        for (const APC &apc : APC::fromDeformationSamplerMatrix(dsm))
            result.emplace_back(apc, closestPtProjector);
        return result;
    }

    void setVars(const Eigen::Ref<const VXd> &vars) {                                               m_update(preprojectionAttachmentPoint.getPosition(vars)); }
    void setVars(const Eigen::Ref<const VXd> &vars, const std::vector<int> &globalVarForLocalVar) { m_update(preprojectionAttachmentPoint.getPosition(vars, globalVarForLocalVar)); }

    // WARNING: the passed `vars` are ignored and instead the projected point computed
    // by the most recent `setVars` call is returned.
    _CoordinateType getPosition()                                                                                            const { return m_projectedPoint; }
    _CoordinateType getPosition(const Eigen::Ref<const VXd> &/* vars */)                                                     const { return m_projectedPoint; }
    _CoordinateType getPosition(const Eigen::Ref<const VXd> &/* vars */, const std::vector<int> &/* globalVarForLocalVar */) const { return m_projectedPoint; }

    void gradContribution(const _CoordinateType &grad_p, Eigen::Ref<VXd> grad) const { preprojectionAttachmentPoint.gradContribution(dp_dq * grad_p, grad); }

    void gradContribution(const _CoordinateType &grad_p, Eigen::Ref<VXd> grad, const std::vector<int> &globalVarForLocalVar) const {
         preprojectionAttachmentPoint.gradContribution(dp_dq * grad_p, grad, globalVarForLocalVar); 
    }

    void validate() const {
        if (coefficients.size() != varIndices.size())
            throw std::runtime_error("ProjectedAttachmentPoint: coefficients and varIndices must have the same size (" + std::to_string(coefficients.size()) + " != " + std::to_string(varIndices.size()) + ")");
        if (m_projector == nullptr)
            throw std::runtime_error("ProjectedAttachmentPoint: projector must be non-null");
    }

    // The second derivative is zero (assuming the point is not moving between
    // tangent planes). Also, it would not contribute to the Hessian of the
    // spring energy ||x - P(x)|| due to the envelope theorem...
    template<class SpMat> void accumulate_contract_d2_dvar2(const _CoordinateType &/* grad_p */, SpMat &/* H */                                                    ) const { }
    template<class SpMat> void accumulate_contract_d2_dvar2(const _CoordinateType &/* grad_p */, SpMat &/* H */, const std::vector<int> &/* globalVarForLocalVar */) const { }

    template<class Derived> static decltype(auto) extract(const Eigen::MatrixBase<Derived> &vars, size_t i) { return APC::extract(vars, i); }
    template<class Derived> static decltype(auto) extract(      Eigen::MatrixBase<Derived> &vars, size_t i) { return APC::extract(vars, i); }

    auto d_dvar(size_t vi)           const { return dp_dq * preprojectionAttachmentPoint.coefficients[vi]; }
    std::shared_ptr<CP> projector()  const { return m_projector; }
    const JacobianBlock &get_dp_dq() const { return dp_dq; }

    APC preprojectionAttachmentPoint;
    // Bound to preprojectionAttachmentPoint
    Eigen::VectorXi &varIndices;
    VXd &coefficients;
private:
    _CoordinateType m_projectedPoint;
    JacobianBlock dp_dq; // Jacobian of the projected point p with respect to the query point q
    std::shared_ptr<CP> m_projector;

    // Recompute the closest point projection and its sensitivity
    void m_update(const _CoordinateType &preprojectedPt) {
        typename CP::ProjectionResult proj = m_projector->project(preprojectedPt, /* computeJacobian = */ true);

        m_projectedPoint = proj.p;
        dp_dq            = proj.dp_dq;
    }
};

} // namespace Loads

#endif /* end of include guard: PROJECTEDATTACHMENTPOINT_HH */
