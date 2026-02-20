////////////////////////////////////////////////////////////////////////////////
// BodyForce.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A force per unit volume (3D) or area (2D) applied to the interior of an
//  elastic object. This is specified as a body force density `f` expressed
//  as a **nodal vector** field with nodal values f_j.
//
//  potential: V(x) = -int_X x . f(X) dX = - int_X (x_i phi_i) . (f_j phi_j) dX
//                  = -x . l where l_i = (M_ij f_j).
//  gradient:  g_i  = -l_i
//  Hessian:   0
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/14/2025 21:31:17
*///////////////////////////////////////////////////////////////////////////////
#ifndef BODYFORCE_HH
#define BODYFORCE_HH

#include "Load.hh"
#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/MassMatrix.hh>

namespace Loads {
    namespace detail {
        template<class Object>
        struct BodyForceLoadVector {
            static constexpr size_t N   = Object::N;
            static constexpr size_t K   = Object::K;
            static constexpr size_t Deg = Object::Deg;

            using VXd  = typename Object::VXd;
            using Real = typename VXd::Scalar;
            using VNd  = VecN_T<Real, N>;

            using Mesh = typename Object::Mesh;
            static VXd compute(const Object &o, const VXd &f) {
                BENCHMARK_SCOPED_TIMER_SECTION timer("BodyForceLoadVector.compute");
                // Note that load is applied only to the nodal position variables!
                // For elastic sheets, this means that we are assuming the load
                // is applied to the midsurface and ignore any bending moments
                // it induces (e.g., not considering how bending of a triangle
                // would affect the load potential).

                if constexpr (K > 1) { // Fast-path for elastic solids and sheets (matrix-free)
                    VXd result = VXd::Zero(o.numVars());
                    const auto &m = o.mesh();
                    // Implement matrix-vector product with the mass matrix.
                    auto M_e = MassMatrix::Impl<Deg>::template elementMatrix<Mesh>();
                    for (size_t ei = 0; ei < m.numElements(); ++ei) {
                        const Real vol = o.element3DVolume(ei);
                        auto enodes = m.elementNodeIndices(ei);
                        for (size_t lni = 0; lni < enodes.size(); ++lni) {
                            VNd M_e_f = VNd::Zero();
                            for (size_t lnj = 0; lnj < enodes.size(); ++lnj)
                                M_e_f += M_e(lni, lnj) * f.template segment<N>(N * enodes[lnj]);
                            result.template segment<N>(N * enodes[lni]) += -vol * M_e_f;
                        }
                    }
                    return result;
                }
                else {
                    // Apply load only to the nodal position variables, not other
                    // variable types like angles, which we assume come after.
                    VXd f_pad(o.numVars());
                    f_pad.head(f.size()) = f;

                    // Don't apply any load to the angular degrees of freedom
                    // (we make the assumption that bending moments applied
                    // by the load are negligible).
                    f_pad.tail(o.numVars() - f.size()).setZero();

                    auto M = o.massMatrix(/* updatedParametrization */ false);
                    return -M.apply(f_pad);
                }
            }

            static VXd contract_d2E_dXdx(const Object &o, const VXd &f, const VXd &dx) {
                VXd result;
                result.setZero(o.numRestVars());

                if constexpr (K > 1) {
                    const auto &m = o.mesh();
                    // Implement matrix-vector product with the mass matrix.
                    auto M_e = MassMatrix::Impl<Deg>::template elementMatrix<Mesh>();
                    for (size_t ei = 0; ei < m.numElements(); ++ei) {
                        auto enodes = m.elementNodeIndices(ei);
                        Real d_dvol = 0;
                        for (size_t lni = 0; lni < enodes.size(); ++lni) {
                            VNd M_e_f = VNd::Zero();
                            for (size_t lnj = 0; lnj < enodes.size(); ++lnj)
                                M_e_f += M_e(lni, lnj) * f.template segment<N>(N * enodes[lnj]);
                            d_dvol += M_e_f.dot(dx.template segment<N>(N * enodes[lni]));
                        }

                        // Accumulate d_dvol * dvol / dX
                        auto everts = m.elementVertexIndices(ei);
                        const Real vol = o.element3DVolume(ei);
                        for (size_t lvi = 0; lvi < everts.size(); ++lvi) {
                            if constexpr (K == N)
                                result.template segment<N>(N * everts[lvi]) -= d_dvol * vol * m.elementData(ei).gradBarycentric().col(lvi);
                            else {
                                static_assert(K == 2 && N == 3, "Expected elastic membrane");
                                result.template segment<N>(N * everts[lvi]) -= d_dvol * vol * o.getB(ei) * o.getBtGradBarycentric(ei).col(lvi);
                            }
                        }
                    }
                }
                else throw std::runtime_error("TODO: other codimensional cases");

                return result;
            }
        };
    }

template<class Object>
struct BodyForce : public ObjectSpecificLoad<Object> {
    using Base = ObjectSpecificLoad<Object>;
    using EO   = Object;
    using ST   = typename Base::EOStorageType;

    using Real = typename Base::Real;
    using VXd  = typename Base::VXd;
    using MXd  = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // Row-major for proper flattened order
    using Base::getObj;

    BodyForce(const ST &obj) : Base(obj) { m_updateCache(); }
    BodyForce(const ST &obj, const MXd &f) : Base(obj) { setNodalForceDensity(f); }

    virtual Real energy() const override { return m_grad.dot(getObj().getVars()); }

    // Derivative with respect to deformed configuration
    virtual VXd grad_x() const override { return m_grad; }

    // Derivative with respect to rest configuration (for shape optimization)
    virtual VXd grad_X() const override { throw std::runtime_error("Unimplemented"); }

    // Potential is linear with respect to the deformed state
    virtual void accumulateHessian(Real weight, NewtonHessian & /* H */, bool /* projectionMask */ = true) const override { }
    virtual NewtonHessian hessianSparsityPattern() const override { return NewtonHessian(); }

    virtual VXd contract_d2E_dXdx(const VXd &dx) const override {
        Eigen::Map<const VXd> f(m_nodalForceDensity.data(), m_nodalForceDensity.size());
        return detail::BodyForceLoadVector<Object>::contract_d2E_dXdx(getObj(), f, dx);
    }

    void setNodalForceDensity(MXd f /* pass-by-value due to copy inside */) {
        const auto &o = getObj();
        size_t nn = o.numNodes();
        if ((f.size() != 2 * nn) && (f.size() != 3 * nn)) throw std::runtime_error("Invalid size of nodal force density: should be a 2D or 3D vector per node");
        if ((f.cols() != 1) && (f.rows() != nn))          throw std::runtime_error("Invalid shape of nodal force density: should be a column vector or an (numNodes x N) matrix");
        m_nodalForceDensity = f;
        m_updateCache();
    }
    const MXd &getNodalForceDensity() const { return m_nodalForceDensity; }

    virtual ~BodyForce() { }
private:
    MXd m_nodalForceDensity;
    VXd m_grad;

    virtual void m_stateUpdated(typename Base::VM vmask) override {
        if (vmask == Base::VM::Rest) m_updateCache();
    }

    void m_updateCache() {
        m_grad.setZero(getObj().numVars());
        if (m_nodalForceDensity.size() == 0) return; // Empty force density interpreted as zero force.

        Eigen::Map<const VXd> f(m_nodalForceDensity.data(), m_nodalForceDensity.size());
        m_grad = detail::BodyForceLoadVector<Object>::compute(getObj(), f);
    }
};

}

#endif /* end of include guard: BODYFORCE_HH */
