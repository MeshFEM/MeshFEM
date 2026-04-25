////////////////////////////////////////////////////////////////////////////////
// RegionNetForce.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Applies a total force vector `f` that is distributed over a portion of an
//  elastic object. A fraction of this force is applied to each selected
//  **node**, defined by a binary-valued nodal indicator χ_j. The result is a
//  constant force density over the selected region (i.e., constant over
//  elements whose nodes are all selected) and a smooth falloff over the
//  elements adjacent to the region as defined by the FEM shape functions.
//
//  Formally, we define the force density to be:
//    f_density(X) = ⍺ f χ_j phi_j(X),
//  where ⍺ is a normalization constant chosen so that the total
//  load applied (i.e., the sum of equivalent point loads at the nodes)
//  is equal to the original `f`. This normalization constant is:
//      ⍺ = 1 / int_Ω χ_j phi_j(X) dX
//
//  We therefore can implement this load as a `BodyForce` with
//      f_density = (X) = f (χ_i phi_i(X)) / (int_Ω χ_j phi_j(X) dX).
//
//  The normalization must be accounted for during sensitivity analysis, but
//  otherwise the `BodyForce` class takes care of everything.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/09/2026 11:08:54
*///////////////////////////////////////////////////////////////////////////////
#ifndef REGIONNETFORCE_HH
#define REGIONNETFORCE_HH

#include "BodyForce.hh"

namespace Loads {
    template<class Object>
    struct RegionNetForce : public BodyForce<Object> {
        using Real = typename Object::Real;
        using Base = BodyForce<Object>;
        using ST   = typename Base::EOStorageType;
        static constexpr size_t N = Object::N;
        using VXd  = typename Object::VXd;
        using VNd  = Eigen::Matrix<Real, N, 1>; // ElasticSolid has the information of N

        RegionNetForce(const ST &obj) : Base(obj) {
            m_indicatorField.setOnes(this->getObj().numNodes());
        }

        void set_f(const VNd &f) { m_f = f; m_updateCache(); }
        const VNd &get_f() const { return m_f; }

        void set_indicator_field(const VXd &ifield) {
            if (ifield.size() != this->getObj().numNodes())
                throw std::runtime_error("Invalid size of indicator field");
            m_indicatorField = ifield;
            m_updateCache();
        }

        const VXd &get_indicator_field() const { return m_indicatorField; }

        VXd contract_d2E_dXdx(const VXd &dx) const override {
            VXd result = Base::contract_d2E_dXdx(dx);

            //      dE/dx = dE/dx|_{alpha = 1} * alpha
            // ==>  d2E/dxdX = d2E/dxX|_{alpha = 1} * alpha + dE/dx|_{alpha = 1} * d alpha/dX
            // `result` currently contains the first term of this sum but not the second.
            // (`BodyForce` treats `alpha = 1 / m_normalizationFactor` as constant.)
            const auto &o = this->getObj();
            VXd d_nf_dX = VXd::Zero(o.numRestVars());

            auto integratedPhis = integratedShapeFunctions<Object::Deg, Object::K>();
            const auto &m = o.mesh();
            for (size_t ei = 0; ei < m.numElements(); ++ei) {
                auto enodes = m.elementNodeIndices(ei);
                Real d_nf_dvol = 0;
                for (size_t lni = 0; lni < enodes.size(); ++lni)
                    d_nf_dvol += m_indicatorField[enodes[lni]] * integratedPhis[lni];

                if (d_nf_dvol == 0) continue;
                // Accumulate d_nf_dvol * dvol / dX
                auto everts = m.elementVertexIndices(ei);
                const Real vol = o.element3DVolume(ei);
                for (size_t lvi = 0; lvi < everts.size(); ++lvi) {
                    if constexpr (Object::K == N)
                        d_nf_dX.template segment<N>(N * everts[lvi]) += d_nf_dvol * vol * m.elementData(ei).gradBarycentric().col(lvi);
                    else {
                        static_assert(Object::K == 2 && N == 3, "Expected elastic membrane");
                        d_nf_dX.template segment<N>(N * everts[lvi]) += d_nf_dvol * vol * o.getB(ei) * o.getBtGradBarycentric(ei).col(lvi);
                    }
                }
            }

            result += Base::grad_x().dot(dx) * (-1 / m_normalizationFactor) * d_nf_dX;

            return result;
        }

    private:
        VNd m_f = VNd::Zero(); // Net force vector
        VXd m_indicatorField; // Indicator field for the nodes across which force is distributed.
        Real m_normalizationFactor;

        void m_updateCache() {
            const auto &o = this->getObj();
            m_normalizationFactor = 0;
            auto integratedPhis = integratedShapeFunctions<Object::Deg, Object::K>();
            const auto &m = o.mesh();
            for (size_t ei = 0; ei < m.numElements(); ++ei) {
                Real vol = o.element3DVolume(ei);
                auto enodes = m.elementNodeIndices(ei);
                for (size_t lni = 0; lni < enodes.size(); ++lni)
                    m_normalizationFactor += m_indicatorField[enodes[lni]] * vol * integratedPhis[lni];
            }
            VNd f_density = m_f / m_normalizationFactor;

            VXd f = VXd::Zero(o.numNodes() * N);
            for (size_t ni = 0; ni < o.numNodes(); ++ni) {
                if (m_indicatorField[ni] == 0.0) continue;
                f.template segment<N>(N * ni) = f_density * m_indicatorField[ni];
            }

            Base::setNodalForceDensity(f);
        }

        void m_stateUpdated(typename Base::VM vmask) override {
            if (vmask == Base::VM::Rest) m_updateCache();
        }
    };

} // namespace Loads

#endif /* end of include guard: REGIONNETFORCE_HH */
