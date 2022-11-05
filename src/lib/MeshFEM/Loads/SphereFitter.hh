////////////////////////////////////////////////////////////////////////////////
// SphereFitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Fits the boundary surface to a sphere of radius `r`.
*/
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Created:  11/04/2022 12:54:38
////////////////////////////////////////////////////////////////////////////////
#ifndef SPHEREFITTER_HH
#define SPHEREFITTER_HH

#include "Load.hh"
#include <MeshFEM/Simplex.hh>

namespace Loads {

// ∫_∂Ω 0.5 (||x||^2 - r_tgt^2)^2 dA
template<class Object>
struct SphereFitter : public ObjectSpecificLoad<Object> {
        using Real = typename Object::Real;
        using Base = ObjectSpecificLoad<Object>;
        using ST   = typename Base::SP::StorageType;
        using VXd  = typename Object::VXd;

        static constexpr size_t N   = Object::N;
        static constexpr size_t K   = Object::K;
        static constexpr size_t Deg = Object::Deg;
        using VNd  = Eigen::Matrix<Real, N, 1>;
        using MNd  = Eigen::Matrix<Real, N, N>;
        // WARNING: Degree 3 rule uses negative weights, which can lead to a negative energy...
        static constexpr size_t QuadratureDegree = 4;

        static constexpr size_t numBdryElemNodes = Simplex::numNodes(K - 1, Deg);
        using PerBdryElementGradient = Eigen::Matrix<Real, N * numBdryElemNodes, 1>;
        using PerBdryElementHessian  = Eigen::Matrix<Real, N * numBdryElemNodes, N * numBdryElemNodes>;

        SphereFitter(const ST &obj, Real r_tgt = 1, Real stiffness = 1) : Base(obj), r_tgt(r_tgt), stiffness(stiffness) { }

        size_t numVars() const { return Base::getObj().numVars(); }

        virtual Real energy() const override {
            const auto &o = Base::getObj();
            Real result = o.template surfaceIntegral<QuadratureDegree>([&](auto be, const EvalPt<K - 1> &x) {
                return std::pow((o.deformedBoundaryPosition(be.index(), x).squaredNorm() - r_tgt * r_tgt), 2);
            });
            return 0.25 * stiffness * result;
        }

        // Gradient with respect to the deformed state
        virtual VXd grad_x() const override {
            const auto &o = Base::getObj();
            const auto &m = o.mesh();

            auto accumulate_per_element_contrib = [&](size_t bei, VXd &g_out) {
                PerBdryElementGradient contrib =
                    o.template surfaceElementIntegral<QuadratureDegree>([&](auto be, const EvalPt<K - 1> &x) {
                            PerBdryElementGradient integrand;
                            VNd p = o.deformedBoundaryPosition(be.index(), x);
                            Real deviation = (p.squaredNorm() - r_tgt * r_tgt);
                            VNd dU_dp = deviation * p;
                            integrand = dU_dp.template replicate<be.numNodes(), 1>();
                            for (auto bn : be.nodes()) {
                                Real phi = shapeFunction<Deg, K - 1>(bn.localIndex(), x);
                                integrand.template segment<N>(N * bn.localIndex()) *= phi;
                            }
                            return integrand;
                        }, bei);
                for (auto bn : m.boundaryElement(bei).nodes()) {
                    const size_t ni = bn.volumeNode().index();
                    g_out.template segment<N>(N * ni) += contrib.template segment<N>(N * bn.localIndex());
                }
            };

            VXd result;
            result.setZero(numVars());
            assemble_parallel(accumulate_per_element_contrib, result, m.numBoundaryElements());

            result *= stiffness;

            return result;
        }

        // Gradient with respect to the rest state
        virtual VXd grad_X() const override {
            return VXd::Zero(numVars());
        }

        // Hessian with respect to the deformed state H_xx
        virtual void hessian(SuiteSparseMatrix &H, bool /* projectionMask */ = true) const override {
            const auto &o = Base::getObj();
            const auto &m = o.mesh();

            auto accumulate_per_element_contrib = [&](size_t bei, auto &Hout) {
                PerBdryElementHessian contrib = o.template surfaceElementIntegral<QuadratureDegree>([&](auto be, const EvalPt<K - 1> &x) {
                        PerBdryElementHessian integrand;
                        VNd p = o.deformedBoundaryPosition(be.index(), x);
                        Real deviation = (p.squaredNorm() - r_tgt * r_tgt);
                        MNd d2U_dp2 = ((2 * p) * p.transpose() + deviation * MNd::Identity());
                        integrand = d2U_dp2.template replicate<be.numNodes(), be.numNodes()>();
                        for (auto bn : be.nodes()) {
                            Real phi = shapeFunction<Deg, K - 1>(bn.localIndex(), x);
                            integrand.template middleCols<N>(N * bn.localIndex()) *= phi;
                            integrand.template middleRows<N>(N * bn.localIndex()) *= phi;
                        }
                        return integrand;
                    }, bei);

                contrib *= stiffness;

                for (auto bn_b : m.boundaryElement(bei).nodes()) {
                    const size_t ni_b = bn_b.volumeNode().index();
                    for (auto bn_a : m.boundaryElement(bei).nodes()) {
                        const size_t ni_a = bn_a.volumeNode().index();
                        if (ni_a > ni_b) continue; // upper triangle only
                        Hout.addNZBlock(N * ni_a, N * ni_b, contrib.template block<N, N>(N * bn_a.localIndex(),
                                                                                         N * bn_b.localIndex()));
                    }
                }
            };

            assemble_parallel(accumulate_per_element_contrib, H, m.numBoundaryElements());
        }

        // *Additional* nonzeros contributed by this load to the potential energy Hessian.
        // (There are none).
        virtual SuiteSparseMatrix hessianSparsityPattern(Real /* val */ = 0.0) const override {
            const size_t nv = numVars();
            TripletMatrix<> Hsp(nv, nv);
            Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
            return SuiteSparseMatrix(Hsp);
        }

        Real r_tgt = 1.0;
        Real stiffness = 1.0;
    };
}

#endif /* end of include guard: SPHEREFITTER_HH */
