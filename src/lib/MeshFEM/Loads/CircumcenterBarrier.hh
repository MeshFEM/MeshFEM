////////////////////////////////////////////////////////////////////////////////
// CircumcenterBarrier.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Element shape optimization for constraining the position of the
//  circumcenter.
*/
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Created:  11/07/2022 11:56:01
////////////////////////////////////////////////////////////////////////////////
#ifndef CIRCUMCENTERBARRIER_HH
#define CIRCUMCENTERBARRIER_HH

#include "Load.hh"
#include "ConstraintBarrier.hh"
#include <MeshFEM/Simplex.hh>
#include <MeshFEM/Geometry.hh>

namespace Loads {

template<class Object>
struct CircumcenterBarrier : public ObjectSpecificLoad<Object> {
        using Real = typename Object::Real;
        using Base = ObjectSpecificLoad<Object>;
        using ST   = typename Base::SP::StorageType;
        using VXd  = typename Object::VXd;

        static constexpr size_t N   = Object::N;
        static constexpr size_t K   = Object::K;
        static constexpr size_t Deg = Object::Deg;
        using VKd  = Eigen::Matrix<Real, K, 1>;
        using MKd  = Eigen::Matrix<Real, K, K>;
        using Barycooords = Eigen::Matrix<Real, K + 1, 1>;

        CircumcenterBarrier(const ST &obj, Real bc_min) : Base(obj), bc_min(bc_min) {
            if (Deg != 1) throw std::runtime_error("CircumcenterBarrier is only intended for linear meshes");
            if (  K != N) throw std::runtime_error("CircumcenterBarrier is not intended for co-dimensional objects");

            barrier.barrierThreshold = bc_min; // Put infinite barrier at constraint violation magnitude of `bc_min` (which will infinitely resist bar becoming negative)
        }

        size_t numVars() const { return Base::getObj().numVars(); }

        MKd getU(size_t ei) const {
            const auto &o = Base::getObj();
            const auto &m = o.mesh();
            auto e = m.element(ei);
            const auto &x = o.deformedPositions();
            MKd U;
            for (size_t i = 0; i < K; ++i) {
                U.col(i) = (x.row(e.vertex(i + 1).index())
                          - x.row(e.vertex(    0).index())).transpose();
            }
            return U;
        }

        Barycooords circumcenter(size_t ei) const {
            MKd U = getU(ei);
            MKd A = U.transpose() * U;
            Barycooords result;
            result.template tail<K>() = 0.5 * A.llt().solve(A.diagonal());
            result[0] = 1.0 - result.template tail<K>().sum();
            return result;
        }

        virtual Real energy() const override {
            const auto &o = Base::getObj();
            const auto &m = o.mesh();
            Real result = 0.0;
            for (auto e : m.elements()) {
                Barycooords bc = circumcenter(e.index());
                // result += bc.squaredNorm(); // Simple function for debugging

                // Impose a constraint on each barycentric coordinate.
                // bc[i] >= bc_min <==>  bc_min - bc[i] <= 0
                for (auto v : e.vertices())
                    result += barrier.b(bc_min - bc[v.localIndex()]);
            }
            return result;
        }

        // Gradient with respect to the deformed state
        virtual VXd grad_x() const override {
            VXd result;
            result.setZero(numVars());

            for (auto e : Base::getObj().mesh().elements()) {
                Barycooords bc = circumcenter(e.index());

                MKd U = getU(e.index());

                // Barycooords dJ_dbc = 2 * bc; // Simple function for debugging
                Barycooords dJ_dbc;
                for (auto v : e.vertices())
                    dJ_dbc[v.localIndex()] = -barrier.db(bc_min - bc[v.localIndex()]);

                // Solve adjoint equation
                VKd s = (U.transpose() * U).llt().solve((dJ_dbc.template tail<K>().array() - dJ_dbc[0]).matrix());

                for (size_t i = 0; i < K; ++i) { // loop over contributions from d/du_i
                    VKd dJ_dui = s[i] * U.col(i) - U * (s[i] * bc.template tail<K>() + bc[i + 1] * s);
                    result.template segment<N>(N * e.vertex(i + 1).index()) += dJ_dui;
                    result.template segment<N>(N * e.vertex(    0).index()) -= dJ_dui;
                }
            }

            return result;
        }

        // Gradient with respect to the rest state
        virtual VXd grad_X() const override {
            return VXd::Zero(numVars());
        }

        // Hessian with respect to the deformed state H_xx
        virtual void hessian(SuiteSparseMatrix &H, bool /* projectionMask */ = true) const override {
            // Add nonzeros to H's **upper triangle** only.
            auto addUpperTriStrip = [&](int i, int j, const VKd &v) {
                if (i > j) return; // skip lower triangle
                H.addNZStrip(i, j, v.head(std::min<size_t>(j - i + 1, v.size()))); // add only the portion of `v` in the upper triangle.
            };
            for (auto e : Base::getObj().mesh().elements()) {
                Barycooords bc = circumcenter(e.index());

                MKd U = getU(e.index());

                // Gradient and Hessian of the objective with respect to the barycentric coordinates
                Barycooords dJ_dbc;
                Eigen::Matrix<Real, K + 1, K + 1> d2J_dbc2;
                // dJ_dbc = 2 * bc; // Simple function for debugging
                // d2J_dbc2 = 2 * Eigen::Matrix<Real, K + 1, K + 1>::Identity(); // Simple function for debugging

                d2J_dbc2.setZero();
                for (size_t i = 0; i < K + 1; ++i) {
                    dJ_dbc[i] = -barrier.db(bc_min - bc[i]);
                    d2J_dbc2(i, i) = barrier.d2b(bc_min - bc[i]);
                }

                MKd A = U.transpose() * U;
                auto A_llt = A.llt();
                // Solve adjoint equation
                VKd s = A_llt.solve((dJ_dbc.template tail<K>().array() - dJ_dbc[0]).matrix());

                // Loop over perturbations "delta u_j[c]" and calculate the change in gradient "delta g".
                // This is a rather brute-force implementation that could be simplified/accelerated.
                for (size_t j = 0; j < K; ++j) {
                    for (size_t c = 0; c < N; ++c) {
                        MKd delta_U = MKd::Zero();
                        delta_U(c, j) = 1;
                        MKd delta_A = delta_U.transpose() * U + U.transpose() * delta_U;

                        VKd delta_bc_tail = A_llt.solve(0.5 * delta_A.diagonal() - delta_A * bc.template tail<K>());

                        // Hessian of the objective with respect to the "independent" barycentric coordinates bc.tail<k>()
                        MKd d2Jtilde_dbc_tail2        = d2J_dbc2.template block<K, K>(1, 1);
                        d2Jtilde_dbc_tail2.rowwise() -= d2J_dbc2.template block<1, K>(0, 1);
                        d2Jtilde_dbc_tail2.colwise() -= d2J_dbc2.template block<K, 1>(1, 0);
                        d2Jtilde_dbc_tail2.array()   += d2J_dbc2(0, 0);

                        VKd delta_s = A_llt.solve(d2Jtilde_dbc_tail2 * delta_bc_tail - delta_A * s);
                        for (size_t i = 0; i < K; ++i) { // loop over delta g_i
                            // g_i = s[i] * U.col(i) - U * (s[i] * bc.template tail<K>() + bc[i + 1] * s);
                            VKd delta_g_i = delta_s[i] * U.col(i) + s[i] * delta_U.col(i)
                                          - delta_U * (s[i] * bc.template tail<K>() + bc[i + 1] * s)
                                          -       U * (delta_s[i] * bc.template tail<K>() + s[i] * delta_bc_tail
                                                        + delta_bc_tail[i] * s + bc[i + 1] * delta_s);
                            addUpperTriStrip(N * e.vertex(i + 1).index(), N * e.vertex(j + 1).index() + c,  delta_g_i);
                            addUpperTriStrip(N * e.vertex(    0).index(), N * e.vertex(j + 1).index() + c, -delta_g_i);
                            addUpperTriStrip(N * e.vertex(i + 1).index(), N * e.vertex(    0).index() + c, -delta_g_i);
                            addUpperTriStrip(N * e.vertex(    0).index(), N * e.vertex(    0).index() + c,  delta_g_i);
                        }
                    }
                }
            }
        }

        // *Additional* nonzeros contributed by this load to the potential energy Hessian.
        // (There are none).
        virtual SuiteSparseMatrix hessianSparsityPattern(Real /* val */ = 0.0) const override {
            const size_t nv = numVars();
            TripletMatrix<> Hsp(nv, nv);
            Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
            return SuiteSparseMatrix(Hsp);
        }

        Real bc_min = 0.1;
        RawBarrierLog barrier; // enforce a constraint of the form `c <= 0`
    };
}

#endif /* end of include guard: CIRCUMCENTERBARRIER_HH */
