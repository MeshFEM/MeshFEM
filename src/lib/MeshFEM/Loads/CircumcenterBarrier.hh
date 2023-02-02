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
        using AK1i = Eigen::Array <int, K + 1, 1>;
        using MKd  = Eigen::Matrix<Real, K, K>;
        using MXNd = typename Object::MXNd;
        using Barycooords = Eigen::Matrix<Real, K + 1, 1>;

        CircumcenterBarrier(const ST &obj, Real bc_min, bool subdivisionBarrier = false) : Base(obj), bc_min(bc_min), m_subdivisionBarrier(subdivisionBarrier) {
            if (Deg != 1) throw std::runtime_error("CircumcenterBarrier is only intended for linear meshes");
            if (  K != N) throw std::runtime_error("CircumcenterBarrier is not intended for co-dimensional objects");

            // Position the infinite barrier (barrierThreshold) automatically:
            // by default we put it at `bc_min` so that the barrier becomes
            // infinite just as bc approaches 0. However this makes recovering from
            // initial configurations with `bc < 0` impossible.
            Real barrierMargin = 0.05;
            barrier.barrierThreshold = std::max(bc_min, bc_min - minCircumcenterBC() + barrierMargin);
        }

        const auto &mesh() const { return Base::getObj().mesh(); }

        size_t numVars() const { return Base::getObj().numVars(); }

        static MKd getU(const MXNd &x, const AK1i &e) {
            MKd U;
            for (size_t i = 0; i < K; ++i) {
                U.col(i) = (x.row(e[i + 1])
                          - x.row(e[    0])).transpose();
            }
            return U;
        }

        AK1i elementCorners(size_t ei) const {
            AK1i c;
            for (auto v : mesh().element(ei).vertices()) c[v.localIndex()] = v.index();
            return c;
        }

        MKd getU(size_t ei) const { return getU(Base::getObj().deformedPositions(), elementCorners(ei)); }

        // Call f(e, B, x) for each subelement `e`,
        // where `e` is the vector of corner indices,
        //       `B` is a #subv x (K + 1) matrix of barycentric coordinates of
        //           the subelement corner positions wrt the macro element
        //           corner positions, and
        //       `x` is a #subv x N matrix of subvertex positions
        template<class F>
        void foreach_subelement(size_t ei, const F& f) const {
            if (K != 3) throw std::runtime_error("Subdivided element optimization is only supported for tetrahedral meshes");

            static constexpr size_t nsubv = Simplex::numVertices(K) + Simplex::numEdges(K);
            Eigen::Matrix<Real, nsubv, K + 1> B;
            Eigen::Matrix<Real, nsubv, N>     x;

            // The first (K + 1) subdivided vertices are simply the macro element corners
            const auto &xMacro = Base::getObj().deformedPositions();
            for (auto v : mesh().element(ei).vertices())
                x.row(v.localIndex()) = xMacro.row(v.index());
            B.template topRows<K + 1>().setIdentity();

            // The rest are at the macro edge midpoints
            B.template bottomRows<Simplex::numEdges(K)>().setZero();
            VecN_T<Real, Simplex::numEdges(K)> edgeLens;
            for (size_t i = 0; i < Simplex::numEdges(K); ++i) {
                size_t uLocal = Simplex::edgeStartNode(i),
                       vLocal = Simplex::edgeEndNode(i);
                size_t u = mesh().element(ei).vertex(uLocal).index(),
                       v = mesh().element(ei).vertex(vLocal).index();
                edgeLens[i] = (xMacro.row(u) - xMacro.row(v)).norm();
                size_t out = (K + 1) + i;
                x.row(out) = 0.5 * (xMacro.row(u) + xMacro.row(v));
                B(out, uLocal) = 0.5;
                B(out, vLocal) = 0.5;
            }

            //      3
            //      *
            //     / \`e4
            //   e3  e5 `* 2
            //   / e2--\ / e1
            // 0*--e0---* 1
            constexpr size_t oppositeSubFaceEdgePts[4][3] = {
                {0, 3, 2},
                {0, 1, 5},
                {2, 4, 1},
                {3, 5, 4}
            };

            // Visit the corner subtetrahedra
            for (size_t c = 0; c < K + 1; ++c) {
                AK1i e((K + 1) + oppositeSubFaceEdgePts[c][0],
                       (K + 1) + oppositeSubFaceEdgePts[c][1],
                       (K + 1) + oppositeSubFaceEdgePts[c][2], c);
                f(e, B, x);
            }

            // The inner octahedron is tetrahedralized by inserting a center
            // edge connecting the base edge midpoints.
            // The base edges are the longest edge pair of the
            // macro element.
            constexpr size_t diagonalPairedEdge[3] = { 4, 3, 5 }; // (e0, e4), (e1, e3), (e2, e5)
            Real longestLen = 0;
            size_t longestPair = 0;
            for (size_t i = 0; i < 3; ++i) {
                Real pairLen = edgeLens[i] + edgeLens[diagonalPairedEdge[i]];
                if (pairLen > longestLen) {
                    longestLen = pairLen;
                    longestPair = i;
                }
            }

            // Visit the for subtetrahedra of the inner octahedron
            for (size_t c = 0; c < K + 1; ++c) {
                AK1i e;
                e[0] = oppositeSubFaceEdgePts[c][1]; // reversed orientation
                e[1] = oppositeSubFaceEdgePts[c][0];
                e[2] = oppositeSubFaceEdgePts[c][2];
                if ((e.template head<K>() == longestPair).any())
                    e[3] = diagonalPairedEdge[longestPair];
                else
                    e[3] = longestPair;
                e += K + 1; // all indices above refer to edge midpoints...
                f(e, B, x);
            }
        }

        // For debugging
        std::pair<Eigen::MatrixXd, Eigen::MatrixXi> subtets(size_t ei) const {
            std::pair<Eigen::MatrixXd, Eigen::MatrixXi> result;
            Eigen::MatrixXd &V = result.first;
            Eigen::MatrixXi &F = result.second;

            F.resize(8, 4);  // 8  subtetrahedra
            int Fback = 0;
            foreach_subelement(ei, [&](const AK1i &e, auto /* B */, auto x) {
                if (V.rows() == 0)
                    V = x;
                F.row(Fback++) = e;
            });
            return result;
        }

        static Barycooords circumcenter(const MXNd &x, const AK1i &e, MKd &U, Eigen::LLT<MKd> &A_llt) {
            U = getU(x, e);
            MKd A = U.transpose() * U;
            A_llt = A.llt();
            Barycooords result;
            result.template tail<K>() = 0.5 * A_llt.solve(A.diagonal());
            result[0] = 1.0 - result.template tail<K>().sum();
            return result;
        }

        static Barycooords circumcenter(const MXNd &x, const AK1i &e) {
            MKd U;
            Eigen::LLT<MKd> A_llt;
            return circumcenter(x, e, U, A_llt);
        }

        Real elementEnergy(const MXNd &x, const AK1i &e) const {
            auto bc = circumcenter(x, e);
            Real result = 0.0;
            for (size_t i = 0; i < K + 1; ++i)
                result += barrier.b(bc_min - bc[i]);
            return result;
        }

        using PerElementGradient = Eigen::Matrix<Real, N * (K + 1), 1>;
        PerElementGradient elementGradient(const MXNd &x, const AK1i &e) const {
            MKd U;
            Eigen::LLT<MKd> A_llt;
            Barycooords bc = circumcenter(x, e, U, A_llt);

            // Barycooords dJ_dbc = 2 * bc; // Simple function for debugging
            Barycooords dJ_dbc;
            for (size_t i = 0; i < K + 1; ++i)
                dJ_dbc[i] = -barrier.db(bc_min - bc[i]);

            // Solve adjoint equation
            VKd s = A_llt.solve((dJ_dbc.template tail<K>().array() - dJ_dbc[0]).matrix());

            PerElementGradient result;
            for (size_t i = 0; i < K; ++i)
                result.template segment<N>(N * (i + 1)) = s[i] * U.col(i) - U * (s[i] * bc.template tail<K>() + bc[i + 1] * s);
            result.template head<N>() = -Eigen::Map<Eigen::Matrix<Real, N, K + 1>>(result.data()).template rightCols<K>().rowwise().sum();
            return result;
        }

        using PerElementHessian = Eigen::Matrix<Real, N * (K + 1), N * (K + 1)>;
        PerElementHessian elementHessian(const MXNd &x, const AK1i &e) const {
            PerElementHessian result;
            result.setZero();
            MKd U;
            Eigen::LLT<MKd> A_llt;
            Barycooords bc = circumcenter(x, e, U, A_llt);

            // Gradient and Hessian of the objective with respect to the barycentric coordinates
            Barycooords dJ_dbc;
            Eigen::Matrix<Real, K + 1, K + 1> d2J_dbc2;

            d2J_dbc2.setZero();
            for (size_t i = 0; i < K + 1; ++i) {
                dJ_dbc[i] = -barrier.db(bc_min - bc[i]);
                d2J_dbc2(i, i) = barrier.d2b(bc_min - bc[i]);
            }

            // Solve adjoint equation
            VKd s = A_llt.solve((dJ_dbc.template tail<K>().array() - dJ_dbc[0]).matrix());

            // Loop over perturbations "delta u_j[c]" and calculate the change in gradient "delta g".
            // This is a rather brute-force implementation that could be simplified and accelerated.
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
                    for (size_t i = 0; i < K; ++i) { // loop over delta g_i (upper tri)
                        // Recall that g_i = s[i] * U.col(i) - U * (s[i] * bc.template tail<K>() + bc[i + 1] * s);
                        VKd delta_g_i = delta_s[i] * U.col(i) + s[i] * delta_U.col(i)
                                      - delta_U * (s[i] * bc.template tail<K>() + bc[i + 1] * s)
                                      -       U * (delta_s[i] * bc.template tail<K>() + s[i] * delta_bc_tail
                                                    + delta_bc_tail[i] * s + bc[i + 1] * delta_s);
                        result.template block<N, 1>(N * (i + 1), N * (j + 1) + c) += delta_g_i;
                        result.template block<N, 1>(          0, N * (j + 1) + c) -= delta_g_i;
                        result.template block<N, 1>(N * (i + 1),               c) -= delta_g_i;
                        result.template block<N, 1>(          0,               c) += delta_g_i;
                    }
                }
            }

            return result;
        }

        Barycooords circumcenter(size_t ei) const { return circumcenter(Base::getObj().deformedPositions(), elementCorners(ei)); }

        // Get the smallest barycentric coordinate of any of the elements
        // (or any of the sub-elements if `m_subdivisionBarrier` is `true`).
        Real minCircumcenterBC() const {
            Real result = safe_numeric_limits<Real>::max();
            const auto &o = Base::getObj();
            for (auto e : mesh().elements()) {
                result = std::min(result, circumcenter(o.deformedPositions(), elementCorners(e.index())).minCoeff());

                if (m_subdivisionBarrier) {
                    foreach_subelement(e.index(), [&](const AK1i &sube, auto /* B */, auto x) {
                        result = std::min(result, circumcenter(x, sube).minCoeff());
                    });
                }
            }
            return result;
        }

        virtual Real energy() const override {
            const auto &o = Base::getObj();
            Real result = 0.0;
            for (auto e : mesh().elements()) {
                result += elementEnergy(o.deformedPositions(), elementCorners(e.index()));

                if (m_subdivisionBarrier) {
                    foreach_subelement(e.index(), [&](const AK1i &sube, auto /* B */, auto x) {
                        result += elementEnergy(x, sube);
                    });
                }
            }
            return result;
        }

        // Gradient with respect to the deformed state
        virtual VXd grad_x() const override {
            BENCHMARK_SCOPED_TIMER_SECTION timer("CircumcenterBarrier.grad_x");
            const auto &o = Base::getObj();
            constexpr size_t nlv = Simplex::numVertices(K);
            auto accumulate_per_element_contrib = [&](size_t ei, VXd &g_out) {
                PerElementGradient g = elementGradient(o.deformedPositions(), elementCorners(ei));

                if (m_subdivisionBarrier) {
                    foreach_subelement(ei, [&](const AK1i &sub_e, auto B, auto x) {
                        PerElementGradient gSub = elementGradient(x, sub_e);
                        for (size_t lvi = 0; lvi < nlv; ++lvi) {
                            auto g_v = g.template segment<N>(N * lvi);
                            for (size_t k = 0; k < nlv; ++k) {
                                size_t sub_v = sub_e[k];
                                Real lambda = B(sub_v, lvi);
                                if (lambda == 0) continue;
                                g_v += lambda * gSub.template segment<N>(N * k);
                            }
                        }
                    });
                }

                auto e = mesh().element(ei);
                for (auto v : e.vertices())
                    g_out.template segment<N>(N * v.index()) += g.template segment<N>(N * v.localIndex());
            };

            VXd result;
            result.setZero(numVars());
            assemble_parallel(accumulate_per_element_contrib, result, mesh().numElements());
            return result;
        }

        // Gradient with respect to the rest state
        virtual VXd grad_X() const override {
            return VXd::Zero(numVars());
        }

        // Hessian with respect to the deformed state H_xx
        virtual void hessian(SuiteSparseMatrix &H, bool /* projectionMask */ = true) const override {
            const auto &o = Base::getObj();
            const auto &m = mesh();
            BENCHMARK_SCOPED_TIMER_SECTION timer("CircumcenterBarrier.hessian");
            auto accumulate_per_element_contrib = [&](size_t ei, auto &Hout) { // `auto` here needed for sparsity-pattern sharing optimization
                PerElementHessian eH = elementHessian(o.deformedPositions(), elementCorners(ei));

                if (m_subdivisionBarrier) {
                    constexpr size_t nlv = Simplex::numVertices(K);
                    foreach_subelement(ei, [&](const AK1i &sub_e, auto B, auto x) {
                        PerElementHessian eHSub = elementHessian(x, sub_e);
                        for (size_t lvi_b = 0; lvi_b < nlv; ++lvi_b) {
                            for (size_t lvi_a = 0; lvi_a <= lvi_b; ++lvi_a) {
                                auto H_ab = eH.template block<N, N>(N * lvi_a, N * lvi_b);
                                for (size_t k = 0; k < nlv; ++k) {
                                    size_t sub_v_k = sub_e[k];
                                    Real lambda_a = B(sub_v_k, lvi_a);
                                    if (lambda_a == 0) continue;
                                    for (size_t l = 0; l < nlv; ++l) {
                                        size_t sub_v_l = sub_e[l];
                                        Real lambda_b = B(sub_v_l, lvi_b);
                                        if (lambda_b == 0) continue;
                                        H_ab += (lambda_a * lambda_b) * eHSub.template block<N, N>(N * k, N * l);
                                    }
                                }
                            }
                        }
                    });
                }

                auto e = m.element(ei);
                for (auto v_b : e.vertices()) {
                    for (auto v_a : e.vertices()) {
                        if (v_a.index() > v_b.index()) continue;
                        // Only the upper triangle was computed above...
                        size_t br = N * v_a.localIndex();
                        size_t bc = N * v_b.localIndex();
                        if (br <= bc)
                            Hout.addNZBlock(N * v_a.index(), N * v_b.index(), eH.template block<N, N>(br, bc));
                        else
                            Hout.addNZBlock(N * v_a.index(), N * v_b.index(), eH.template block<N, N>(bc, br).transpose());
                    }
                }
            };

            assemble_parallel(accumulate_per_element_contrib, H, m.numElements());
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
        RawBarrierLog barrier;           // enforce a constraint of the form `c <= 0`
    private:
        bool m_subdivisionBarrier = false; // also enforce a barrier on the mesh generated by one level of subdivision
    };
}

#endif /* end of include guard: CIRCUMCENTERBARRIER_HH */
