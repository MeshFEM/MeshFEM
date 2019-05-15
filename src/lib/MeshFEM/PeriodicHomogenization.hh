#ifndef PERIODICHOMOGENIZATION_HH
#define PERIODICHOMOGENIZATION_HH

#include <vector>
#include <string>
#include <MeshFEM/OneForm.hh>

#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/InterpolantRestriction.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/Parallelism.hh>

// #define FD_SD_DEBUG
#ifdef FD_SD_DEBUG
#include <MeshFEM/MSHFieldWriter.hh>
#endif

namespace PeriodicHomogenization {

////////////////////////////////////////////////////////////////////////////
/*! Solve the linear elasticity periodic homogenization cell problems for
//  each constant strain e^ij:
//       -div E : [ strain(w^ij) + e^ij ] = 0 in omega
//        n . E : [ strain(w^ij) + e^ij ] = 0 on omega's boundary
//        w^ij periodic
//        w^ij = 0 on arbitrary internal node ("pin" no rigid translation constraint)
//  @param[out]   w_ij   Fluctuation displacements (cell problem solutions)
//  @param[inout] sim    Linear elasticity simulator for omega.
//  Warning: this function mutates sim by applying periodic and pin
//           constraints.
*///////////////////////////////////////////////////////////////////////////
template<class _Sim>
void solveCellProblems(std::vector<typename _Sim::VField> &w_ij, _Sim &sim,
                       Real cellEpsilon = 1e-7,
                       bool ignorePeriodicMismatch = false,
                       std::unique_ptr<PeriodicCondition<_Sim::N>> pc = nullptr) {
    typedef typename _Sim::VField  VField;
    typedef typename _Sim::SMatrix SMatrix;
    constexpr size_t numStrains = SMatrix::flatSize();

    sim.applyPeriodicConditions(cellEpsilon, ignorePeriodicMismatch, std::move(pc));
    sim.applyNoRigidMotionConstraint();
    sim.setUsePinNoRigidTranslationConstraint(true);

    w_ij.reserve(numStrains), w_ij.clear();
    for (size_t i = 0; i < numStrains; ++i) {
        BENCHMARK_START_TIMER("Constant Strain Load");
        VField rhs(sim.constantStrainLoad(-SMatrix::CanonicalBasis(i)));
        BENCHMARK_STOP_TIMER("Constant Strain Load");
        w_ij.push_back(sim.solve(rhs));
    }
}

template<class _Sim>
std::vector<typename _Sim::VField> solveCellProblems(_Sim &sim, Real cellEpsilon = 1e-7) {
    std::vector<typename _Sim::VField> w_ij;
    solveCellProblems(w_ij, sim, cellEpsilon);
    return w_ij;
}

////////////////////////////////////////////////////////////////////////////
/*! Compute homogenized elasticity tensor (stress-like version):
//     Eh_ijkl = 1/|Y| int_omega [E : strain(w_ij)]_kl + E_ijkl dV
//  where |Y| = periodic cell (grid bounding box) volume
//  @param[in] w_ij           Fluctuation displacements
//  @param[in] sim            Linear elasticity simulator for omega.
//  @param[in] baseCellVolume |Y| (defaults to mesh.boundingBox().volume())
//  @return    Homogenized elasticity tensor
*///////////////////////////////////////////////////////////////////////////
template<class _Sim>
typename _Sim::ETensor homogenizedElasticityTensor(
        const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim,
        Real baseCellVolume = 0.0) {
    const auto &mesh = sim.mesh();
    if (baseCellVolume == 0.0) baseCellVolume = mesh.boundingBox().volume();

    typedef typename _Sim::SMatrix SMatrix;
    constexpr size_t numStrains = SMatrix::flatSize();
    (void) (numStrains);
    assert(w_ij.size() == numStrains);

    typename _Sim::ETensor Eh;
    typename _Sim::Strain  strain_ij;
    for (auto e : mesh.elements()) {
        typename _Sim::ETensor Econtrib;
        for (size_t i = 0; i < w_ij.size(); ++i) {
            sim.elementStrain(e.index(), w_ij[i], strain_ij);
            Econtrib.DRowAsSymMatrix(i) =
                e->E().doubleContract(strain_ij.average());
        }
        // Elasticity tensor is always constant on each element.
        Econtrib += e->E();
        Econtrib *= e->volume();
        Eh += Econtrib;
    }
    Eh /= baseCellVolume;

    return Eh;
#if 0
        // The following "energy-like" version is equivalent to the more efficient
        // "stress-like" version above:
        // Eh_ijkl = 1/|Y| int_w <E (e(w_ij) + e_ij), e(w_kl) + e_kl> dV,
        typename _Sim::ETensor EhE;
        typename _Sim::Strain  strain_ij, strain_kl;
        for (size_t ei = 0; ei < mesh.numElements(); ++ei) {
            auto e = mesh.element(ei);
            for (size_t ij = 0; ij < numStrains; ++ij) {
                sim.elementStrain(ei, w_ij[ij], strain_ij);
                strain_ij += SMatrix::CanonicalBasis(ij);
                for (size_t kl = ij; kl < numStrains; ++kl) {
                    sim.elementStrain(ei, w_ij[kl], strain_kl);
                    strain_kl += SMatrix::CanonicalBasis(kl);
                    EhE.D(ij, kl) +=
                        Quadrature<_Sim::K, 2 * (_Sim::Degree - 1)>::integrate(
                            [&] (const EvalPt<_Sim::K> &p) {
                                return e->E().doubleContract(strain_ij(p))
                                             .doubleContract(strain_kl(p));
                            }, e->volume());
                }
            }
        }
        EhE /= baseCellVolume;

        return EhE;
#endif
    }

////////////////////////////////////////////////////////////////////////////
/*! Displacement form of homognized tensor:
//  Assuming that the base elasticity tensor is constant over omega, we can
//  rewrite the homogenized elasticity tensor stress integral formula in
//  terms of displacements (using Green's theorem):
//  Eh_ijkl = 1/|Y| int_w [E : strain(w_kl)]_ij + E_ijkl dy
//          = 1/|Y| int_dw E_ijpq frac{1}{2} (w^{kl}_p n_q + w^{kl}_q n_p) dA(y) + E * volFrac
//          = 1/|Y| E_ijpq nw_pq + E * volFrac
//  Where   |Y|  = periodic cell (grid bounding box) volume
//           w   = periodic base cell geometry
//         nw_pq = 0.5 * int_dw [w^{kl}]_p n_q + [w^{kl}]_q n_p dA(y)
//  @param[in] w_ij           Fluctuation displacements
//  @param[in] sim            Linear elasticity simulator for omega.
//  @param[in] baseCellVolume |Y| (could differ from sim.boundingBox().volume())
//  @return    Homogenized elasticity tensor
*///////////////////////////////////////////////////////////////////////////
template<class _Sim>
typename _Sim::ETensor homogenizedElasticityTensorDisplacementForm(
        const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim,
        Real baseCellVolume = 0.0) {
    const auto &mesh = sim.mesh();
    if (baseCellVolume == 0.0) baseCellVolume = mesh.boundingBox().volume();
    using SMatrix = typename _Sim::SMatrix ;
    constexpr size_t numStrains = SMatrix::flatSize();
    (void) (numStrains);
    assert(w_ij.size() == numStrains);

    // Assume elasticity tensor is constant over the entire base cell
    const typename _Sim::ETensor &EBase = mesh.element(0)->E();

    typename _Sim::ETensor Eh;
    SMatrix nw_pq;

    // Displacement restricted to a boundary element
    Interpolant<VectorND<_Sim::N>, _Sim::K - 1, _Sim::Degree> w_be;
    for (auto be : mesh.boundaryElements()) {
        typename _Sim::ETensor Econtrib;
        const auto &n = be->normal();
        for (size_t i = 0; i < w_ij.size(); ++i) {
            const auto &w = w_ij[i];
            // Copy the boundary node displacements into interpolant
            for (size_t ni = 0; ni < w_be.size(); ++ni)
                w_be[ni] = w(be.node(ni).volumeNode().index());
            auto w_be_int = w_be.integrate(be->volume());

            for (size_t p = 0; p < _Sim::N; ++p)
                for (size_t q = p; q < _Sim::N; ++q)
                    nw_pq(p, q) = 0.5 * (w_be_int[p] * n[q] + w_be_int[q] * n[p]);
            Eh.DRowAsSymMatrix(i) += EBase.doubleContract(nw_pq);
        }
    }

    Eh += EBase * mesh.volume();
    Eh /= baseCellVolume;

    return Eh;
}

////////////////////////////////////////////////////////////////////////////
/*! Compute the macroscopic-strain-to-microscopic strain tensors.
//  These are rank 4 tensors that have minor but not major symmetries:
//      G_ijkl(x) = [e(w^kl)(x) + e^kl]_ij
//  @return     A vector of per-element tensors. The tensor at index i is
//              the average of G_ijkl(x) over element i.
*///////////////////////////////////////////////////////////////////////////
template<class _Sim>
std::vector<ElasticityTensor<Real, _Sim::N, false>>
macroStrainToMicroStrainTensors(const std::vector<typename _Sim::VField> &w, const _Sim &sim) {
    size_t numElems = sim.mesh().numElements();
    std::vector<ElasticityTensor<Real, _Sim::N, false>> G(numElems);
    typename _Sim::Strain  strain_ij;
    for (size_t e = 0; e < numElems; ++e) {
        for (size_t ij = 0; ij < w.size(); ++ij) {
            sim.elementStrain(e, w[ij], strain_ij);
            G[e].DColAsSymMatrix(ij) = strain_ij.average();
            G[e].DColAsSymMatrix(ij) += _Sim::SMatrix::CanonicalBasis(ij);
        }
    }
    return G;
}

// Per-boundary-element interpolant type needed to express the homogenized
// tensor shape derivative.
template<class _Sim>
using BEHTensorGradInterpolant = Interpolant<typename _Sim::ETensor,
    _Sim::K - 1, 2 * (_Sim::Degree - 1)>;

////////////////////////////////////////////////////////////////////////////
/*! Computes the steepest ascent direction (i.e. the theta maximizing the
//  shape derivative DS[theta]) of each component of the homogenized
//  elasticity tensor. This is a per-boundary-element piecewise constant
//  (FEM degree 1) or quadratic (FEM degree 2) rank 4 tensor field.
//  @param[in]  w       fluctuation displacements (cell problem solutions)
//  @param[in]  sim     linear elasticity solver
//  @return     per-boundary-element rank 4 tensor field.
*///////////////////////////////////////////////////////////////////////////
template<class _Sim>
std::vector<BEHTensorGradInterpolant<_Sim>>
homogenizedElasticityTensorGradient(
        const std::vector<typename _Sim::VField> &w, const _Sim &sim) {
    typedef typename _Sim::ETensor ETensor;
    typedef typename _Sim::SMatrix SMatrix;
    constexpr size_t numStrains = SMatrix::flatSize();
    constexpr size_t K = _Sim::K;
    constexpr size_t Deg = _Sim::Degree;
    assert(w.size() == numStrains);

    const auto &mesh = sim.mesh();
    Real bboxVol = mesh.boundingBox().volume();

    // Shape derivative evaluated on normal velocity v_n:
    // DS(E_H)[v_n n] = 1/|Y| int_dt <E [e_ij + e(w_ij)], e_kl + e(w_kl)> v_n dA
    // So the steepest ascent direction is to evolve with
    //      v_n(x) = 1/|Y| <E [e_ij + e(w_ij)], e_kl + e(w_kl)> := G_ijkl(x)
    // for each non-periodic boundary point x.
    // For degree d FEM, G_ijkl is a degree 2 * (d - 1) polynomial on each
    // boundary element and is stored as a rank 4 tensor interpolant per
    // boundary element.
    constexpr size_t GDeg = 2 * (Deg - 1);
    typedef Interpolant<ETensor, K - 1, GDeg> G_t;
    std::vector<G_t> gradient(mesh.numBoundaryElements());
    typename _Sim::Strain  we_ij, we_kl;
    // Compute volume quantity
    Interpolant<ETensor, K, GDeg> G_elem;
    for (auto e : mesh.elements()) {
        if (!e.isBoundary()) continue;
        for (size_t ij = 0; ij < numStrains; ++ij) {
            sim.elementStrain(e.index(), w[ij], we_ij);
            we_ij += SMatrix::CanonicalBasis(ij);
            for (size_t kl = ij; kl < numStrains; ++kl) {
                sim.elementStrain(e.index(), w[kl], we_kl);
                we_kl += SMatrix::CanonicalBasis(kl);
                auto G_ijkl = Interpolation<K, GDeg>::interpolant(
                    [&] (const EvalPt<K> &p) {
                        return e->E().doubleContract(we_ij(p))
                                     .doubleContract(we_kl(p));
                    });
                G_ijkl /= bboxVol;
                // Copy single entry interpolant over into interpolated rank
                // 4 tensor's entries.
                for (size_t n = 0; n < Simplex::numNodes(K, GDeg); ++n)
                    G_elem[n].D(ij, kl) = G_ijkl[n];
            }
        }

        // Distribute G_elem to all of this element's boundary faces/edges
        for (auto f : e.interfaces()) {
            auto be = mesh.boundaryElement(f.boundaryEntity().index());
            if (!be) continue;
            auto &beGrad = gradient.at(be.index());
            // Zero gradient on the periodic boundary. ETensor default
            // constructor zero-inits, so beGrad should currently be zero.
            if (be->isInternal) continue;
            restrictInterpolant(e, be, G_elem, beGrad);
        }
    }

    return gradient;
}

////////////////////////////////////////////////////////////////////////////////
// Continuous Shape Derivatives (Eulerian)
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
// Shape derivative of fluctuation displacements evaluated on a particular
// velocity field. This is the "direct" approach not using the adjoint
// method:
// Solve cell problems with load
//      - int_bdry (v dot n) (strain(phi) : C : [strain(w^kl) + e^kl]) dA
////////////////////////////////////////////////////////////////////////////
template<class _Sim, class _NormalShapeVelocity>
void fluctuationDisplacementShapeDerivatives(const _Sim &sim,
        const std::vector<typename _Sim::VField> &w,
        const _NormalShapeVelocity &vn,
        std::vector<typename _Sim::VField> &dot_w,
        bool projectOutNormalStress = false) {
    BENCHMARK_START_TIMER("Fluctuation Shape Derivatives");

    constexpr size_t Deg = _Sim::Degree;
    constexpr size_t   K = _Sim::K;
    typename _Sim::Strain  strain_kl;

    const auto &mesh = sim.mesh();

    using SMatrix = typename _Sim::SMatrix;
    std::vector<Interpolant<SMatrix, K - 1, Deg - 1>> bdry_stresses;
    bdry_stresses.resize(mesh.numBoundaryElements());

#ifdef FD_SD_DEBUG
        static size_t it = 0;
        MSHFieldWriter writer("debug_fd_sd_" + std::to_string(it) + ".msh", sim.mesh());
        ++it;
#endif // FD_SD_DEBUG

    dot_w.clear(), dot_w.reserve(w.size());
    for (size_t kl = 0; kl < w.size(); ++kl) {
        for (auto e : mesh.elements()) {
            if (!e.isBoundary()) continue;
            const auto &C = e->E();
            sim.elementStrain(e.index(), w[kl], strain_kl);
            strain_kl += SMatrix::CanonicalBasis(kl);

            for (size_t fi = 0; fi < e.numNeighbors(); ++fi) {
                auto f = mesh.boundaryElement(e.interface(fi).boundaryEntity().index());
                if (!f) continue;
                auto &bdry_stress_kl = bdry_stresses.at(f.index());
                if (f->isInternal) bdry_stress_kl = 0;
                else               restrictInterpolant(e, f, strain_kl, bdry_stress_kl);
                for (size_t n = 0; n < bdry_stress_kl.size(); ++n)
                    bdry_stress_kl[n] = C.doubleContract(bdry_stress_kl[n]);

                if (projectOutNormalStress) {
                    // Projection:  s - (sn) n^T - n (sn)^T + (n^T s n)(n n^T)
                    auto nnt = SMatrix::ProjectionMatrix(f->normal());
                    for (size_t n = 0; n < bdry_stress_kl.size(); ++n) {
                        auto &s = bdry_stress_kl[n];
                        // Subtract tangent-normal components (double subtracting normal-normal components)
                        auto s_half_tn = SMatrix::SymmetrizedOuterProduct(s.contract(f->normal()), f->normal());
                        s -= s_half_tn, s -= s_half_tn;
                        // Clear out normal-normal component
                        s -= s.doubleContract(nnt) * nnt;
                        // assert(bdry_stress_kl[n].contract(f->normal()).norm() < 1e-13);
                    }
                }
            }
        }

        auto loadChange = sim.changeInDivTensorLoad(vn, bdry_stresses, true);
        dot_w.push_back(sim.solve(loadChange));

#ifdef FD_SD_DEBUG
            typename _Sim::VField outField;
            // Subtract off average displacements so that fields are comparable
            // across meshes.
            outField = w[kl];
            outField -= outField.mean();
            writer.addField("w " + std::to_string(kl), outField);
            outField = dot_w[kl];
            outField -= outField.mean();
            writer.addField("dot w " + std::to_string(kl), outField);
            writer.addField("rhs " + std::to_string(kl), sim.dofToNodeField(loadChange));
#endif // FD_SD_DEBUG
        }

    BENCHMARK_STOP_TIMER("Fluctuation Shape Derivatives");
}

////////////////////////////////////////////////////////////////////////////////
// Discrete Shape Derivatives (Lagrangian)
////////////////////////////////////////////////////////////////////////////////
// Compute the exact derivative of the homogenized elasticity tensor with
// respect to each mesh vertex position.
template<class _Sim>
OneForm<typename _Sim::ETensor, _Sim::N>
homogenizedElasticityTensorDiscreteDifferential(const std::vector<typename _Sim::VField> &w,
        const _Sim &sim) {
    static constexpr size_t N = _Sim::N;
    static constexpr size_t Deg = _Sim::Degree;
    using ETensor = typename _Sim::ETensor;
    using SMatrix = typename _Sim::SMatrix;
    using SFGradient = Interpolant<VectorND<_Sim::N>, _Sim::N, _Sim::Degree - 1>; // Shape fn grad
    using OF = OneForm<ETensor, _Sim::N>;

    const auto &mesh = sim.mesh();

    // Zero-initializes due to ETensor's default constructor.
    OF dCh(mesh.numVertices());

#if MESHFEM_WITH_TBB
    tbb::combinable<OF> sum(dCh);
#endif

    // True deformation strains/stresses under each cell problem load
    // (constant + fluctuation)
    using Strain = typename _Sim::Strain;

    assert(w.size() == flatLen(N));

    auto accumElementContrib = [&](size_t ei) {
        auto e = mesh.element(ei);
        std::vector<Strain> strain(flatLen(N));
        std::vector<Strain> stress(flatLen(N));
        std::vector<SFGradient> gradPhi_n(e.numNodes());
#if MESHFEM_WITH_TBB
        OF &result = sum.local();
#else
        OF &result = dCh;
#endif
        // Precompute stresses/strains
        for (size_t ij = 0; ij < flatLen(N); ++ij) {
            e->strain(e, w[ij], strain[ij]);
            strain[ij] += SMatrix::CanonicalBasis(ij);
            stress[ij]  = strain[ij].doubleContract(e->E());
        }

        // Precompute (scalar) shape function gradients.
        for (size_t ni = 0; ni < e.numNodes(); ++ni)
            gradPhi_n[ni] = e->gradPhi(ni);

        // (Only need to compute upper tri of flattened elasticity tensor).
        for (size_t ij = 0; ij < flatLen(N); ++ij) {
            for (size_t kl = ij; kl < flatLen(N); ++kl) {
                // Initialize with energy dilation term
                Real mutualEnergy = Quadrature<Strain::K, 2 * Strain::Deg>::integrate([&](const EvalPt<N> &pt) { return
                        strain[ij](pt).doubleContract(stress[kl](pt)); }, e->volume());
                for (auto v : e.vertices()) {
                    for (size_t c = 0; c < N; ++c)
                        result(v.index())[c].D(ij, kl) += mutualEnergy * e->gradBarycentric()(c, v.localIndex());
                }

                // Compute the delta strain phi contribution
                for (auto n : e.nodes()) {
                    const SFGradient &gphi = gradPhi_n[n.localIndex()];

                    // Avoid recomputing ij-kl term with ij == kl
                    Interpolant<VectorND<N>, Strain::K, Strain::Deg> stress_contract_w;
                    for (size_t inode = 0; inode < stress_contract_w.size(); ++inode) {
                        stress_contract_w[inode]                = stress[kl][inode].contract(w[ij](n.index()));
                        if (ij != kl) stress_contract_w[inode] += stress[ij][inode].contract(w[kl](n.index()));
                        else          stress_contract_w[inode] *= 2;
                    }

                    for (auto v : e.vertices()) {
                        VectorND<N> gbary = e->gradBarycentric().col(v.localIndex());
                        // delta-strain term of effect of perturbing component c of vertex v
                        VectorND<N> dstrainTerm = Quadrature<N, Deg>::integrate([&](const EvalPt<N> &pt) { return
                                VectorND<N>(gbary.dot(stress_contract_w(pt)) * gphi(pt)); }, e->volume());
                        for (size_t c = 0; c < N; ++c)
                            result(v.index())[c].D(ij, kl) -= dstrainTerm[c];
                    }
                }
            }
        }
    };

#if MESHFEM_WITH_TBB
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, mesh.numElements()),
        [&](const tbb::blocked_range<size_t> &r) {
            for (size_t ei = r.begin(); ei < r.end(); ++ei) accumElementContrib(ei);
        });
    dCh = sum.combine([](const OF &a, const OF &b) { return a + b; } );
#else
    for (auto e : sim.mesh().elements()) { accumElementContrib(e.index()); }
#endif

    dCh /= sim.mesh().boundingBox().volume();

    return dCh;
}

// Change in the homogenized elasticity tensor due to mesh vertex
// perturbations delta_p.
// Currently just uses homogenizedElasticityTensorGradient; could be
// optimized by directly computing the integrals one boundary element at a
// time. We could also use the simpler (no interpolant restriction) but
// more expensive volume integral version:
// dCh_ijkl = 1/|Y| int_omega (mutual_energy) delta vol/vol dV
//                + int_omega [(delta strain(phi^m)) w^ij_m] : C : [strain(w^kl) + e^kl] dV
//                + int_omega [strain(w^ij) + e^ij] : C : [(delta strain(phi^m)) w^kl_m] dV
// where delta terms are lagrangian derivatives.
template<class _Sim>
typename _Sim::ETensor
deltaHomogenizedElasticityTensor(const _Sim &sim,
        const std::vector<typename _Sim::VField> &w,
        const typename _Sim::VField &delta_p) {
    auto sd = homogenizedElasticityTensorGradient(w, sim);
    constexpr size_t N = _Sim::N;
    using NSVI = Interpolant<Real, N - 1, 1>;
    NSVI nsv;
    typename _Sim::ETensor deltaCh;
    for (auto be : sim.mesh().boundaryElements()) {
        // Compute boundary element's linear normal velocity under delta_p
        for (auto bv : be.vertices())
            nsv[bv.localIndex()] = be->normal().dot(delta_p(bv.volumeVertex().index()));
        // Integrate it against the shape derivative
        const auto &dCh = sd.at(be.index());
        deltaCh += Quadrature<N - 1, NSVI::Deg + std::decay<decltype(dCh)>::type::Deg>::
            integrate([&](const EvalPt<N - 1> &pt) { return
                    nsv(pt) * dCh(pt);
                }, be->volume());
    }
    return deltaCh;
}

template<class _Sim>
typename _Sim::ETensor
deltaHomogenizedComplianceTensor(const _Sim &sim,
        const std::vector<typename _Sim::VField> &w,
        const typename _Sim::VField &delta_p) {
    auto Ch  = homogenizedElasticityTensor(w, sim);
    auto deltaCh = deltaHomogenizedElasticityTensor(sim, w, delta_p);
    return -Ch.inverse().doubleDoubleContract(deltaCh);
}

// Change in the fluctuation displacements due to mesh vertex perturbations
// delta_p
template<class _Sim>
std::vector<typename _Sim::VField>
deltaFluctuationDisplacements(const _Sim &sim,
        const std::vector<typename _Sim::VField> &w,
        const typename _Sim::VField &delta_p)
{
    typedef typename _Sim::VField  VField;
    using SMatrix = typename _Sim::SMatrix;

    std::vector<VField> delta_w;
    delta_w.reserve(w.size());
    for (size_t ij = 0; ij < w.size(); ++ij) {
        auto rhs = sim.deltaConstantStrainLoad(-SMatrix::CanonicalBasis(ij), delta_p);
        rhs     -= sim.applyDeltaStiffnessMatrix(w[ij], delta_p);
        delta_w.push_back(sim.solve(rhs));
    }
    return delta_w;
}

// Change in macro-to-micro strain tensors due to mesh vertex perturbations delta_p:
//    delta G_ijkl(x) = delta [e(w^kl)(x) + e^kl]_ij = delta e(w^kl)(x)_ij
template<class _Sim>
std::vector<ElasticityTensor<Real, _Sim::N, false>>
deltaMacroStrainToMicroStrainTensors(const _Sim &sim,
        const std::vector<typename _Sim::VField> &w,
        const std::vector<typename _Sim::VField> &delta_w,
        const typename _Sim::VField &delta_p) {
    size_t numElems = sim.mesh().numElements();
    std::vector<ElasticityTensor<Real, _Sim::N, false>> deltaG(numElems);
    for (size_t kl = 0; kl < w.size(); ++kl) {
        auto delta_we = sim.deltaAverageStrainField(w[kl], delta_w[kl], delta_p);
        for (size_t e = 0; e < numElems; ++e)
            deltaG[e].DColAsSymMatrix(kl) = delta_we(e);
    }
    return deltaG;
}

} // namespace PeriodicHomogenization

#endif /* end of include guard: PERIODICHOMOGENIZATION_HH */
