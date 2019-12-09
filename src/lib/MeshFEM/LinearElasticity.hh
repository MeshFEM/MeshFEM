#ifndef NEWLINEARELASTICITY_HH
#define NEWLINEARELASTICITY_HH

#include <MeshFEM/SymmetricMatrixInterpolant.hh>
#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/Fields.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Parallelism.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/OneForm.hh>

namespace LinearElasticity {

////////////////////////////////////////////////////////////////////////////
// Policies for getting material tensors
////////////////////////////////////////////////////////////////////////////
template<size_t _N>
struct ETensorStoreGetter {
    typedef ElasticityTensor<Real, _N> ETensor;
    ETensorStoreGetter(const ETensor &E) : m_E(E) { }
    ETensorStoreGetter() : m_E(1, 0) { }
    const ETensor &operator()() const { return m_E; }
          ETensor &operator()()       { return m_E; }
private:
    ETensor m_E;
};

template<class _Material>
struct _HMG {
    typedef typename _Material::ETensor ETensor;
    static _Material material;
    const ETensor &operator()() const { return material.getTensor(); }
          ETensor &operator()()       { return material.getTensor(); }
};
template<class _Material>
_Material _HMG<_Material>::material;
template<template<size_t> class _Material>
struct HomogenousMaterialGetter {
    template<size_t _N>
    using Getter = _HMG<_Material<_N>>;
};

// To allow extra configuration of the linear elasticity data we store on the
// FEMMesh, LinearElasticityData is a templated wrapper class that contains the
// templated FEMData "Data" subclass.
template<template<size_t> class _ETensorGetter = ETensorStoreGetter>
struct LinearElasticityData {
template<size_t _K, size_t _Deg, class EmbeddingSpace>
struct Data : public DefaultFEMData<_K, _Deg, EmbeddingSpace> {
    static_assert(EmbeddingSpace::RowsAtCompileTime == _K,
                 "Embedding space dimension, N, must match simplex dimension, K.");
    static constexpr size_t N = _K;
    static constexpr size_t Degree = _Deg;
    typedef _ETensorGetter<N> ETensorGetter;
    typedef EmbeddingSpace Vector;
    typedef EmbeddingSpace Point;
    typedef DefaultFEMData<_K, _Deg, Vector>   BaseData;
    typedef SymmetricMatrixValue<Real, N> SMatrix;
    typedef SymmetricMatrixInterpolant<SMatrix, _K, _Deg - 1> Strain;
    typedef Strain Stress;
    typedef VectorField<Real, N> VField;

    // All of these routines can be heavily optimized...
    struct Element : public BaseData::Element {
        typedef typename BaseData::Element Base;
        using Base::gradPhi; using Base::deltaGradPhi;

        static constexpr size_t nNodes = Simplex::numNodes(_K, _Deg);
        static constexpr size_t nVecPhi = N * nNodes;
        typedef Eigen::Matrix<Real, N,  nNodes> ElementLoad;
        typedef Eigen::Matrix<Real, nVecPhi, nVecPhi> PerElementStiffness;

        void configure(const ETensorGetter &EGetter) { m_E = EGetter; }
        decltype(((const ETensorGetter *) 0)->operator()()) E() const { return m_E(); }

        std::vector<Strain> vecPhiStrains() const {
            std::vector<Strain> strains(N * nNodes);
            // Compute the strain of vector basis function i * N + c.
            // In 2D, these vector basis functions look like:
            // (phi0, 0), (0, phi0), (phi1, 0), (0, phi1), ...
            for (size_t i = 0; i < nNodes; ++i) {
                auto gPhi = gradPhi(i);
                for (size_t c = 0; c < N; ++c) {
                    // We need the strain value at each interpolation node.
                    for (size_t inode = 0; inode < Strain::numNodalValues; ++inode) {
                        for (size_t var = 0; var < N; ++var) {
                            strains[i * N + c][inode](c, var) +=
                                ((var == c) ? 1.0 : 0.5) * gPhi[inode](var);
                        }
                    }
                }
            }
            return strains;
        }

        template<class _ElemHandle>
        void strain(_ElemHandle elem, const VField &u, Strain &out) const {
            out.clear();
            for (size_t i = 0; i < nNodes; ++i) {
                auto gPhi = gradPhi(i);
                const auto &ui = u(elem.node(i).index());
                for (size_t c = 0; c < N; ++c) {
                    auto ui_c = ui[c];
                    // We need the strain value at each interpolation node.
                    for (size_t inode = 0; inode < Strain::numNodalValues; ++inode) {
                        for (size_t var = 0; var < N; ++var) {
                            out[inode](c, var) += ((var == c) ? 1.0 : 0.5) *
                                ui_c * gPhi[inode](var);
                        }
                    }
                }
            }
        }

        template<class _ElemHandle>
        void stress(_ElemHandle elem, const VField &u, Stress &out) const {
            Strain eps;
            strain(elem, u, eps);
            out = eps.doubleContract(m_E());
        }

        // Contribution to the "per-element constant stress load"
        //      -div C : strain(u) = -div cstress in omega
        //       n . C : strain(u) =  n . cstress on domega
        // Integrating the volume equation against shape function phi:
        //      int_omega   -phi . div (C : strain(u)) dV = int_omega    -phi . div c_stress dV
        // ==>  int_omega strain(phi) : C : strain(u)  dV = int_omega strain(phi) : c_stress dV
        //(The boundary terms disappear due to the Neumann condition). The RHS
        // vector can be interpreted as applying the stiffness matrix to the linear
        // displacement field "(C^-1 : cstress) x".
        // Since c_stress is constant, we can pull it out of the integral for efficiency.
        void perElementConstantStressLoad(const SMatrix &cstress, ElementLoad &l) const {
            // Original version:
            // std::vector<Strain> phiStrains = vecPhiStrains();
            // for (size_t i = 0; i < nNodes; ++i) {
            //     for (size_t c = 0; c < N; ++c)
            //         l(c, i) = phiStrains[i * N + c].integrate(Base::volume()).doubleContract(cstress);
            // }

            // Optimized version (take advantage of cstress' symmetries)
            // strain(phi^(i * N + c))_ab : cstress_ab
            //     = [gphi^i]_a delta_bc cstress_ab
            //     = [gphi^i]_a cstress_ac = [cstress . gphi^i]_c
            for (size_t i = 0; i < nNodes; ++i) {
                const VectorND<N> &gpi_int = gradPhi(i).integrate(Base::volume());
                l.col(i) = cstress.contract(gpi_int);
            }
        }

        // Contribution to the "constant strain load"
        //      -div C : strain(u) = -div C : cstrain in omega
        //       n . C : strain(u) =  n . C : cstrain on domega
        // Since material properties C are constant over the element, this is a
        // "constant stress load."
        // NOTE: this is actually the **negated** RHS for the periodic
        // homogenization cell problem corresponding to cstrain.
        void perElementConstantStrainLoad(const SMatrix &cstrain, ElementLoad &l) const {
            perElementConstantStressLoad(m_E().doubleContract(cstrain), l);
        }

        // Gets ***upper triangle*** of the per-element stiffness matrix.
        void perElementStiffness(PerElementStiffness &Ke) const {
            // Unoptimized version:
            // std::vector<Strain> strainPhi = vecPhiStrains();
            // std::vector<Stress> stressPhi;
            // stressPhi.reserve(strainPhi.size());
            // for (size_t i = 0; i < strainPhi.size(); ++i)
            //     stressPhi.emplace_back(strainPhi[i].doubleContract(m_E()));
            // for (size_t i = 0; i < stressPhi.size(); ++i) {
            //     for (size_t j = i; j < strainPhi.size(); ++j) {
            //         Ke(i, j) = Quadrature<_K, 2 * Strain::Deg>::integrate(
            //             [&] (const EvalPt<_K> &p) {
            //                 return stressPhi[i](p).doubleContract(strainPhi[j](p));
            //         }, Base::volume());
            //     }
            // }

            // Optimized version (~2x faster):
            // We take advantage of C's major/minor symmetries to simplify:
            // strain(phi^(i * N + c)) : C : strain(phi^(j * N + d))
            // = grad(phi^i)_a delta_pc C_parb grad(phi^j)_b delta_rd   (de-symmetrize)
            // = grad(phi^i)_a C_cadb grad(phi^j)_b
            // = grad(phi^i)_a C_acdb grad(phi^j)_b
            // = gpi . M(c, d) gpj
            // where [M(c, d)]_ab := C_acdb
            // Notice by C's symmetries M(d, c) = C_adcb = C_bcda = M(c, d)^T.
            // We can exploit this symmetry to further eliminate duplicate computations
            // Ke(j * N + d, i * N + c) = gpj . M(d, c) gpi = gpj . M(c, d)^T gpi
            //                          = gpi . M(c, d) gpj = Ke(i * N + c, j * N + d)
            // We loop over only (c, d) with (c <= d) but compute the d > c
            // entries by "swapping" i and j
            std::vector<typename Base::SFGradient> grad_phis(nNodes);
            for (size_t n = 0; n < nNodes; ++n)
                grad_phis[n] = gradPhi(n);

            const auto &C = m_E();
            Eigen::Matrix<Real, N, N> M;
            for (size_t c = 0; c < N; ++c) {
                for (size_t d = c; d < N; ++d) {
                    for (size_t a = 0; a < N; ++a)
                        for (size_t b = 0; b < N; ++b)
                            M(a, b) = C(a, c, d, b);

                    // All pairs ((c, i), (d, j)) with (i N + c <= j N + d)
                    // Since we only consider c <= d we must consider all (i, j)
                    for (size_t j = 0; j < nNodes; ++j) {
                        size_t vj = j * N + d;
                        typename Base::SFGradient Mgpj;
                        for (size_t inode = 0; inode < Mgpj.size(); ++inode)
                            Mgpj[inode] = M * grad_phis[j][inode];
                        for (size_t i = 0; i < nNodes; ++i) {
                            size_t vi = i * N + c;
                            // Note: if c != d, then either (c, d) or (d, c)
                            // entry will lie in the upper triangle. Only if
                            // c == d can this loop contribute to lower tri only
                            if ((c == d) && (vi > vj)) continue; // don't compute lower tri-only values

                            Real val = Quadrature<_K, 2 * Strain::Deg>::integrate([&] (const EvalPt<_K> &p) {
                                return grad_phis[i](p).dot(Mgpj(p));
                            }, Base::volume());
                            // (c, d) entry, if in upper tri
                            if (vi <= vj) Ke(vi, vj) = val;
                            // (d, c) entry, if in upper tri
                            else          Ke(vj, vi) = val;
                        }
                    }
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // Discrete shape derivative functions
        ////////////////////////////////////////////////////////////////////////
        // Change in basis functions' strains due to element corner perturbation
        template<class CornerPerturbations>
        std::vector<Strain> deltaVecPhiStrains(const CornerPerturbations &delta_p) const {
            std::vector<Strain> deltaStrains(N * nNodes);
            for (size_t i = 0; i < nNodes; ++i) {
                auto dgPhi = deltaGradPhi(i, delta_p);
                for (size_t c = 0; c < N; ++c) {
                    for (size_t inode = 0; inode < Strain::numNodalValues; ++inode) {
                        for (size_t var = 0; var < N; ++var) {
                            deltaStrains[i * N + c][inode](c, var) +=
                                ((var == c) ? 1.0 : 0.5) * dgPhi[inode](var);
                        }
                    }
                }
            }
            return deltaStrains;
        }

        // Change in strain due to element corner perturbation of a **fixed**
        // nodal vector field u. In other words, this is only one of the two
        // terms giving the change in a linear elasticity solution's strain.
        template<class _ElemHandle, class CornerPerturbations>
        void deltaStrain(const CornerPerturbations &delta_p, _ElemHandle elem,
                         const VField &u, Strain &deps) const {
            deps.clear();
            for (size_t i = 0; i < nNodes; ++i) {
                auto dgPhi = deltaGradPhi(i, delta_p);
                const auto &ui = u(elem.node(i).index());
                for (size_t c = 0; c < N; ++c) {
                    auto ui_c = ui[c];
                    // We need the strain value at each interpolation node.
                    for (size_t inode = 0; inode < Strain::numNodalValues; ++inode) {
                        for (size_t var = 0; var < N; ++var) {
                            deps[inode](c, var) += ((var == c) ? 1.0 : 0.5) *
                                ui_c * dgPhi[inode](var);
                        }
                    }
                }
            }
        }

        // Change in stress due to element corner perturbation of a **fixed**
        // nodal vector field u.
        template<class _ElemHandle, class CornerPerturbations>
        void deltaStress(const CornerPerturbations &delta_p, _ElemHandle elem,
                         const VField &u, Stress &dsigma) const {
            Strain deps;
            deltaStrain(delta_p, elem, u, deps);
            dsigma = deps.doubleContract(m_E());
        }

        // Change in constant strain load due to element corner perturbation.
        template<class CornerPerturbations>
        void deltaPerElementConstantStrainLoad(const CornerPerturbations &delta_p,
                const SMatrix &cstrain, ElementLoad &l) const {
            std::vector<Strain> phiStrains = vecPhiStrains(),
                               dphiStrains = deltaVecPhiStrains(delta_p);
            SMatrix s(m_E().doubleContract(cstrain));
            Real dvol = Base::volume() * Base::relativeDeltaVolume(delta_p);
            for (size_t i = 0; i < Simplex::numNodes(_K, _Deg); ++i) {
                for (size_t c = 0; c < N; ++c) {
                    l(c, i) = dphiStrains[i * N + c].integrate(Base::volume()).doubleContract(s);
                    l(c, i) += phiStrains[i * N + c].integrate(dvol).doubleContract(s);
                }
            }
        }

        // Change in per-element stiffness matrix due to element corner perturbation.
        // Computes ***upper triangle*** only.
        template<class CornerPerturbations>
        void deltaPerElementStiffness(const CornerPerturbations &delta_p, PerElementStiffness &dKe) const {
            std::vector<Strain> strainPhi = vecPhiStrains(),
                               dstrainPhi = deltaVecPhiStrains(delta_p);
            Real dvol = Base::volume() * Base::relativeDeltaVolume(delta_p);

            std::vector<Stress>  stressPhi;
            stressPhi.reserve(strainPhi.size());
            for (size_t i = 0; i < strainPhi.size(); ++i)
                stressPhi.emplace_back(strainPhi[i].doubleContract(m_E()));

            for (size_t i = 0; i < stressPhi.size(); ++i) {
                for (size_t j = i; j < strainPhi.size(); ++j) {
                    dKe(i, j) = Quadrature<_K, 2 * (_Deg - 1)>::integrate(
                        [&] (const EvalPt<_K> &p) {
                            return stressPhi[i](p).doubleContract(dstrainPhi[j](p)) +
                                   stressPhi[j](p).doubleContract(dstrainPhi[i](p));
                    }, Base::volume());
                    dKe(i, j) += Quadrature<_K, 2 * (_Deg - 1)>::integrate(
                        [&] (const EvalPt<_K> &p) {
                            return stressPhi[i](p).doubleContract(strainPhi[j](p));
                    }, dvol);
                }
            }
        }

    private:
        ETensorGetter m_E;
    };

    struct BoundaryElement : public BaseData::BoundaryElement {
        typedef typename BaseData::BoundaryElement Base;

        // Note: this could be optimized by adding a lookup table of shape
        // function integrals.
        Vector nodalNeumannLoad(size_t ni) const {
            Interpolant<Real, _K - 1, _Deg> phi;
            phi = 0;
            phi[ni] = 1.0;
            Real weight = phi.integrate(Base::volume());
            return weight * neumannTraction;
        }

        template<class CornerPerturbations>
        Vector nodalDeltaNeumannLoad(size_t ni, const CornerPerturbations &delta_p, Real neumannArea, Real intNablaDotV) const {
            Vector result;

            // prepare phi
            Interpolant<Real, _K - 1, _Deg> phi;
            phi = 0;
            phi[ni] = 1.0;

            // prepare dvol, representing \nabla . v
            Real dvol = Base::volume() * Base::relativeDeltaVolume(delta_p);

            // compute weight
            Real weight = phi.integrate(dvol);

            result = weight * neumannTraction;

            // Now, sum up the term related to how the area change
            result -= intNablaDotV / neumannArea * phi.integrate(Base::volume()) * neumannTraction;

            return result;
        }

        Vector neumannTraction;

        // In some cases, "boundary" faces can actually represent internal faces
        // of the true object we're simulating. For example, for periodic
        // boundary conditions, the faces with all nodes on the periodic
        // boundary are actually internal after the identified nodes are
        // stitched together.
        bool isInternal = false;

        bool isInContactRegion = false; // true only if edge is in contact with other object

        int  contactElement = -1; // index to other element of the mesh with exact same vertices
    };

    struct BoundaryNode {
        ComponentMask dirichletComponents;
        Vector dirichletDisplacement;
        size_t dirichletRegionIdx = 0;

        bool hasDirichlet() const { return dirichletComponents.hasAny(N); }
        void setDirichlet(ComponentMask mask, const Vector &val) {
            for (size_t c = 0; c < N; ++c) {
                if (!mask.has(c)) continue;
                // If a new component is being constrained, merge
                if (!dirichletComponents.has(c)) {
                    dirichletComponents.set(c);
                    dirichletDisplacement[c] = val[c];
                }
                // Otherwise, make sure there isn't a conflict
                else {
                    if (std::abs(dirichletDisplacement[c] - val[c]) > 1e-10)
                        throw std::runtime_error("Conflicting dirichlet displacements.");
                }
            }
        }

        void setDirichletRegion(size_t idx) {
            if (dirichletRegionIdx != 0 && dirichletRegionIdx != idx) {
                std::cerr << "WARNING: region traction currently unsupported for vertices "
                          << "belonging to multiple regions" << std::endl;
            }
            dirichletRegionIdx = idx;
        }

        int contactNode = -1; // index to other node of the mesh with exact same position
    };
};
};

template<size_t _K, size_t _Deg,
         template<size_t> class _ETensorGetter = ETensorStoreGetter>
using Mesh = FEMMesh<_K, _Deg, VectorND<_K>,
         LinearElasticityData<_ETensorGetter>::template Data>;

// Mesh where all elements share the same, static material.
// WARNING: This material is actually shared by all meshes of the same dimension!
template<size_t _K, size_t _Deg,
         template<size_t> class _ETensorGetter = HomogenousMaterialGetter<Materials::Constant>::template Getter>
using HomogenousMesh =
        FEMMesh<_K, _Deg, VectorND<_K>,
                 LinearElasticityData<_ETensorGetter>::template Data>;

template<class _Mesh>
class Simulator {
public:
    typedef _Mesh    Mesh;
    typedef typename Mesh::FEMData LEData;
    using ETensorGetter = typename LEData::ETensorGetter;

    typedef typename LEData::Point Point;

    static constexpr size_t N = Mesh::FEMData::N;
    static constexpr size_t K = Mesh::FEMData::N;
    static constexpr size_t Degree = Mesh::FEMData::Degree;
    static constexpr size_t numElemVertices = Simplex::numVertices(N);

    using  OForm = ScalarOneForm<N>;

    typedef ScalarField<Real>             SField;
    typedef VectorField<Real, N>          VField;
    typedef ElasticityTensor<Real, N>     ETensor;
    typedef SymmetricMatrixValue<Real, N> SMatrix;
    typedef SymmetricMatrixField<Real, N> SMField;
    typedef typename LEData::Strain Strain;
    typedef typename LEData::Strain Stress;

    typedef TripletMatrix<Triplet<Real> > TMatrix;

    template<class Elements, class Vertices>
    Simulator(const Elements &elems, const Vertices &vertices)
        : m_useRigidMotionConstraint(false), m_useNRTPinConstraint(false),
          m_mesh(elems, vertices)
    {
        size_t negativeElements = 0;
        for (auto e : m_mesh.elements())
            if (e->volume() < 0) ++negativeElements;
        if (negativeElements > 0) {
            std::cerr << "Found " << negativeElements << " elements with negative volume..." << std::endl;
            throw std::runtime_error(
                    "Mesh has negatively oriented elements.\nCorrect with: mesh_convert --reorientNegativeElements.");
        }
    }

    const _Mesh &mesh() const { return m_mesh; }
          _Mesh &mesh()       { return m_mesh; }

    // Solve for equilibrium under DoF load f
    VField solve(const VField &f) const {
        if (!m_system.isSet()) m_buildConstrainedSystem();

        BENCHMARK_START_TIMER_SECTION("Elasticity Solve");
        std::vector<Real> x;
        m_system.solve(f, x);
        BENCHMARK_STOP_TIMER_SECTION("Elasticity Solve");
        return dofToNodeField(x);
    }

    VField solveAdjoint(const VField &f, const VField &/*u*/) const {
        return solve(f);
    }

    // Get strain on element i (interpolant)
    void elementStrain(size_t i, const VField &u, Strain &e) const {
        assert(i < m_mesh.numElements());
        auto elem = m_mesh.element(i);
        elem->strain(elem, u, e);
    }

    // Get stress on element i (interpolant)
    void elementStress(size_t i, const VField &u, Stress &s) const {
        assert(i < m_mesh.numElements());
        auto elem  = m_mesh.element(i);
        elem->stress(elem, u, s);
    }

    // Element strain/stress by return value.
    Strain elementStrain(size_t i, const VField &u) const { Strain result; elementStrain(i, u, result); return result; }
    Stress elementStress(size_t i, const VField &u) const { Stress result; elementStress(i, u, result); return result; }

    // Strain field as a per-element interpolant
    std::vector<Strain> strainField(const VField &u) const {
        std::vector<Strain> sfield(m_mesh.numElements());
        for (size_t i = 0; i < m_mesh.numElements(); ++i)
            elementStrain(i, u, sfield[i]);
        return sfield;
    }

    // Stress field as a per-element interpolant
    std::vector<Stress> stressField(const VField &u) const {
        std::vector<Stress> sfield(m_mesh.numElements());
        for (size_t i = 0; i < m_mesh.numElements(); ++i)
            elementStress(i, u, sfield[i]);
        return sfield;
    }

    // Strain averaged over each element.
    SMField averageStrainField(const VField &u) const {
        SMField strainField(m_mesh.numElements());
        Strain s;
        for (size_t i = 0; i < m_mesh.numElements(); ++i) {
            elementStrain(i, u, s);
            strainField(i) = s.average();
        }

        return strainField;
    }

    // Stress averaged over each element.
    SMField averageStressField(const VField &u) const {
        SMField stressField(m_mesh.numElements());
        Stress s;
        for (size_t i = 0; i < m_mesh.numElements(); ++i) {
            elementStress(i, u, s);
            stressField(i) = s.average();
        }

        return stressField;
    }

    template<class _SymMat>
    VField constantStrainLoad(const _SymMat &strain) const {
        VField load(numDoFs());
        load.clear();
        typename _Mesh::ElementData::ElementLoad eLoad;
        for (auto e : m_mesh.elements()) {
            e->perElementConstantStrainLoad(strain, eLoad);
            for (auto n : e.nodes())
                load(DoF(n.index())) += eLoad.col(n.localIndex());
        }
        return load;
    }

    template<class _StressField>
    VField perElementStressFieldLoad(const _StressField &stress) const {
        BENCHMARK_START_TIMER("perElementStressFieldLoad");
        VField load(numDoFs());
        load.clear();
        typename _Mesh::ElementData::ElementLoad eLoad;
        for (auto e : m_mesh.elements()) {
            e->perElementConstantStressLoad(stress(e.index()), eLoad);
            for (size_t n = 0; n < e.numNodes(); ++n)
                load(DoF(e.node(n).index())) += eLoad.col(n);
        }
        BENCHMARK_STOP_TIMER("perElementStressFieldLoad");
        return load;
    }

    // Computes change of nodal load of the forces and tractions:
    //     div t  in volume
    //      -t n  on boundary
    // due to perturbation of the boundary with linear normal velocity vn
    // (assuming tensor field is constant wrt shape):
    //      d[vn] int_vol phi . div t dV - int_bdry phi . t n dA =
    //    - d[vn] int_vol strain(phi) : t dV =
    //    - int_bdry (strain(phi) : t) vn dA
    // This is useful for forward shape differentiation of an solution.
    // Since t is only needed on the boundary, we expect a per-boundary-element
    // interpolant.
    // If ignorePeriodic is true, no contribution from periodic boundaries is
    // accumulated.
    template<class NSVInterpolant, class BdryTensorInterpolant>
    VField changeInDivTensorLoad(const std::vector<NSVInterpolant>       &vn,
                                 const std::vector<BdryTensorInterpolant> &t,
                                 bool ignorePeriodicBdry = false) const {
        static_assert(BdryTensorInterpolant::K == K - 1,
                      "Invalid boundary tensor interpolant simplex type");
        static_assert((NSVInterpolant::K == K - 1) && (NSVInterpolant::Deg <= 1),
                      "Invalid normal shape velocity interpolant.");
        assert(vn.size() == m_mesh.numBoundaryElements());
        assert (t.size() == m_mesh.numBoundaryElements());

        VField loadChange(numDoFs());
        loadChange.clear();

        // Real totalContrib = 0, totalVN = 0, totalBdryPhiStrain = 0, totalT = 0;
        for (auto e : m_mesh.elements()) {
            if (!e.isBoundary()) continue;

            std::vector<Strain> phiStrains = e->vecPhiStrains();
            SymmetricMatrixInterpolant<SMatrix, K - 1, Strain::Deg> bdryPhiStrain;
            for (size_t n = 0; n < e.numNodes(); ++n) {
                for (size_t c = 0; c < N; ++c) {
                    const auto &volPhiStrain = phiStrains.at(n * N + c);
                    for (size_t fi = 0; fi < e.numNeighbors(); ++fi) {
                        auto f = m_mesh.boundaryElement(e.interface(fi).boundaryEntity().index());
                        if (!f) continue;
                        if (ignorePeriodicBdry && f->isInternal) continue;
                        restrictInterpolant(e, f, volPhiStrain, bdryPhiStrain);

                        auto  t_f =  t.at(f.index());
                        auto vn_f = vn.at(f.index());

                        constexpr size_t IntegrandDeg = NSVInterpolant::Deg +
                                    Strain::Deg + BdryTensorInterpolant::Deg;
                        // Subtract (strain(phi) : t vn) bdry elem contribution
                        Real contrib = Quadrature<K - 1, IntegrandDeg>::integrate(
                                [&] (const EvalPt<K - 1> &p) {
                                    return vn_f(p) * bdryPhiStrain(p).doubleContract(t_f(p));
                        }, f->volume());
                        loadChange(DoF(e.node(n).index()))[c] -= contrib;
                        // totalContrib += std::abs(contrib);
                        // totalVN += std::abs(vn_f.integrate(f->volume()));
                        // auto bphiI = bdryPhiStrain.integrate(f->volume());
                        // totalBdryPhiStrain += bphiI.doubleContract(bphiI);
                        // auto tI = t_f.integrate(f->volume());
                        // totalT += tI.doubleContract(tI);
                    }
                }
            }
        }

        // Real totalVNManual = 0;
        // for (auto be : m_mesh.boundaryElements()) {
        //     if (ignorePeriodicBdry && be->isInternal) continue;
        //     totalVNManual += std::abs(vn[be.index()].integrate(be->volume()));
        // }

        // std::cout << "loadChange stats: " << loadChange.minMag() << ", " << loadChange.maxMag() << std::endl;
        // std::cout << "total contrib: " << totalContrib << std::endl;
        // std::cout << "total VN, bphi, T: " << totalVN << ", " << totalBdryPhiStrain << ", " << totalT << std::endl;
        // std::cout << "total VN manual: " << totalVNManual;

        return loadChange;
    }

    VField solve() const { return solve(neumannLoad()); }

    ////////////////////////////////////////////////////////////////////////
    /*! Expand the reduced DoFs' values into per-node quantities
    //  @param[in]  x       DoF solution values
    //  @return     per-node displacement vector field.
    *///////////////////////////////////////////////////////////////////////
    template<class _Vec>
    VField dofToNodeField(const _Vec &x) const {
        // This also trims off lagrange multipliers, but they should be gone
        // by this point anyway.
        assert(x.size() >= N * numDoFs());

        VField f(m_mesh.numNodes());
        for (size_t i = 0; i < m_mesh.numNodes(); ++i) {
            int d = DoF(i);
            for (size_t c = 0; c < N; ++c)
                f(i)[c] = x[N * d + c];
        }
        return f;
    }

    ////////////////////////////////////////////////////////////////////////
    /*! Extract the per-vertex vertex values from a nodal vector field.
    //  @param[in]  x       per-node vector field values
    //  @return     per-vertex displacement vector field.
    *///////////////////////////////////////////////////////////////////////
    template<class _Vec>
    VField nodeToVertexField(const _Vec &x) const {
        // This also trims off lagrange multipliers, but they should be gone
        // by this point anyway.
        assert(x.size() >= N * m_mesh.numNodes());

        VField f(m_mesh.numVertices());
        for (size_t i = 0; i < m_mesh.numVertices(); ++i) {
            int n = m_mesh.vertex(i).node().index();
            for (size_t c = 0; c < N; ++c) {
                assert(N * n + c < x.size());
                f(i)[c] = x[N * n + c];
            }
        }
        return f;
    }

    // Compute the load on the DoFs from the Neumann boundary conditions.
    // (And optional per-vertex delta function forces)
    VField neumannLoad() const {
        VField load(numDoFs());
        load.clear();
        for (auto be : m_mesh.boundaryElements()) {
            for (size_t n = 0; n < be.numNodes(); ++n)
                load(DoF(be.node(n).volumeNode().index()))
                    += be->nodalNeumannLoad(n);
        }

        // Add in the delta function forces
        for (auto &ndf : m_nodalDeltaFunctionForces)
            load(DoF(ndf.first)) += ndf.second;

        return load;
    }

    // Computes a differential form that can be used with velocity fields to produce the
    // perturbation in volume that the object suffers
    OForm deltaVolumeForm() const {
        OForm result(mesh().vertices().size());

        for (unsigned v = 0; v < mesh().vertices().size(); v++) {
            result(v).fill(0.0);
        }

        for (auto e : mesh().elements()) {
            for (auto v_m : e.vertices()) {
                auto gradLam_m = e->gradBarycentric().col(v_m.localIndex());

                result(v_m.index()) += gradLam_m * e->volume();
            }
        }

        return result;
    }

    // Computes the surface area where some neumann load is acting
    Real neumannBoundaryArea() {
        // Compute boundary length/area
        double neumannArea = 0.0;
        for (auto be : m_mesh.boundaryElements()) {
            if (be->neumannTraction.norm() < 1e-15)
                continue;

            neumannArea += be->volume();
        }

        return neumannArea;
    }


    // Compute the delta of the load from the Neumann boundary conditions.
    // This load is computed differentiating the neumann load w.r.t time and for a given velocity delta_p
    VField deltaNeumannLoad(const VField &delta_p) const {
        VField load(numDoFs());
        load.clear();

        // Compute boundary length/area
        double neumannArea = 0.0;
        double intNablaDotV = 0.0;
        for (auto be : m_mesh.boundaryElements()) {
            if (be->neumannTraction.norm() < 1e-15)
                continue;

            neumannArea += be->volume();

            //std::vector<VectorND<N>> cornerPerturbations;
            double relativeDeltaVolume = 0.0;
            for (auto v : be.vertices()) {
                //cornerPerturbations.push_back(delta_p(v.volumeVertex().index()));
                auto gradLam_m = be->gradBarycentric().col(v.localIndex());
                relativeDeltaVolume += gradLam_m.dot(delta_p(v.volumeVertex().index()));
            }

            intNablaDotV += be->volume() * relativeDeltaVolume;
        }

        for (auto be : m_mesh.boundaryElements()) {
            if (be->neumannTraction.norm() < 1e-15)
                continue;

            std::vector<VectorND<N>> cornerPerturbations;
            //extractElementCornerValues(be, delta_p, cornerPerturbations); // does not work for boundary elements

            for (auto v : be.vertices())
                cornerPerturbations.push_back(delta_p(v.volumeVertex().index()));

            for (size_t n = 0; n < be.numNodes(); ++n) {
                load(DoF(be.node(n).volumeNode().index())) += be->nodalDeltaNeumannLoad(n, cornerPerturbations, neumannArea, intNablaDotV);
            }
        }

        return load;
    }

    // Compute the load on the nodes due to external forces under a particular
    // equilibrium deformation. (I.e. apply stiffness matrix: K * u). The
    // internal forces are -applyStiffnessMatrix(u)
    VField applyStiffnessMatrix(const VField &u) const {
        assert(u.domainSize() == m_mesh.numNodes());
        VField load(m_mesh.numNodes());
        load.clear();
        typename _Mesh::ElementData::PerElementStiffness Ke;
        for (auto e : m_mesh.elements()) {
            e->perElementStiffness(Ke);
            for (size_t ni = 0; ni < e.numNodes(); ++ni) {
                size_t globalni = e.node(ni).index();
                for (size_t nj = 0; nj < e.numNodes(); ++nj) {
                    size_t globalnj = e.node(nj).index();
                    for (size_t ci = 0; ci < N; ++ci) {
                        size_t row = N * ni + ci;
                        for (size_t cj = 0; cj < N; ++cj) {
                            size_t col = N * nj + cj;
                            load(globalni)[ci] += ((row < col) ? Ke(row, col) : Ke(col, row)) * u(globalnj)[cj];
                        }
                    }
                }
            }
        }
        return load;
    }

    bool   usingReducedDoFs() const { return m_dofForNode.size() == m_mesh.numNodes(); }
    size_t numDoFs()          const { return usingReducedDoFs() ? m_numDoFs : m_mesh.numNodes(); }

    // Degree of freedom tag associated with a node.
    // Note: this is only a variable index for scalar fields--for (flattened)
    // vector fields, dof i comprises variables Dim() * i...Dim() * (i + 1) - 1
    size_t DoF(int node) const {
        assert(size_t(node) < m_mesh.numNodes());
        if (usingReducedDoFs())
            return m_dofForNode[node];
        return node;
    }

    ////////////////////////////////////////////////////////////////////////
    /*! Apply the periodic boundary conditions by determing a "DOF index"
    //  for every node in the mesh. conditions. For internal nodes, these
    //  are all unique. On the periodic boundary, these will be shared by
    //  identified nodes.
    //  Updates m_dofForNode.
    *///////////////////////////////////////////////////////////////////////
    void applyPeriodicConditions(Real epsilon = 1e-7,
                                 bool ignoreMismatch = false,
                                 std::unique_ptr<PeriodicCondition<N>> pc = nullptr) {
        m_system.clear();
        if (!pc) pc = Future::make_unique<PeriodicCondition<N>>(m_mesh, epsilon, ignoreMismatch);
        m_dofForNode = pc->periodicDoFsForNodes();
        m_numDoFs = pc->numPeriodicDoFs();
        for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i)
            m_mesh.boundaryElement(i)->isInternal = pc->isPeriodicBE(i);
    }

    // In non periodic structures, it may be the case where a shape is clipped by a frame. The interior
    // of the shape can then become part of the boundary, although we don't want to consider it in
    // shape velocity computations (for example, if this region is fixed). This function sets these
    // boundary mesh elements as internal.
    void setInternalElements(BBox<VectorND<N>> cell) {
        std::vector<VectorND<N>> bdryPts;
        bdryPts.reserve(m_mesh.numBoundaryNodes());
        for (auto bn : m_mesh.boundaryNodes()) bdryPts.push_back(bn.volumeNode()->p);

        std::vector<PeriodicBoundaryMatcher::FaceMembership<N>> periodicBoundariesForBoundaryNode;
        PeriodicBoundaryMatcher::determineCellBoundaryFaceMembership(bdryPts, cell, periodicBoundariesForBoundaryNode);

        std::vector<bool> isPeriodicBoundaryElement = PeriodicBoundaryMatcher::determineCellFaceBoundaryElements(m_mesh, periodicBoundariesForBoundaryNode);

        for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i)
            m_mesh.boundaryElement(i)->isInternal = isPeriodicBoundaryElement.at(i);
    }

    void removePeriodicConditions() {
        m_system.clear();
        m_dofForNode.clear();
        for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i)
            m_mesh.boundaryElement(i)->isInternal = false;
    }

    void applyBoundaryConditions(const std::vector<CondPtr<N>> &conds) {
        // Set up evaluator environment
        ExpressionEnvironment env;
        auto mbb = m_mesh.boundingBox();
        env.setVectorValue("mesh_size_", mbb.dimensions());
        env.setVectorValue("mesh_min_", mbb.minCorner);
        env.setVectorValue("mesh_max_", mbb.maxCorner);

        size_t dirichletRegionIdx = 0;
        if (conds.size() > 0) m_system.clear();
        for (const auto &cond : conds) {
            env.setVectorValue("region_size_", cond->region->dimensions());
            env.setVectorValue("region_min_",  cond->region->minCorner);
            env.setVectorValue("region_max_",  cond->region->maxCorner);
            std::runtime_error illegalCondition("Illegal BC type");
            std::runtime_error unimplemented("Unimplemented BC type");
            std::string nonbdryMsg("Condition applied to non-boundary node ");

            if (auto nc = dynamic_cast<const NeumannCondition<N> *>(cond.get())) {
                Real regionArea = 0.0;
                std::vector<size_t> region;
                for (auto be : m_mesh.boundaryElements()) {
                    Point center(Point::Zero());
                    for (size_t c = 0; c < be.numVertices(); ++c)
                        center += be.vertex(c).volumeVertex().node()->p;
                    center /= be.numVertices();
                    if (nc->containsPoint(center)) {
                        env.setXYZ(center);
                        regionArea += be->volume();
                        region.push_back(be.index());
                        if (nc->type == NeumannType::Pressure)
                             be->neumannTraction = -nc->pressure(env) * be->normal();
                        else if (nc->type == NeumannType::Traction)
                             be->neumannTraction =  nc->traction(env);
                        else if (nc->type == NeumannType::Force) {
                            // In the Force case, "traction" is actually a
                            // force that will be distributed uniformly among all
                            // boundary elements in the region.
                            be->neumannTraction = nc->traction(env);
                        }
                        else throw unimplemented;
                    }
                }
                if (region.size() == 0)
                    throw std::runtime_error("Neumann region unmatched");
                if (nc->type == NeumannType::Force) {
                    // Actual traction for the force condition is total
                    // force (stored in neumannTraction) / region area.
                    for (size_t bei : region) {
                        m_mesh.boundaryElement(bei)->neumannTraction /= regionArea;
                    }
                }
            }
            else if ((dynamic_cast<const TargetCondition<N> *>(cond.get())) ||
                     (dynamic_cast<const TargetNodesCondition<N> *>(cond.get()))) {
                // Prevent TargetConditions from being interpreted as Dirichlet
                // conditions.
                std::cerr << "WARNING: ignoring target boundary conditions." << std::endl;
            }
            else if (auto dc = dynamic_cast<const DirichletCondition<N> *>(cond.get())) {
                ++dirichletRegionIdx;
                for (auto bn : m_mesh.boundaryNodes()) {
                    if (dc->containsPoint(bn.volumeNode()->p)) {
                        env.setXYZ(bn.volumeNode()->p);
                        bn->setDirichlet(dc->componentMask, dc->displacement(env));
                        bn->setDirichletRegion(dirichletRegionIdx);
                    }
                }
            }
            else if (auto dec = dynamic_cast<const DirichletElementsCondition<N> *>(cond.get())) {
                ++dirichletRegionIdx;
                for (auto be : m_mesh.boundaryElements()) {
                    IVectorND<N> idx;
                    for (size_t c = 0; c < be.numVertices(); ++c) {
                        idx[c] = be.vertex(c).volumeVertex().index();
                    }
                    if (dec->containsElement(idx)) {
                        for (size_t n = 0; n < be.numNodes(); ++n) {
                            env.setXYZ(be.node(n).volumeNode()->p);
                            be.node(n)->setDirichlet(dec->componentMask, dec->displacement(env));
                            be.node(n)->setDirichletRegion(dirichletRegionIdx);
                        }
                    }
                }
            }
            else if (auto nec = dynamic_cast<const NeumannElementsCondition<N> *>(cond.get())) {
                size_t numSet = 0;
                Real regionArea = 0.0;
                std::vector<size_t> forceRegion;
                for (auto be : m_mesh.boundaryElements()) {
                    UnorderedTriplet elem(
                                   be.vertex(0).volumeVertex().index(),
                                   be.vertex(1).volumeVertex().index(),
                        (N == 3) ? be.vertex(2).volumeVertex().index() : 0);
                    if (nec->hasValueForElement(elem)) {
                        const auto &val = nec->getValue(elem);
                        if (val.type == NeumannType::Pressure)
                             be->neumannTraction = -val.pressure() * be->normal();
                        else if (val.type == NeumannType::Traction)
                            be->neumannTraction =  val.traction();
                        else if (val.type == NeumannType::Force) {
                            be->neumannTraction =  val.force();
                            regionArea += be->volume();
                            forceRegion.push_back(be.index());
                        }
                        ++numSet;
                    }
                }
                if (numSet != nec->numElements())
                    throw std::runtime_error("Some element boundary conditions weren't matched.");

                // If force is described, it corresponds to the force applied to a group of
                // elements in a common region.
                for (size_t bei : forceRegion) {
                    m_mesh.boundaryElement(bei)->neumannTraction /= regionArea;
                }
            }
            else if (auto dnc = dynamic_cast<const DirichletNodesCondition<N> *>(cond.get())) {
                std::cerr << "WARNING: dirichlet region index currently not set for DirichletNodesCondition;"
                          << " region force printout will be inaccurate."
                          << std::endl;
                for (size_t i = 0; i < dnc->indices.size(); ++i) {
                    size_t ni = dnc->indices[i];
                    auto n = m_mesh.node(ni);
                    auto bn = n.boundaryNode();
                    if (!bn) throw std::runtime_error(nonbdryMsg + std::to_string(ni));
                    bn->setDirichlet(dnc->componentMask, dnc->displacements[i]);
                }
            }
            else if (auto fc = dynamic_cast<const DeltaForceCondition<N> *>(cond.get())) {
                for (auto n : m_mesh.nodes()) {
                    if (fc->containsPoint(n->p)) {
                        env.setXYZ(n->p);
                        m_nodalDeltaFunctionForces.emplace_back(n.index(), fc->force(env));
                    }
                }
            }
            else if (auto fnc = dynamic_cast<const DeltaForceNodesCondition<N> *>(cond.get())) {
                for (size_t i = 0; i < fnc->indices.size(); ++i) {
                    size_t ni = fnc->indices[i];
                    if (ni > m_mesh.numNodes()) throw std::runtime_error("DeltaForceNodesCondition node index out of bounds: " + std::to_string(ni));
                    m_nodalDeltaFunctionForces.emplace_back(ni, fnc->forces[i]);
                }
            }
            else throw illegalCondition;
        }
    }

    void removeDirichletConditions() {
        int removeCount = 0;
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            if (bn->hasDirichlet()) {
                bn->dirichletComponents.clear();
                ++removeCount;
            }
        }
        if (removeCount > 0)
            m_system.clear();
    }

    void removeNeumanConditions() {
        for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i)
            m_mesh.boundaryElement(i)->neumannTraction = Point::Zero();
    }

    void removeAllBoundaryConditions() {
        removeNeumanConditions();
        removeDirichletConditions();
    }

    void applyNoRigidMotionConstraint() {
        if (!m_useRigidMotionConstraint ||
             m_rigidMotionConstraintRHS.size() != 0) {
            m_rigidMotionConstraintRHS.clear();
            m_system.clear();
            m_useRigidMotionConstraint = true;
        }
    }

    // Set whether the no rigid translation constraint should be implemented
    // using a node pinning constraint.
    void setUsePinNoRigidTranslationConstraint(bool use) {
        m_useNRTPinConstraint = use;
    }

    // Apply a constraint to match the rigid motion of u
    // This is the same as the no rigid motion constraint, but with a RHS
    // given by the product R * u
    void applyRigidMotionConstraint(const VField &u) {
        applyNoRigidMotionConstraint();
        // Currently we must rebuild the system--in the future, we should
        // support rebuilding the constraint RHS without
        // rebuilding/factoring the system matrix.
        m_system.clear();
        getRigidInnerProduct(u, m_rigidMotionConstraintRHS);
    }

    void removeNoRigidMotionConstraint() {
        if (m_useRigidMotionConstraint) {
            m_system.clear();
            m_useRigidMotionConstraint = false;
        }
    }

    // Cannot currently be undone!
    void applyPeriodicPairDirichletConditions(std::vector<PeriodicPairDirichletCondition<N>> &pps) {
        for (auto &pp : pps) {
            std::pair<size_t, size_t> p = pp.pair(m_mesh);
            m_mesh.boundaryNode(p.first)->setDirichlet(pp.component(), VectorND<N>::Zero());
            m_mesh.boundaryNode(p.second)->setDirichlet(pp.component(), VectorND<N>::Zero());
        }
    }

    void applyTranslationPins(const ComponentMask &c) {
        assert(m_mesh.numBoundaryNodes() > 0);
        for (size_t d = 0; d < N; ++d) {
            if (!c.has(d)) continue;
            // Pin to zero the dth translation component of the
            // minimally-positioned node along direction d.
            auto bnMin = m_mesh.boundaryNode(0);
            for (auto bn : m_mesh.boundaryNodes()) {
                if (bn.volumeNode()->p[d] < bnMin.volumeNode()->p[d])
                    bnMin = bn;
            }
            ComponentMask dmask;
            dmask.set(d);
            assert(bnMin.valid());
            bnMin->setDirichlet(dmask, VectorND<N>::Zero());
        }
    }

    // Compute R * u. This is useful for computing a no-rigid-motion right
    // hand side that is compatible with a particular Dirichlet solution.
    void getRigidInnerProduct(const VField &u, std::vector<Real> &innerProduct) const {
        TMatrix R;
        m_assembleRigidModeMatrix(R);
        assert(R.n == N * numDoFs());
        assert(u.domainSize() == numDoFs());

        // Compute row norm and inner product;
        innerProduct.assign(R.m, 0.0);
        for (size_t i = 0; i < R.nnz(); ++i) {
            const auto &nz = R.nz[i];
            innerProduct.at(nz.i) += nz.v * u[nz.j];
        }
    }

    // Remove the rigid transform component from a per-DoF vector field.
    // v = v - sum_i (R(i, :) * v) * R(i, :)' / ||R(i, :)||^2;
    // If dofMask is passed then nodes i for which dofMask[i] is false are
    // ignored. This allows rigid motion in a vector field over only a subset of
    // the object to be projected out (originally I thought this was needed for
    // local/global material optimization--maybe it's not so useful).
    void projectOutRigidComponent(VField &v,
            const std::vector<bool> &dofMask = std::vector<bool>()) const {
        assert(v.domainSize() == numDoFs());
        bool hasDofMask = dofMask.size() == numDoFs();
        TMatrix R;
        // Note: rows of rigid mode matrix are orthogonal, but not
        // normalized.
        m_assembleRigidModeMatrix(R);
        assert(R.n == N * numDoFs());
        assert(v.domainSize() == numDoFs());

        // Note: the following operations assume the rigid mode matrix has
        // no repeated indices.

        // Compute row norm and inner product;
        std::vector<Real> rowSqNorms(R.m, 0.0), innerProduct(R.m, 0.0);
        for (size_t i = 0; i < R.nnz(); ++i) {
            const auto &nz = R.nz[i];
            if (hasDofMask && dofMask.at(nz.j / N)) continue;
            rowSqNorms.at(nz.i)   += nz.v * nz.v;
            innerProduct.at(nz.i) += nz.v * v[nz.j];
        }

        // Subtract off projection onto rigid transform basis
        for (size_t i = 0; i < R.nnz(); ++i) {
            const auto &nz = R.nz[i];
            if (hasDofMask && dofMask.at(nz.j / N)) continue;
            v[nz.j] -= innerProduct[nz.i] * nz.v / rowSqNorms[nz.i];
        }
    }

    // If not enough Dirichlet conditions are applied, or if some components
    // aren't constrained, we may need to add partial no-rigid-motion
    // constraints to make the problem well-posed.
    void analyzeDirichletPosedness(ComponentMask &needsTranslations,
                                   ComponentMask &needsRotations) const {
        std::vector<size_t> counts(N, 0);
        needsTranslations.set();
        size_t totalConstrained = 0;
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            for (size_t c = 0; c < N; ++c) {
                if (bn->dirichletComponents.has(c)) {
                    ++counts[c]; ++totalConstrained;
                    needsTranslations.clear(c);
                }
            }
        }
        needsRotations.clear();
        if (totalConstrained == 0) needsRotations.set();
        else if (needsTranslations.hasAny(N) || (totalConstrained < ((N == 2) ? 3 : 6))) {
            std::cerr << "WARNING: analysis of partial Dirichlet rotational posedness not yet implemented!"
                << std::endl;
            std::cerr << "Unconstrained translation components: " << needsTranslations.componentString() << std::endl;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Build up the components of the constrained system.
    //  @param[out] Ktrip           unconstrained stiffness matrix
    //  @param[out] constraintRows  arbitrary linear constraints on variables
    //  @param[out] constraintRHS   RHS for those arbitrary constraints
    //  @param[out] fixedVars       indices of vars to fix at specified values
    //                              (i.e. for Dirichlet constraints).
    //  @param[out] fixedVarValues  the values variables are fixed to.
    *///////////////////////////////////////////////////////////////////////////
    void assembleConstrainedSystem(TMatrix &Ktrip, TMatrix &constraintRows,
            std::vector<Real> &constraintRHS,
            std::vector<size_t> &fixedVars,
            std::vector<Real>   &fixedVarValues,
            bool allowIllPosed = false) const {
        BENCHMARK_START_TIMER("Assemble System");
        m_assembleStiffnessMatrix(Ktrip);

        constraintRows.clear();
        constraintRHS.clear();
        fixedVars.clear();
        fixedVarValues.clear();

        TMatrix R;
        if (m_useRigidMotionConstraint) {
            m_appendInfinitesimalRotationMatrix(R); // NO RIGID ROTATIONS
            if (m_useNRTPinConstraint) m_pinNode(fixedVars, fixedVarValues);
            else                       m_appendTranslationMatrix(R);

            // We do a rigid-motion = 0 constraint if no RHS is supplied
            // Note: if a RHS was supplied but m_useNRTPinConstraint is true, an
            // error will be thrown if the RHS has a translation part.
            constraintRHS = m_rigidMotionConstraintRHS;
            if (constraintRHS.size() == 0) constraintRHS.assign(R.m, 0);
            if (constraintRHS.size() != R.m)
                throw std::runtime_error("Invalid rigid motion RHS");
        }
        else if (!allowIllPosed) {
            ComponentMask needsTranslations, needsRotations;
            analyzeDirichletPosedness(needsTranslations, needsRotations);
            if (needsTranslations.hasAny(N)) {
                if (m_useNRTPinConstraint) {
                    m_pinNode(fixedVars, fixedVarValues, needsTranslations);
                }
                else {
                    m_appendTranslationMatrix(R, needsTranslations);
                    constraintRHS.assign(needsTranslations.count(N), 0);
                }
            }
            if (needsRotations.hasAny(N)) throw std::runtime_error("Unimplemented");
        }

        constraintRows = R;

        // TODO: test by fixing variables in batches.
        m_getDirichletVarsAndValues(fixedVars, fixedVarValues);

        BENCHMARK_STOP_TIMER("Assemble System");
    }

    void reportRegionSurfaceForces(const VField &u) const {
        VField f = applyStiffnessMatrix(u);
        std::vector<VectorND<N>> forces;
        for (size_t bni = 0; bni < m_mesh.numBoundaryNodes(); ++bni) {
            auto bn = m_mesh.boundaryNode(bni);

            // Start new region integral if needed
            size_t ri = bn->dirichletRegionIdx;
            if (ri + 1 > forces.size())
                forces.resize(ri + 1, VectorND<N>::Zero());
            forces[ri] += f(bn.volumeNode().index());
        }

        for (size_t i = 0; i < forces.size(); ++i) {
            std::cout << "region " << i << " force:";
            for (size_t j = 0; j < N; ++j)
                std::cout << "\t" << forces[i][j];
            std::cout << std::endl;
        }
    }

    void dumpSystem(const std::string &path) const {
        if (!m_system.isSet()) m_buildConstrainedSystem();
        // side effect: sums and sorts nonzeros in system--ok since m_system is
        // mutable.
        m_system.sumAndDumpUpper(path);
    }

    // (re-)embed the mesh elements.
    template<typename Vertices>
    void updateMeshNodePositions(const Vertices &vertices) {
        m_mesh.setNodePositions(vertices);
        m_system.clear();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Discrete shape derivative functions
    ////////////////////////////////////////////////////////////////////////////
    template<class ElementHandle, typename PerVertexField, typename T>
    void extractElementCornerValues(const ElementHandle &e, const PerVertexField &f,
                                   std::vector<T> &cornerValues) const {
        cornerValues.clear();
        for (auto v : e.vertices())
            cornerValues.push_back(f(v.index()));
    }

    // Change in the force for fixed nodal displacement field u due to mesh
    // vertex perturbation deltaP: (delta K) * u
    // WARNING: u is a per-node field, but we return a per-dof load field (this
    // distinction only matters under periodic conditions)
    VField applyDeltaStiffnessMatrix(const VField &u, const VField &deltaP) const {
        assert(u.domainSize() == m_mesh.numNodes());
        assert(deltaP.domainSize() == m_mesh.numVertices());

        VField load(numDoFs());
        load.clear();
        typename _Mesh::ElementData::PerElementStiffness dKe;
        std::vector<VectorND<N>> elem_deltaP;
        for (auto e : m_mesh.elements()) {
            extractElementCornerValues(e, deltaP, elem_deltaP);
            e->deltaPerElementStiffness(elem_deltaP, dKe);
            // Compute the effective traction load on each boundary vertex
            for (auto ni : e.nodes()) {
                size_t globaldi = DoF(ni.index()); // DoF not Node index!
                for (auto nj : e.nodes()) {
                    size_t globalnj = nj.index();  // Node not DoF index!
                    for (size_t ci = 0; ci < N; ++ci) {
                        size_t row = N * ni.localIndex() + ci;
                        for (size_t cj = 0; cj < N; ++cj) {
                            size_t col = N * nj.localIndex() + cj;
                            load(globaldi)[ci] += ((row < col) ? dKe(row, col) : dKe(col, row)) * u(globalnj)[cj];
                        }
                    }
                }
            }
        }
        return load;
    }

    // Change in the constant strain load due to mesh vertex perturbation deltaP.
    template<class _SymMat>
    VField deltaConstantStrainLoad(const _SymMat &cstrain, const VField &deltaP) const {
        assert(deltaP.domainSize() == m_mesh.numVertices());

        VField dload(numDoFs());
        dload.clear();

        typename _Mesh::ElementData::ElementLoad deLoad;
        std::vector<VectorND<N>> elem_deltaP;
        for (auto e : m_mesh.elements()) {
            extractElementCornerValues(e, deltaP, elem_deltaP);
            e->deltaPerElementConstantStrainLoad(elem_deltaP, cstrain, deLoad);
            for (size_t n = 0; n < e.numNodes(); ++n)
                dload(DoF(e.node(n).index())) += deLoad.col(n);
        }
        return dload;
    }

    // Change in per-element strain field: (delta strain)(u) + strain(delta u)
    std::vector<Strain> deltaStrainField(const VField &u, const VField &deltaU, const VField &deltaP) const {
        std::vector<Strain> dsfield(m_mesh.numElements());
        Strain strainDelta;
        std::vector<VectorND<N>> elem_deltaP;
        for (auto e : m_mesh.elements()) {
            extractElementCornerValues(e, deltaP, elem_deltaP);
            e->deltaStrain(elem_deltaP, e,      u, dsfield[e.index()]);
            e->     strain(             e, deltaU,        strainDelta);
            dsfield[e.index()] += strainDelta;
        }
        return dsfield;
    }

    // Change in strain averaged over each element: (delta strain)(u) + strain(delta u)
    SMField deltaAverageStrainField(const VField &u, const VField &deltaU, const VField &deltaP) const {
        SMField dsfield(m_mesh.numElements());
        Strain s;
        std::vector<VectorND<N>> elem_deltaP;
        for (auto e : m_mesh.elements()) {
            extractElementCornerValues(e, deltaP, elem_deltaP);
            e->deltaStrain(elem_deltaP, e,      u, s), dsfield(e.index())  = s.average();
            e->     strain(             e, deltaU, s), dsfield(e.index()) += s.average();
        }
        return dsfield;
    }

private:
    void m_buildConstrainedSystem() const {
        TMatrix Ktrip, C;
        std::vector<Real> constraintRHS;
        std::vector<size_t> fixedVars;
        std::vector<Real>   fixedVarValues;
        assembleConstrainedSystem(Ktrip, C, constraintRHS, fixedVars, fixedVarValues);
#ifdef USE_LAGRANGE_MULTIPLIERS
            C.m += fixedVars.size();
            for (size_t i = 0; i < fixedVars.size(); ++i) {
                size_t vi = fixedVars[i];
                C.addNZ(vi, vi, 1.0);
                constraintRHS.push_back(fixedVarValues[i]);
            }
            fixedVars.clear();
            fixedVarValues.clear();
        }
#endif // USE_LAGRANGE_MULTIPLIERS
        BENCHMARK_START_TIMER_SECTION("Set System");
        m_system.setConstrained(Ktrip, C, constraintRHS);
        BENCHMARK_STOP_TIMER_SECTION("Set System");
        BENCHMARK_START_TIMER_SECTION("Fix Variables");
        m_system.fixVariables(fixedVars, fixedVarValues);
        BENCHMARK_STOP_TIMER_SECTION("Fix Variables");

        // We promise not to modify the system after solving without rebuilding
        // it from scratch--save some memory.
        m_system.setEconomyMode(true);
    }

public:
    // Build *upper triangle* of stiffness matrix
    void m_assembleStiffnessMatrix(TMatrix &Ktrip) const {
        typedef typename _Mesh::ElementData::PerElementStiffness PerElementStiffness;
        constexpr size_t KeSize = PerElementStiffness::RowsAtCompileTime;
        const size_t nelem = m_mesh.numElements();
        const size_t n = N * numDoFs();

        auto accumToSparseMatrix = [&](size_t ei, const PerElementStiffness &Ke, TMatrix &_K) {
            auto elem = m_mesh.element(ei);
            constexpr size_t nNodes = Mesh::ElementData::nNodes;
            for (size_t i = 0; i < nNodes; ++i) {
                int di = DoF(elem.node(i).index());
                for (size_t j = 0; j < nNodes; ++j) {
                    int dj = DoF(elem.node(j).index());
                    if (di > dj) continue;
                    // xx, xy, xz, yx, yy, yz, zx, zy, zz
                    for (size_t ci = 0; ci < N; ++ci) {
                        for (size_t cj = 0; cj < N; ++cj) {
                            if (N * di + ci > N * dj + cj) continue;
                            int row = N * i + ci, col = N * j + cj;
                            // Only read upper triangle of symmetric Ke.
                            Real val = (row <= col) ? Ke(row, col) : Ke(col, row);
                            _K.addNZ(N * di + ci, N * dj + cj, val);
                        }
                    }
                }
            }
        };

        // Note: it's difficult to predict the nonzero count of the stiffness
        // matrix's upper triangle due to periodic DoFs. For now, allocate space
        // for the full matrix's triplets. If memory is an issue, we can
        // precompute the size by using a loop similar to the accumulation loop
        // below.
        const size_t preallocSize = KeSize * KeSize * nelem;
        Ktrip.init(n, n);
        Ktrip.reserve(preallocSize);
#if MESHFEM_WITH_TBB
        // Build all per-element matrices in parallel, then collect nonzeros
        std::vector<PerElementStiffness> elemMatrices(nelem);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, nelem),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t ei = r.begin(); ei < r.end(); ++ei)
                    m_mesh.element(ei)->perElementStiffness(elemMatrices[ei]);
            }
        );

        for (size_t i = 0; i < nelem; ++i)
            accumToSparseMatrix(i, elemMatrices[i], Ktrip);
#else
        for (size_t i = 0; i < nelem; ++i) {
            PerElementStiffness Ke;
            m_mesh.element(i)->perElementStiffness(Ke);
            accumToSparseMatrix(i, Ke, Ktrip);
        }
        // Make sure our upper bound was correct--reallocation is undesirable.
        assert(Ktrip.nnz() <= preallocSize);
#endif

    }

    // Append to dirichletVars and dirichletValues
    void m_getDirichletVarsAndValues(std::vector<size_t> &dirichletVars,
                                     std::vector<Real> &dirichletValues) const {
        // Validate and convert to per-periodic DoF constraints.
        // constraintDisplacements[i] holds the displacement to which
        // components constraintComponents[i] of DoF constraintDoFs[i] are
        // constrained.
        std::vector<Point>         constraintDisplacements;
        std::vector<int>           constraintDoFs;
        std::vector<ComponentMask> constraintComponents;
        // Index into the above arrays a DoF's constraint, or -1 for none.
        // I.e. if constraintDoFs[i] > -1, the following holds:
        //  constraintDoFs[constraintIndex[i]] = i
        std::vector<int> constraintIndex(numDoFs(), -1); // HAS SIZE EQUAL TO NUMBER OF NODES = numDoFs!
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i); // FOR EACH NODE!
            if (bn->hasDirichlet()) { // HAS DIRICHLET CONDITION ON NODE
                int dof = DoF(bn.volumeNode().index()); // USUALLY, DOF IS SAME AS NODE. IT CAN BE DIFFERENT WHEN DEALING WITH PERIODIC BOUNDARIES
                if (constraintIndex[dof] < 0) { // NO CONSTRAINT? THEN ADD NEW CONSTRAINT WITH DISPLACEMENTS
                    constraintIndex[dof] = constraintDoFs.size();
                    constraintDoFs.push_back(dof);
                    constraintDisplacements.push_back(
                            bn->dirichletDisplacement);
                    constraintComponents.push_back(
                            bn->dirichletComponents);
                }
                else { // IF ALREADY HAD CONSTRAINTS ON DOF, SPITS ERROR!!
                    std::cerr << "WARNING: Dirichlet condition on periodic "
                              << "boundary applies to all identified nodes."
                              << std::endl;
                    auto diff = bn->dirichletDisplacement -
                                constraintDisplacements[constraintIndex[dof]];
                    bool cdiffer = (bn->dirichletComponents !=
                                    constraintComponents[constraintIndex[dof]]);
                    if ((diff.norm() > 1e-10) || cdiffer) {
                        throw std::runtime_error("Mismatched Dirichlet "
                                                         "constraint on periodic DoF");
                    }
                    // Ignore redundant but compatible Dirichlet conditions.
                }
            }
        }

        for (size_t i = 0; i < constraintDoFs.size(); ++i) {
            for (size_t c = 0; c < N; ++c) {
                if (!constraintComponents[i].has(c)) continue;
                dirichletVars.push_back(N * constraintDoFs[i] + c); // constraintDoFs[i] MEANS THE NODE TO BE CONSTRAINED? WHILE dirichletVars is the variable in the linear system
                dirichletValues.push_back(constraintDisplacements[i][c]);
            }
        }
    }
private:

    static constexpr size_t numRotModes = (N == 3) ? 3 : 1;
    void m_assembleRigidModeMatrix(TMatrix &R) const {
        R.init(); // empty the matrix
        R.reserve((N + 2 * numRotModes) * m_mesh.numNodes());
        m_appendInfinitesimalRotationMatrix(R);
        m_appendTranslationMatrix(R);
    }

    // Append no rigid rotation constraint rows to R.
    void m_appendInfinitesimalRotationMatrix(TMatrix &R) const {
        if (R.n == 0) R.n = N * numDoFs(); // Make sure matrix is sized properly
        if (R.n != N * numDoFs()) throw std::runtime_error("Invalid R size.");

        // Periodic boundary conditions pin down the rotational DoFs, so we
        // only need to constrain the translational ones.
        // However, in 3D, if there's only one pair of periodic nodes,
        // there's still a remaining rotational mode around the axis
        // connecting the two nodes.
        if ((N == 2) && (numDoFs() < m_mesh.numNodes())) return;
        if (numDoFs() < m_mesh.numNodes() - 1) return;
        if (numDoFs() < m_mesh.numNodes())
            throw std::runtime_error("Single pair periodic BC unsupported in 3D.");

        size_t oldRows = R.m;
        if (N == 3) {
            R.m += numRotModes;
            for (size_t k = 0; k < m_mesh.numNodes(); ++k) {
                const auto &x = m_mesh.node(k)->p;
                // x axis infinitesimal rotation (0, -z, y)
                R.addNZ(oldRows    , N * k + 1, -x[2]);
                R.addNZ(oldRows    , N * k + 2,  x[1]);
                // y axis infinitesimal rotation (z, 0, -x)
                R.addNZ(oldRows + 1, N * k    ,  x[2]);
                R.addNZ(oldRows + 1, N * k + 2, -x[0]);
                // z axis infinitesimal rotation (-y, x, 0)
                R.addNZ(oldRows + 2, N * k    , -x[1]);
                R.addNZ(oldRows + 2, N * k + 1,  x[0]);
            }
        }
        else if (N == 2) {
            R.m += numRotModes;
            for (size_t k = 0; k < m_mesh.numNodes(); ++k) {
                const auto &x = m_mesh.node(k)->p;
                // "z axis" infinitesimal rotation (-y, x, 0)
                R.addNZ(oldRows, N * k    , -x[1]);
                R.addNZ(oldRows, N * k + 1,  x[0]);
            }
        }
        else assert(false);
    }

    // Append no rigid translation constraint rows to T.
    void m_appendTranslationMatrix(TMatrix &T,
            const ComponentMask &components = ComponentMask("xyz")) const {
        // If we've removed some degrees of freedom (e.g. by imposing
        // periodic boundary conditions), the translational constraints only
        // act on the remaining variables.
        // "components" determines which components of the DoFs are
        // constrained.
        if (T.n == 0) T.n = N * numDoFs(); // Make sure matrix is sized properly
        if (T.n != N * numDoFs()) throw std::runtime_error("Invalid T size.");
        size_t numComps = components.count(N);
        size_t oldRows = T.m, oldSize = T.nnz();
        T.m += numComps;
        T.reserve(oldSize + numComps * numDoFs());
        for (size_t i = 0; i < numDoFs(); ++i) {
            size_t row = oldRows;
            if (components.hasX())             T.addNZ(row++, N * i    , 1.0);
            if (components.hasY())             T.addNZ(row++, N * i + 1, 1.0);
            if ((N == 3) && components.hasZ()) T.addNZ(row++, N * i + 2, 1.0);
        }
        assert(T.nnz() == oldSize + numComps * numDoFs());
    }

    void m_pinNode(std::vector<size_t> &fixedVars,
                   std::vector<Real> &fixedVarValues,
                   const ComponentMask &components = ComponentMask("xyz")) const
    {
        size_t nodeToPin = m_mesh.numNodes();
        for (size_t i = 0; i < m_mesh.numNodes(); ++i) {
            // Prefer to fix an interior node
            if (!m_mesh.node(i).boundaryNode()) {
                nodeToPin = i;
                break;
            }
        }
        // But fix a boundary vertex if necessary
        if (nodeToPin == m_mesh.numNodes())
            nodeToPin = 0;

        assert(nodeToPin < m_mesh.numNodes());
        for (size_t d = 0; d < N; ++d) {
            if (components.has(d)) {
                fixedVars.push_back(N * DoF(nodeToPin) + d);
                fixedVarValues.push_back(0.0);
            }
        }
    }

    // Dirichlet constraint matrix is put in D
    // Dirichlet constraint RHS is appended to rhs
    void m_assembleDirichletConstraint(TMatrix &D, std::vector<Real> &rhs) const {
        std::vector<size_t> dirichletVars;
        std::vector<Real> dirichletValues;
        m_getDirichletVarsAndValues(dirichletVars, dirichletValues);

        assert((D.m == 0) && (D.n == 0)); // just checking...
        D.init(dirichletVars.size(), N * numDoFs());
        rhs.reserve(rhs.size() + dirichletVars.size());
        for (size_t i = 0; i < dirichletVars.size(); ++i) {
            D.addNZ(i, dirichletVars[i], 1.0);
            rhs.push_back(dirichletValues[i]);
        }
    }

    // Note: a "DoF" here is actually vector-valued--there are actualy
    // N * m_numDoFs variables in the elastostatic equation.
    size_t m_numDoFs = 0;
    std::vector<size_t> m_dofForNode;

    bool m_useRigidMotionConstraint;
    std::vector<Real> m_rigidMotionConstraintRHS;
    // Pin a single node (or a subset of its components) with direct elimination
    // instead of using a Lagrange multiplier-based no rigid translation
    // constraint.
    bool m_useNRTPinConstraint;

    // Optional per-node delta function forces to be added directly to the
    // load vector; stored in a sparse format to avoid wasting space if unused.
    std::vector<std::pair<size_t, VectorND<N>>> m_nodalDeltaFunctionForces;

protected:
    // m_system implements caching of system matrices for multiple solves.
    // It should be mutable because building and solving the system doesn't
    // affect user-visible state.
    mutable SPSDSystem<Real> m_system;

    _Mesh m_mesh;
};

// Convenience wrapper for homogenous linear elastic simulator
template<size_t _K, size_t _Deg>
using HomogenousSimulator = Simulator<HomogenousMesh<_K, _Deg>>;

}

#endif /* end of include guard: NEWLINEARELASTICITY_HH */
