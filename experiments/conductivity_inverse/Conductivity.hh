#ifndef CONDUCTIVITY_HH
#define CONDUCTIVITY_HH

#include <FEMMesh.hh>
#include <GaussQuadrature.hh>
#include <SparseMatrices.hh>
#include <InterpolantRestriction.hh>

#include <vector>

namespace Conductivity {

template<class _FEMMesh>
TripletMatrix<> forwardProblemMatrix(const _FEMMesh &m, const std::vector<Real> &a) {
    constexpr size_t K = _FEMMesh::K;
    constexpr size_t Deg = _FEMMesh::Deg;

    assert(a.size() == m.numNodes());
    TripletMatrix<> L(m.numNodes(), m.numNodes());
    for (auto e : m.elements()) {
        Interpolant<Real, K, Deg> a_interp;
        for (size_t i = 0; i < e.numNodes(); ++i)
            a_interp[i] = a.at(e.node(i).index());
        for (size_t i = 0; i < e.numNodes(); ++i) {
            for (size_t j = 0; j < e.numNodes(); ++j) {
                Real val = Quadrature<K, Deg + 2 * (Deg - 1)>::integrate(
                    [&] (const VectorND<Simplex::numVertices(K)> &p) {
                        return (a_interp(p) * e->gradPhi(i)(p).dot(e->gradPhi(j)(p)));
                    }, e->volume());
                L.addNZ(e.node(i).index(), e.node(j).index(), val);
            }
        }
    }
    return L;
}

// Load for both forward and inverse problems
template<class _FEMMesh>
std::vector<Real> load(const _FEMMesh &m, const std::vector<Real> &f) {
    constexpr size_t K = _FEMMesh::K;
    constexpr size_t Deg = _FEMMesh::Deg;
    std::vector<Real> result(m.numNodes(), 0.0);

    assert(f.size() == m.numNodes());
    for (auto e : m.elements()) {
        Interpolant<Real, K, Deg> f_interp;
        for (size_t i = 0; i < e.numNodes(); ++i)
            f_interp[i] = f.at(e.node(i).index());
        for (size_t i = 0; i < e.numNodes(); ++i) {
            Interpolant<Real, K, Deg> phi;
            phi = 0; phi[i] = 1.0;

            result.at(e.node(i).index()) = Quadrature<K, 2 * Deg>::integrate(
                    [&] (const VectorND<Simplex::numVertices(K)> &p) {
                    return f_interp(p) * phi(p);
                }, e->volume());
        }
    }
    return result;
}

// Additional surface load for the inverse problem
template<class _FEMMesh>
std::vector<Real> surfaceLoad(const _FEMMesh &m, const std::vector<Real> &u,
                                                 const std::vector<Real> &a) {
    constexpr size_t K = _FEMMesh::K;
    constexpr size_t Deg = _FEMMesh::Deg;
    std::vector<Real> result(m.numNodes(), 0.0);

    assert(a.size() == m.numNodes());
    assert(u.size() == m.numNodes());
    for (auto e : m.elements()) {
        if (!e.isBoundary()) continue;

        Interpolant<VectorND<K>, K, Deg - 1> gradu_interp = u.at(e.node(0).index()) * e->gradPhi(0);
        for (size_t i = 1; i < e.numNodes(); ++i)
            gradu_interp += u.at(e.node(i).index()) * e->gradPhi(i);

        // Boundary element contribution
        for (size_t fi = 0; fi < e.numNeighbors(); ++fi) {
            auto f = e.interface(fi);
            auto be = m.boundaryElement(f.boundaryEntity().index());
            if (!be) continue;

            Interpolant<Real, K - 1, Deg> a_interp;
            for (size_t i = 0; i < be.numNodes(); ++i)
                a_interp[i] = a.at(be.node(i).volumeNode().index());

            Interpolant<Real, K, Deg - 1> du_dn_vol;
            for (size_t n = 0; n < gradu_interp.size(); ++n)
                du_dn_vol[n] = gradu_interp[n].dot(be->normal());

            Interpolant<Real, K - 1, Deg - 1> du_dn_surf;
            restrictInterpolant(e, be, du_dn_vol, du_dn_surf);

            for (size_t i = 0; i < be.numNodes(); ++i) {
                Interpolant<Real, K - 1, Deg> phi_i; phi_i = 0; phi_i[i] = 1.0;

                constexpr size_t QDeg = 2 * Deg + (Deg - 1);
                // Unfortunately, we only have quadrature up to deg 4
                // currently...
                Real bdryIntVal = Quadrature<K - 1, (QDeg < 5 ? QDeg : 4)>::integrate(
                    [&] (const VectorND<Simplex::numVertices(K - 1)> &p) {
                        return phi_i(p) * a_interp(p) * du_dn_surf(p);
                    }, be->volume());
                result.at(be.node(i).volumeNode().index()) += bdryIntVal;
            }
        }
    }
    return result;
}

template<class _FEMMesh>
std::vector<Real> solveForwardProblem(const _FEMMesh &m, const std::vector<Real> &a,
                                      const std::vector<Real> &f) {
    TripletMatrix<> L = forwardProblemMatrix(m, a);
    std::vector<Real> b = load(m, f);
    std::cout << "Built matrix with " << L.nnz() << " nonzeros." << std::endl;
    SPSDSystem<Real> system(L);

    // Apply Dirichlet conditions
    std::vector<size_t> fixedVars;
    for (auto bv : m.boundaryNodes())
        fixedVars.push_back(bv.volumeNode().index());
    std::vector<Real> fixedVarValues(fixedVars.size(), 0.0);

    system.fixVariables(fixedVars, fixedVarValues);

    std::vector<Real> u;
    system.solve(b, u);
    return u;
}

// Asymmetric!
template<class _FEMMesh>
TripletMatrix<> directInverseProblemMatrix(const _FEMMesh &m, const std::vector<Real> &u) {
    constexpr size_t K = _FEMMesh::K;
    constexpr size_t Deg = _FEMMesh::Deg;
    assert(u.size() == m.numNodes());

    TripletMatrix<> M(m.numNodes(), m.numNodes());
    for (auto e : m.elements()) {
        Interpolant<VectorND<K>, K, Deg - 1> gradu_interp = u.at(e.node(0).index()) * e->gradPhi(0);
        for (size_t i = 1; i < e.numNodes(); ++i)
            gradu_interp += u.at(e.node(i).index()) * e->gradPhi(i);
        
        for (size_t i = 0; i < e.numNodes(); ++i) {
            Interpolant<Real, K, Deg> phi_i; phi_i = 0; phi_i[i] = 1.0;
            for (size_t j = 0; j < e.numNodes(); ++j) {
                Interpolant<Real, K, Deg> phi_j; phi_j = 0; phi_j[j] = 1.0;
                Real volIntVal = Quadrature<K, Deg + 2 * (Deg - 1)>::integrate(
                    [&] (const VectorND<Simplex::numVertices(K)> &p) {
                        return phi_j(p) * e->gradPhi(i)(p).dot(gradu_interp(p));
                    }, e->volume());
                M.addNZ(e.node(i).index(), e.node(j).index(), volIntVal);
            }
        }

        // NOTE: the following shouldn't actually be used...

        // if (!e.isBoundary()) continue;
        // // Boundary element contribution
        // for (size_t fi = 0; fi < e.numNeighbors(); ++fi) {
        //     auto f = e.interface(fi);
        //     if (!f.isBoundary()) continue;
        //     auto be = m.boundaryElement(f.boundaryEntity().index());
        //     assert(be);

        //     Interpolant<Real, K, Deg - 1> du_dn_vol;
        //     for (size_t n = 0; n < gradu_interp.size(); ++n)
        //         du_dn_vol[n] = gradu_interp[n].dot(be->normal());

        //     Interpolant<Real, K - 1, Deg - 1> du_dn_surf;
        //     restrictInterpolant(e, be, du_dn_vol, du_dn_surf);

        //     for (size_t i = 0; i < be.numNodes(); ++i) {
        //         Interpolant<Real, K - 1, Deg> phi_i; phi_i = 0; phi_i[i] = 1.0;
        //         for (size_t j = 0; j < be.numNodes(); ++j) {
        //             Interpolant<Real, K - 1, Deg> phi_j; phi_j = 0; phi_j[j] = 1.0;

        //             constexpr size_t QDeg = 2 * Deg + (Deg - 1);
        //             // Unfortunately, we only have quadrature up to deg 4
        //             // currently...
        //             Real bdryIntVal = Quadrature<K - 1, (QDeg < 5 ? QDeg : 4)>::integrate(
        //                 [&] (const VectorND<Simplex::numVertices(K - 1)> &p) {
        //                     return phi_i(p) * phi_j(p) * du_dn_surf(p);
        //                 }, be->volume());
        //             M.addNZ(be.node(i).volumeNode().index(),
        //                     be.node(j).volumeNode().index(), -bdryIntVal);
        //         }
        //     }
        // }
    }

    return M;
}

template<class _FEMMesh>
std::vector<Real> solveDirectInverseProblem(const _FEMMesh &m, const std::vector<Real> &u,
                                            const std::vector<Real> &a, const std::vector<Real> &f, const ScalarField<Real> &r,
                                            MSHFieldWriter &writer) {
    TripletMatrix<> M = directInverseProblemMatrix(m, u);
    std::cout << "Built inverse system matrix with " << M.nnz() << " nonzeros." << std::endl;
    std::vector<Real> b = load(m, f);

    // std::vector<Real> bsurf = surfaceLoad(m, u, a);
    // for (auto bn : m.boundaryNodes())
    //     b[bn.volumeNode().index()] = bsurf[bn.volumeNode().index()];
    for (size_t i = 0; i < b.size(); ++i)
        b[i] += r[i];

    writer.addField("inverse rhs", ScalarField<Real>(b));

    M.dumpBinary("M.bin");
    UmfpackFactorizer LU(M);
    LU.factorize();

    std::vector<Real> a_inf;
    LU.solve(b, a_inf);

    writer.addField("inverse problem solved rhs", ScalarField<Real>(M.apply(a_inf)));

    return a_inf;
}

}

#endif /* end of include guard: CONDUCTIVITY_HH */
