////////////////////////////////////////////////////////////////////////////////
// InterpolantRestriction.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Restricts simplex interpolants to sub-simplices of a FEMMesh. For
//      example, restricts a tetrahdron interpolant to one of its boundary
//      triangles.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/04/2015 22:01:26
////////////////////////////////////////////////////////////////////////////////
#ifndef INTERPOLANTRESTRICTION_HH
#define INTERPOLANTRESTRICTION_HH
#include <MeshFEM/Functions.hh>

////////////////////////////////////////////////////////////////////////////////
/*! Uses the underlying mesh, so currently interpolants of higher degree than
//  the mesh are unsupported. This is enforced by checking that FEM simplices
//  have at least as many nodes as the interpolants.
//
// @param[in]   dh          domain mesh entity handle
// @param[in]   sdh         subdomain mesh entity handle
// @param[in]   fdomain     input interpolant over domain
// @param[out]  fsdomain    output interpolant over subdomain
*///////////////////////////////////////////////////////////////////////////////
template<class DataType, size_t Deg, size_t DomainK, size_t SubdomainK,
         class DomainHandle, class SubdomainHandle>
void restrictInterpolant(const DomainHandle &dh, const SubdomainHandle &sdh,
                         const Interpolant<DataType,    DomainK, Deg> &fdomain,
                               Interpolant<DataType, SubdomainK, Deg> &fsdomain)
{
    // We don't support restricting interpolants of higher degree than the mesh.
    using Domain    = typename    DomainHandle::value_type;
    using Subdomain = typename SubdomainHandle::value_type;
    static_assert(Deg <= Domain::Deg,
                  "Restriction only supports interpolants of mesh degree or lower");
    static_assert(Deg <= Subdomain::Deg,
                  "Restriction only supports interpolants of mesh degree or lower");
    static_assert(Domain::K == DomainK,
                  "Domain simplex dimensions must match");
    static_assert(Subdomain::K == SubdomainK,
                  "Domain simplex dimensions must match");

    // Deg 0 interpolants are not nodal.
    if (Deg == 0) {
        fsdomain[0] = fdomain[0];
        return;
    }

    // Pick out subdomain nodal values from domain interpolant.
    // Could be optimized (traversal operations instead of brute-force search)
    for (size_t sdni = 0; sdni < fsdomain.size(); ++sdni) {
        size_t sdnvi = sdh.node(sdni).volumeNode().index();
        bool set = false;
        for (size_t dni = 0; dni < fdomain.size(); ++dni) {
            if (size_t(dh.node(dni).volumeNode().index()) == sdnvi) {
                fsdomain[sdni] = fdomain[dni];
                set = true;
                break;
            }
        }
        assert(set);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Embed an evaluation point living in a subdomain (e.g., boundary triangle)
// as an equivalent evaluation point in the containing higher-dimensional
// element. This is useful, e.g., for computing boundary integrals of integrands
// that are defined volumetrically.
////////////////////////////////////////////////////////////////////////////////
template<class SubdomainHandle, class DomainHandle>
void embedEvalPt(const SubdomainHandle &sdh, const DomainHandle &dh,
                 const EvalPt<SubdomainHandle::numVertices() - 1> &xsdomain,
                       EvalPt<   DomainHandle::numVertices() - 1> &xdomain)
{
    xdomain.fill(0);
    for (auto sv : sdh.vertices()) {
        size_t vi = sv.volumeVertex().index();
        bool set = false;
        for (auto dv : dh.vertices()) {
            if (size_t(dv.volumeVertex().index()) == vi) {
                set = true;
                xdomain[dv.localIndex()] = xsdomain[sv.localIndex()];
            }
        }
        assert(set);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Take an arbitrary volumetric integrand over some domain and restrict it to
// form an integrand over a subdomain (e.g., for computing a surface integral of
// a volumetric quantity like stress).
////////////////////////////////////////////////////////////////////////////////
template<class F, class SubdomainHandle, class DomainHandle>
auto restrictIntegrand(const F &integrand, const SubdomainHandle &sdh, const DomainHandle &dh) {
    return [&integrand, &sdh, &dh](const EvalPt<SubdomainHandle::numVertices() - 1> &x_subdomain) {
        EvalPt<DomainHandle::numVertices() - 1> x_domain;
        embedEvalPt(sdh, dh, x_subdomain, x_domain);
        return integrand(x_domain);
    };
}

#endif /* end of include guard: INTERPOLANTRESTRICTION_HH */
