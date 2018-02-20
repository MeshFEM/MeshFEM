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
void restrictInterpolant(DomainHandle dh, SubdomainHandle sdh,
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

#endif /* end of include guard: INTERPOLANTRESTRICTION_HH */
