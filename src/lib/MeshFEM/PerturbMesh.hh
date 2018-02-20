////////////////////////////////////////////////////////////////////////////////
// PerturbMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Perturbs the (non-periodic) mesh boundary by a prescribed velocity
//      field and solves for new internal vertex positions using a uniform
//      Laplacian.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/12/2015 14:03:28
////////////////////////////////////////////////////////////////////////////////
#ifndef PERTURBMESH_HH
#define PERTURBMESH_HH
#include <MeshFEM/UniformLaplacian.hh>

// Perturb mesh by boundary vector field h * v, leaving the periodic boundary
// vertices fixed.
// Internal vertex coordinates are solved for using a uniform graph Laplacian
// with Dirichlet boundary conditions given by the perturbed boundary vertex
// positions.
template<class _FEMMesh>
void perturbedMesh(Real h, const VectorField<Real, 2> &v, const _FEMMesh &mesh,
                   std::vector<MeshIO::IOVertex>  &outVertices,
                   std::vector<MeshIO::IOElement> &outElements) {

    std::vector<bool> isFixed(mesh.numVertices(), false);
    std::vector<VectorND<N>> fixedLoc;
    for (auto be : mesh.boundaryElements()) {
        for (size_t i = 0; i < be.numVertices(); ++i) {
            auto bvv = be.vertex(i).volumeVertex();
            size_t vi = bvv.index();
            // Vertices on the periodic boundary don't move. This must
            // override any other setting, so we do it even if the vertex
            // has been visited before.
            if (be->isInternal)
                fixedLoc[vi] = bvv.node()->p;
            else {
                // Visits from non-periodic boundary should not override.
                if (isFixed[vi]) continue;
                fixedLoc[vi] = h * v(bvv.index()) + bvv.node()->p;
            }
            isFixed[vi] = true;
        }
    }

    // Extract the fixed vertices (i.e. the boundary vertices)
    std::vector<size_t> fixedVars;
    for (size_t i = 0; i < isFixed.size(); ++i)
        if (isFixed[i]) fixedVars.push_back(i);

    // Solve for all vertex positions using uniform graph Laplacian, one
    // coordinate at a time.
    std::vector<Real> coord, zero(mesh.numVertices(), 0.0);
    outVertices.resize(mesh.numVertices());
    // TODO: implement changing the Dirichlet constraint *values* on
    // SPSDSystem without rebuilding/factorizing the system.
    SPSDSystem<Real> L = UniformLaplacian::assemble(mesh);
    for (size_t c = 0; c < N; ++c) {
        SPSDSystem<Real> L2 = L;
        std::vector<Real> fixedVarValues(fixedVars.size());
        for (size_t i = 0; i < fixedVars.size(); ++i)
            fixedVarValues[i] = fixedLoc.at(fixedVars[i])[c];
        L2.fixVariables(fixedVars, fixedVarValues);
        L2.solve(zero, coord);
        assert(coord.size() == outVertices.size());
        for (size_t i = 0; i < outVertices.size(); ++i)
            outVertices[i][c] = coord[i];
    }
    
    // Copy over the elements
    outElements.resize(mesh.numElements());
    for (auto e : mesh.elements()) {
        for (size_t i = 0; i < e.numVertices(); ++i) {
            size_t vi = e.vertex(i).index();
            assert(vi < outVertices.size());
            outElements[e.index()] = vi;
        }
    }
}

#endif /* end of include guard: PERTURBMESH_HH */
