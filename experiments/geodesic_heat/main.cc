////////////////////////////////////////////////////////////////////////////////
// main.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements the Geodesics in Heat paper (Crane et al. 2014).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/09/2016 15:11:57
////////////////////////////////////////////////////////////////////////////////
#include <iostream>

using Real = double;

#include <SparseMatrices.hh>
#include "../../Laplacian.hh"
#include "../../MeshIO.hh"
#include "../../FEMMesh.hh"
#include "../../MSHFieldWriter.hh"
#include "../../MassMatrix.hh"
#include <Fields.hh>

using namespace std;

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        cerr << "usage: geodesic_heat mesh.msh soln.msh" << endl;
        exit(-1);
    }

    string inPath = argv[1];
    string solnPath = argv[2];

    vector<MeshIO::IOVertex > vertices;
    vector<MeshIO::IOElement> elements;
    MeshIO::load(inPath, vertices, elements);

    FEMMesh<2, 1, Point3D> mesh(elements, vertices);
    const size_t nTri = mesh.numElements();


    Real t = 0, c = 5.0;
    for (auto tri : mesh.elements()) t += tri->volume();
    t *= c / nTri;

    // Construct backward Euler system matrix
    // d/dt u = Laplacian u
    // ==> (u_t - u_0) / t = Laplacian u
    // ==> M (u_t - u_0) / t = -L u_t    (FEM gives negative Laplacian)
    // ==> (M + t L) u_t = M u_0
    // ==> A u_t = M u_0
    // where A = (M + t L)
    auto L = Laplacian::construct(mesh);
    auto M = MassMatrix::construct(mesh);
    auto A = L;
    for (auto & entry : A.nz) { entry.v *= t; }
    for (auto &mentry : M.nz) { A.nz.push_back(mentry); }

    SPSDSystem<Real> timeStepper(A);
    // Specify source vertex
    vector<size_t> fixedVars(1, 0);
    vector<Real> fixedVarValues(1, 1.0);
    timeStepper.fixVariables(fixedVars, fixedVarValues);

    // Solve time stepping equation (zero rhs)
    auto v_t = timeStepper.solve(vector<Real>(A.m));

    // Compute normalized linear FEM gradients
    // X = -grad v_t / || grad v_t ||
    VectorField<Real, 3> X(nTri);
    for (auto tri : mesh.elements()) {
        Vector3D g(Vector3D::Zero());
        for (auto vert : tri.vertices())
            g += v_t.at(vert.index()) * tri->gradPhi(vert.localIndex())[0];
        X(tri.index()) = -g / g.norm();
    }

    // Solve the Poisson equation
    // -laplacian phi = -div X  in Omega
    //  dphi/dn = X . n         on dOmega
    // ==> L phi = b
    SPSDSystem<Real> poissonSystem(L);
    vector<Real> b(L.m, 0.0);
    for (auto tri : mesh.elements()) {
        for (auto vert : tri.vertices()) {
            b.at(vert.index()) += X(tri.index()).dot(
                    tri->gradPhi(vert.localIndex()).integrate(tri->volume()));
        }
    }

    fixedVarValues.assign(1, 0.0);
    poissonSystem.fixVariables(fixedVars, fixedVarValues);
    auto soln = poissonSystem.solve(b);

    MSHFieldWriter writer(solnPath, vertices, elements);
    writer.addField("geodesic", ScalarField<Real>(soln), DomainType::PER_NODE);
    writer.addField("heat",     ScalarField<Real>(v_t),  DomainType::PER_NODE);
    writer.addField("X",        X,                       DomainType::PER_ELEMENT);

    // Compute gradient of geodesic
    VectorField<Real, 3> grad_geodesic(nTri);
    grad_geodesic.clear();
    for (auto tri : mesh.elements()) {
        for (auto vert : tri.vertices())
            grad_geodesic(tri.index()) += soln.at(vert.index()) *
                                          tri->gradPhi(vert.localIndex())[0];
    }

    writer.addField("grad geodesic", grad_geodesic, DomainType::PER_ELEMENT);

    return 0;
}
