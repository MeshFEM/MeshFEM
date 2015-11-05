////////////////////////////////////////////////////////////////////////////////
// bc_debug.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Helps debug boundary conditions by displaying the vertices matched by
//		dirichlet constraints and the traction acting on the boundary faces.
//
//		Note: because of the difficulty of outputting boundary element fields,
//		the traction is reported as a volume element field. The reported
//		quantity is taken from an boundary face in the element.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/19/2014 21:14:00
////////////////////////////////////////////////////////////////////////////////
#include "../Types.hh"
#include "../MaterialOptimization.hh"
#include "../MSHFieldWriter.hh"
#include <iostream>

using namespace std;

template<size_t _N>
void execute(const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements,
             const string &bcPath, const string &outMSH) {
    // Generate a material optimization simulator so that we can also display
    // target displacement conditions.
    typedef MaterialOptimization::Mesh<_N, 1, Materials::Isotropic> Mesh;
    typedef typename MaterialOptimization::Simulator<Mesh> Simulator;
    typedef typename Simulator::MField MField;
    shared_ptr<MField> matField(new MField(inElements.size()));
    Simulator sim(inElements, inVertices, matField);
    typedef ScalarField<Real> SField;
    typedef VectorField<Real, _N> VField;

    bool noRigidMotion;
    auto bconds = readBoundaryConditions<_N>(bcPath, sim.mesh().boundingBox(), noRigidMotion);
    sim.applyBoundaryConditions(bconds);
    if (noRigidMotion) sim.applyNoRigidMotionConstraint();

    auto mesh = sim.mesh();

    SField dirichletType(mesh.numNodes());
    for (size_t i = 0; i < mesh.numNodes(); ++i) {
        auto bn = mesh.node(i).boundaryNode();
        size_t val = 0;
        if (bn) {
            for (size_t c = 0; c < _N; ++c) {
                val <<= 1;
                val |= bn->dirichletComponents.has(c);
            }
        }
        dirichletType(i) = val;
    }

    SField targetType(mesh.numNodes());
    for (size_t i = 0; i < mesh.numNodes(); ++i) {
        auto bn = mesh.node(i).boundaryNode();
        size_t val = 0;
        if (bn) {
            for (size_t c = 0; c < _N; ++c) {
                val <<= 1;
                val |= bn->targetComponents.has(c);
            }
        }
        targetType(i) = val;
    }

    VField traction(mesh.numElements());
    for (size_t i = 0; i < mesh.numElements(); ++i) {
        auto e = mesh.element(i);
        VectorND<_N> val = VectorND<_N>::Zero();
        for (size_t f = 0; f < e.numNeighbors(); ++f) {
            auto be = mesh.boundaryElement(e.interface(f).boundaryEntity().index());
            if (be)
                val = be->neumannTraction;
        }
        traction(i) = val;
    }

    MSHFieldWriter writer(outMSH, sim.mesh());
    writer.addField("dirichletType", dirichletType, DomainType::PER_NODE);
    writer.addField("targetType",    targetType,    DomainType::PER_NODE);
    writer.addField("traction",      traction,      DomainType::PER_ELEMENT);
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    if (argc != 4) {
        cerr << "usage: bc_debug mesh bc out.msh" << endl;
        exit(-1);
    }

    vector<MeshIO::IOVertex>  inVertices;
    vector<MeshIO::IOElement> inElements;
    string meshPath(argv[1]);
    string   bcPath(argv[2]);
    string  outPath(argv[3]);

    auto type = load(meshPath, inVertices, inElements, MeshIO::FMT_GUESS,
                     MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    // Look up and run appropriate simulation instantiation.
    auto exec = (dim == 3) ? execute<3> : execute<2>;

    exec(inVertices, inElements, bcPath, outPath);

    return 0;
}
