#include "PeriodicHomogenization.hh"
#include "BoundaryConditions.hh"
#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;
using namespace PeriodicHomogenization3D;

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    vector<MESH_IO::IOVertex<Point3D> > inVertices;
    vector<MESH_IO::IOElement> inTets;
    std::string mshPath("Meshes/cylinder_cross.msh");
    if (argc >= 2) mshPath = std::string(argv[1]);

    load(mshPath, inVertices, inTets, MESH_IO::FMT_GUESS,
         MESH_IO::MESH_TET);

    Simulator<> sim(inTets, inVertices);
    MSHFieldWriter writer("htest.msh", sim.mesh());

    std::vector<LinearElasticity3D::VField> w_ij;
    solveCellProblems(w_ij, sim, &writer);
    ETensor Eh = homogenizedElasticityTensor(w_ij, sim);

    for (size_t i = 0; i < w_ij.size(); ++i) {
        string name("w_ij ");
        name += to_string(i);
        writer.addField(string("w_ij ") + to_string(i), w_ij[i],
                        MSHFieldWriter::PER_NODE);
    }

    cout << std::setprecision(16) << Eh << endl;

    return 0;
}
