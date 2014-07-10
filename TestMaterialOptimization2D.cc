#include "BoundaryConditions.hh"
#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
#include "MaterialOptimization.hh"
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;
using namespace MaterialOptimization2D;

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    vector<MeshIO::IOVertex>  inVertices;
    vector<MeshIO::IOElement> inTris;
    if (argc < 3) {
        std::cout << "usage: ./TestMaterialOptimization2D mesh bc" << std::endl;
        exit(-1);
    }

    string mshPath(argv[1]), condPath(argv[2]);
    bool noRigidMotion;
    auto bconds = readBoundaryConditions<Vector2D>(condPath, noRigidMotion);

    load(mshPath, inVertices, inTris, MeshIO::FMT_GUESS,
         MeshIO::MESH_TRI);
    shared_ptr<IsotropicField> matField(new IsotropicField(inTris.size()));
    typedef Optimizer<IsotropicMaterial> Opt;
    Opt matOpt(inTris, inVertices, matField, bconds, noRigidMotion);

    typedef typename Opt::SField  SField;
    auto u = matOpt.currentDisplacement();

    MSHFieldWriter writer("mtest.msh", matOpt.mesh());
    writer.addField("u", u, MSHFieldWriter::PER_NODE);

    size_t numElements = matOpt.mesh().numElements();
    SField gradE(numElements), gradNu(numElements);

    std::vector<Real> g = matOpt.objectiveGradient(u);
    assert(g.size() == 2 * numElements);
    for (size_t i = 0; i < numElements; ++i) {
        gradE[i]  = g[2 * i + 0];
        gradNu[i] = g[2 * i + 1];
    }

    writer.addField("gradE" , gradE , MSHFieldWriter::PER_ELEMENT);
    writer.addField("gradNu", gradNu, MSHFieldWriter::PER_ELEMENT);

    return 0;
}
