#include "BoundaryConditions.hh"
#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
#include "MSHFieldParser.hh"
#include "Materials.hh"
#include "MaterialField.hh"
#include "MaterialOptimization.hh"
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>

using namespace std;
using namespace MaterialOptimization3D;

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    vector<MeshIO::IOVertex>  inVertices;
    vector<MeshIO::IOElement> inTets;
    if (argc < 3) {
        std::cout << "usage: ./TestMaterialOptimization2D mesh bc [numIters regularizationWeight]" << std::endl;
        exit(-1);
    }

    size_t iterations = (argc > 3) ? std::stoi(argv[3]) : 1;
    Real regularizationWeight = (argc > 4) ? std::stod(argv[4]) : 0.0;

    string mshPath(argv[1]), condPath(argv[2]);
    bool noRigidMotion;
    auto bconds = readBoundaryConditions<Vector3D>(condPath, noRigidMotion);

    typedef Optimizer<IsotropicMaterial> Opt;
    typedef typename Opt::SField  SField;
    typedef typename Opt::VField  VField;

    load(mshPath, inVertices, inTets, MeshIO::FMT_GUESS,
         MeshIO::MESH_TET);

    // Read in tet->hex association
    MSHFieldParser<3> fieldParser(mshPath);
    SField hex_index = fieldParser.scalarField("hex_index");
    vector<size_t> matIdxForElement(inTets.size());
    for (size_t i = 0; i < inTets.size(); ++i)
        matIdxForElement.at(i) = (size_t) round(hex_index[i]);

    shared_ptr<IsotropicField> matField(
            new IsotropicField(inTets.size(), matIdxForElement));

    Opt matOpt(inTets, inVertices, matField, bconds, noRigidMotion);

    VField targetDisplacements(matOpt.mesh().numNodes());
    targetDisplacements.clear();
    for (size_t i = 0; i < matOpt.mesh().numBoundaryNodes(); ++i) {
        auto bn = matOpt.mesh().boundaryNode(i);
        if (bn->hasTarget) {
            targetDisplacements(bn.volumeVertex().index()) = bn->targetDisplacement;
        }
    }

    MSHFieldWriter writer("mtest.msh", matOpt.mesh());
    writer.addField("target", targetDisplacements, MSHFieldWriter::PER_NODE);

    auto u = matOpt.currentDisplacement();
    writer.addField("Initial u", u, MSHFieldWriter::PER_NODE);

    size_t numElements = matOpt.mesh().numElements();
    SField gradE(numElements), gradNu(numElements);

    std::vector<Real> g = matOpt.objectiveGradient(u);
    matField->writeVariableFields(writer, "Initial ");
    matField->writeVariableFields(writer, "Initial grad", g);

    std::cout << "Attempting optimization" << std::endl;
    matOpt.run(writer, iterations, regularizationWeight);

    auto u_opt = matOpt.currentDisplacement();
    g = matOpt.objectiveGradient(u_opt);

    writer.addField("Final u", u_opt, MSHFieldWriter::PER_NODE);
    matField->writeVariableFields(writer, "Final ");
    matField->writeVariableFields(writer, "Final grad", g);

    return 0;
}
