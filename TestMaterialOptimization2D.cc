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
        std::cout << "usage: ./TestMaterialOptimization2D mesh bc [numIters regularizationWeight]" << std::endl;
        exit(-1);
    }

    size_t iterations = (argc > 3) ? std::stoi(argv[3]) : 1;
    Real regularizationWeight = (argc > 4) ? std::stod(argv[4]) : 0.0;

    string mshPath(argv[1]), condPath(argv[2]);
    bool noRigidMotion;
    auto bconds = readBoundaryConditions<Vector2D>(condPath, noRigidMotion);

    // MSHFieldParser<2> parser(mshPath);

    // auto initialYoung = parser.scalarField("young");
    // auto initialPoisson = parser.scalarField("poisson");

    load(mshPath, inVertices, inTris, MeshIO::FMT_GUESS,
         MeshIO::MESH_TRI);
    shared_ptr<IsotropicField> matField(new IsotropicField(inTris.size()));

    // assert(initialYoung.domainSize() == initialPoisson.domainSize());
    // assert(initialYoung.domainSize() == matField->numMaterials());
    // for (size_t i = 0; i < matField->numMaterials(); ++i) {
    //     matField->material(i).vars[0] = initialYoung[i];
    //     matField->material(i).vars[1] = initialPoisson[i];
    // }

    typedef Optimizer<IsotropicMaterial> Opt;
    Opt matOpt(inTris, inVertices, matField, bconds, noRigidMotion);

    typedef typename Opt::SField  SField;
    typedef typename Opt::VField  VField;

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
    writer.addField("initial u", u, MSHFieldWriter::PER_NODE);
    // auto lambda = matOpt.simulator().solveAdjoint(u);
    // writer.addField("lambda", lambda, MSHFieldWriter::PER_NODE);
    // writer.addField("e_u", matOpt.simulator().strain(u), MSHFieldWriter::PER_ELEMENT);
    // writer.addField("e_lambda", matOpt.simulator().strain(lambda), MSHFieldWriter::PER_ELEMENT);

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
