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
        std::cout << "usage: ./TestMaterialOptimization2D mesh bc" << std::endl;
        exit(-1);
    }

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
    assert(g.size() == 2 * numElements);
    for (size_t i = 0; i < numElements; ++i) {
        gradE[i]  = g[2 * i + 0];
        gradNu[i] = g[2 * i + 1];
    }

    SField E(numElements), nu(numElements);
    std::vector<Real> vars(matField->numVars());
    matField->getVars(vars);
    assert(vars.size() == 2 * numElements);
    for (size_t i = 0; i < numElements; ++i) {
         E[i] = vars[2 * i + 0];
        nu[i] = vars[2 * i + 1];
    }

    writer.addField("initial E" , E , MSHFieldWriter::PER_ELEMENT);
    writer.addField("initial nu", nu, MSHFieldWriter::PER_ELEMENT);
    writer.addField("initial gradE" , gradE , MSHFieldWriter::PER_ELEMENT);
    writer.addField("initial gradNu", gradNu, MSHFieldWriter::PER_ELEMENT);

    std::cout << "Attempting optimization" << std::endl;
    size_t iterations = (argc > 3) ? std::stoi(argv[3]) : 1;
    matOpt.run(writer, iterations);

    matField->getVars(vars);
    assert(vars.size() == 2 * numElements);
    for (size_t i = 0; i < numElements; ++i) {
         E[i] = vars[2 * i + 0];
        nu[i] = vars[2 * i + 1];
    }

    assert(g.size() == 2 * numElements);
    auto u_opt = matOpt.currentDisplacement();
    g = matOpt.objectiveGradient(u_opt);
    for (size_t i = 0; i < numElements; ++i) {
        gradE[i]  = g[2 * i + 0];
        gradNu[i] = g[2 * i + 1];
    }

    writer.addField("final u", u_opt, MSHFieldWriter::PER_NODE);
    writer.addField("final E" , E , MSHFieldWriter::PER_ELEMENT);
    writer.addField("final nu", nu, MSHFieldWriter::PER_ELEMENT);
    writer.addField("final gradE" , gradE , MSHFieldWriter::PER_ELEMENT);
    writer.addField("final gradNu", gradNu, MSHFieldWriter::PER_ELEMENT);

    return 0;
}
