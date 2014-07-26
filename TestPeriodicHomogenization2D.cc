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
using namespace LinearElasticity2D;
using namespace PeriodicHomogenization;

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
    string mshPath;
    if (argc >= 2) mshPath = std::string(argv[1]);

    load(mshPath, inVertices, inTris, MeshIO::FMT_GUESS,
         MeshIO::MESH_TRI);
    Simulator<> sim(inTris, inVertices);
    MSHFieldWriter writer("htest.msh", sim.mesh());

    std::vector<LinearElasticity2D::VField> w_ij;
    solveCellProblems(w_ij, sim, &writer);

    for (size_t i = 0; i < w_ij.size(); ++i) {
        string name("w_ij ");
        name += to_string(i);
        writer.addField(string("w_ij ") + to_string(i), w_ij[i],
                        MSHFieldWriter::PER_NODE);
    }

    ETensor Eh = homogenizedElasticityTensor(w_ij, sim);
    ETensor ETargetinv(Eh.inverse());
    // // Try to double x Young's modulus
    // ETargetinv.D(0, 0) /= 2.0;
    // ETargetinv.D(0, 1) /= 2.0;
    
    // Try to reduce all Poisson ratios
    // Scalar parameterStep = args["parameterStep"].as<double>();
    Real parameterStep = 0.1;
    cout << "Parameter step: " << parameterStep << endl;
    Real currentPoisson = -ETargetinv.D(0, 1) / ETargetinv.D(1, 1);
    Real targetPoisson = currentPoisson + parameterStep * (-0.5 - currentPoisson);
    cout << "currentPoisson, targetPoisson:\t" << currentPoisson << "\t"
         << targetPoisson << endl;

    ETargetinv.D(0, 1) = -targetPoisson * ETargetinv.D(1, 1);

    // // Try to double all Young's moduli
    // ETargetinv.D(0, 0) /= 2.0;
    // ETargetinv.D(1, 1) /= 2.0;
    // ETargetinv.D(0, 1) /= 2.0;

    SField v_n = homogenizedElasticityTensorShapeDerivative(
                        ETargetinv.inverse(), w_ij, sim);
    // auto bdry = sim.mesh().boundary();
    // VField descent(bdry.numElements());
    // for (size_t i = 0; i < bdry.numElements(); ++i)
    //     descent(i) = v_n[i] * bdry.element(i)->normal();
    // MSHFieldWriter surfaceWriter("htestSurf.msh", sim.mesh(), true);
    // surfaceWriter.addField(string("normal descent velocity"), v_n,
    //                        MSHFieldWriter::PER_ELEMENT);
    // surfaceWriter.addField(string("descent direction"), descent,
    //                        MSHFieldWriter::PER_ELEMENT);

    cout << setprecision(16);
    cout << "Homogenized elasticity tensor:" << endl;
    cout << Eh << endl << endl;;

    // cout << "Tensor Diff:" << endl << Eh - ETargetinv.inverse() << endl << endl;;
    ETensor Einv = Eh.inverse();

    // cout << "Homogenized compliance tensor:" << endl;
    // cout << Einv << endl << endl;
    Eigen::Matrix<Real, 3, 1> moduli(1.0 / Einv.diag().array());
    cout << "Approximate Young moduli:\t"  << moduli[0] << "\t" << moduli[1] << endl;
    cout << "Approximate shear modulus:\t" << moduli[2] << endl;

    cout << "v_yx, v_xy:\t" << -Einv.D(0, 1) / Einv.D(1, 1) << "\t"
                            << -Einv.D(1, 0) / Einv.D(0, 0) << endl;

    return 0;
}
