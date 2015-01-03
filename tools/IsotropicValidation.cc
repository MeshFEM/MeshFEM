////////////////////////////////////////////////////////////////////////////////
// IsotropicValidation.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Determines whether a pattern is isotropic under a given base material by
//      homogenizing rotated versions of the pattern.
//
//      Rotations are chosen randomly by picking a random axis and angle.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/02/2015 06:17:34
////////////////////////////////////////////////////////////////////////////////
#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
#include "LinearElasticity.hh"
#include "Materials.hh"
#include "PeriodicHomogenization.hh"
#include "GlobalBenchmark.hh"
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>
#include <random>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace PeriodicHomogenization;

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: PeriodicHomogenization_cli [options] mesh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("mesh",       po::value<string>(),                     "input mesh")
        ;
    po::positional_options_description p;
    p.add("mesh",                1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("material,m", po::value<string>(), "base material")
        ("degree,d",   po::value<int>()->default_value(2), "degree of finite elements")
        ("nsamples,n", po::value<int>()->default_value(50), "Number of samples to test")
        ;

    po::options_description cli_opts;
    cli_opts.add(visible_opts).add(hidden_opts);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).
                  options(cli_opts).positional(p).run(), vm);
        po::notify(vm);
    }
    catch (exception &e) {
        cout << "Error: " << e.what() << endl << endl;
        usage(1, visible_opts);
    }

    bool fail = false;
    if (vm.count("mesh") == 0) {
        cout << "Error: must specify input mesh" << endl;
        fail = true;
    }

    int d = vm["degree"].as<int>();
    if (d < 1 || d > 2) {
        cout << "Error: FEM Degree must be 1 or 2" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N>
using HMG = LinearElasticity::HomogenousMaterialGetter<Materials::Constant>::template Getter<_N>;

template<size_t _N>
Eigen::Matrix<Real, _N, _N> randRotation();

template<>
Eigen::Matrix<Real, 3, 3> randRotation<3>() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> angle(0, 2 * M_PI);

    Real theta = angle(gen), phi = angle(gen) / 2.0;
    Real rotationAngle = angle(gen) / 2.0;

    Vector3D rotationAxis(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
    return Eigen::AngleAxisd(rotationAngle, rotationAxis).matrix();
}

template<>
Eigen::Matrix<Real, 2, 2> randRotation<2>() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> angle(0, 2 * M_PI);
    return Eigen::Rotation2D<Real>(angle(gen)).matrix();
}

template<size_t _N, size_t _FEMDegree>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements) {
    auto &mat = HMG<_N>::material;
    if (args.count("material")) mat.setFromFile(args["material"].as<string>());

    typedef LinearElasticity::Mesh<_N, _FEMDegree, HMG> Mesh;
    typedef LinearElasticity::Simulator<Mesh> Simulator;
    Simulator sim(inElements, inVertices);
    typedef typename Simulator::ETensor ETensor;
    typedef typename Simulator::VField  VField;

    // The axis-aligned bounding box of the unrotated mesh is the base cell.
    // After rotating, homogenizedElasticityTensor is unable to compute the base
    // cell volume accurately since it uses the axis-aligned bounding box of the
    // rotated mesh. So we must store it here.
    Real baseCellVolume = sim.mesh().boundingBox().volume();

    BENCHMARK_START_TIMER_SECTION("Cell Problems");
    vector<VField> w_ij;
    solveCellProblems(w_ij, sim);
    BENCHMARK_STOP_TIMER_SECTION("Cell Problems");

    BENCHMARK_START_TIMER_SECTION("Compute Tensor");
    ETensor Eh = homogenizedElasticityTensor(w_ij, sim);
    BENCHMARK_STOP_TIMER_SECTION("Compute Tensor");

    cout << setprecision(16) << endl;
    cout << "Homogenized elasticity tensor:" << endl;
    cout << Eh << endl << endl;
    cout << "Testing rotations..." << endl;

    Real totalRelError = 0;
    Real totalTransformRelError = 0;
    int nSamples = args["nsamples"].as<int>();
    for (int s = 0; s < nSamples; ++s) {
        auto rot = randRotation<_N>();
        vector<MeshIO::IOVertex> rotVertices;
        for (const auto &v : inVertices)
            rotVertices.emplace_back(PointND<_N>(rot * PointND<_N>(v)));
        sim.updateMeshNodePositions(rotVertices);

        w_ij.clear();
        typedef typename Simulator::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        for (size_t i = 0; i < numStrains; ++i) {
            VField rhs(sim.constantStrainLoad(-SMatrix::CanonicalBasis(i)));
            w_ij.push_back(sim.solve(rhs));
        }
        ETensor EhRot = homogenizedElasticityTensor(w_ij, sim, baseCellVolume);
        ETensor diff = EhRot - Eh;
        Real relError = sqrt(diff.quadrupleContract(diff) / Eh.quadrupleContract(Eh));
        cout << "Tensor:" << endl << EhRot << endl;
        cout << "Rel error:\t" << relError << endl << endl;
        totalRelError += relError;

        ETensor EhTrans = Eh.orthogonalTransform(rot);
        diff = EhTrans - Eh;
        relError = sqrt(diff.quadrupleContract(diff) / Eh.quadrupleContract(Eh));
        cout << "Transformed original tensor:" << endl << EhTrans << endl;
        cout << "Transformed rel error:\t" << relError << endl << endl;
        totalTransformRelError += relError;
    }

    cout << "Average rel error:\t" << totalRelError / nSamples << endl;
    cout << "Average tranform rel error:\t" << totalTransformRelError / nSamples << endl;

    BENCHMARK_REPORT();
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    po::variables_map args = parseCmdLine(argc, argv);

    vector<MeshIO::IOVertex>  inVertices;
    vector<MeshIO::IOElement> inElements;
    string meshPath = args["mesh"].as<string>();
    auto type = load(meshPath, inVertices, inElements, MeshIO::FMT_GUESS,
                     MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw runtime_error("Mesh must be triangle or tet.");

    // Look up and run appropriate homogenizer instantiation.
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(args, inVertices, inElements);

    return 0;
}
