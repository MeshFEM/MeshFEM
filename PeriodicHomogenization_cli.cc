#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
#include "JSFieldWriter.hh"
#include "LinearElasticity.hh"
#include "Materials.hh"
#include "PeriodicHomogenization.hh"
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>

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
        ("material,m",      po::value<string>(), "base material")
        ("output,o",        po::value<string>(), "output .js mesh + fields")
        ("parameterStep,p", po::value<double>(), "(fractional) ammount by which to attempt to change elastic coefficients")
        ;

    po::options_description cli_opts;
    cli_opts.add(visible_opts).add(hidden_opts);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).
                  options(cli_opts).positional(p).run(), vm);
        po::notify(vm);
    }
    catch (std::exception &e) {
        cout << "Error: " << e.what() << endl << endl;
        usage(1, visible_opts);
    }

    bool fail = false;
    if (vm.count("mesh") == 0) {
        cout << "Error: must specify input mesh" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N>
using ETensor = typename LinearElasticityND<_N>::ETensor;
template<size_t _N>
using VField = typename LinearElasticityND<_N>::VField;
typedef ScalarField<Real> SField;

template<size_t _N, class Simulator>
void writeDescentVectors(JSFieldWriter<_N> &writer,
        const string &name, const SField &v_n, const Simulator &sim) {
    size_t numBE = sim.mesh().numBoundaryElements();
    VField<_N> direction(numBE);
    for (size_t be = 0; be < numBE; ++be)
        direction(be) = v_n[be] * sim.mesh().boundaryElement(be)->normal();
    writer.addField(name + " direction", direction,
            JSFieldWriter<_N>::PER_BDRY_ELEM);
}

// Writes steepest descent direction for
//      1/2 sum_ijkl (target_ijkl - Eh_ijlk|)^2
// That is, -grad(1/2 sum_ijkl (target_ijkl - Eh_ijlk|)^2) =
//  -(target_ijkl - Eh_ijlk) * -grad(Eh_ikjl)) =
//   (target_ijkl - Eh_ijlk) *  grad(Eh_ikjl))
template<size_t _N, class Simulator>
void writeTargetTensorShapeDerivative(JSFieldWriter<_N> &writer,
                const string &name, const ETensor<_N> &target,
                const ETensor<_N> &current, const vector<ETensor<_N>> &gradEh,
                const std::vector<VField<_N>> &w_ij, const Simulator &sim) {
    size_t numBE = sim.mesh().numBoundaryElements();
    assert(gradEh.size() == numBE);
    ETensor<_N> diff = target - current;
    SField v_n(numBE);

    for (size_t be = 0; be < numBE; ++be)
        v_n[be] = diff.quadrupleContract(gradEh[be]);

    writer.addField(name + " v_n", v_n, JSFieldWriter<_N>::PER_BDRY_ELEM);
    writeDescentVectors(writer, name, v_n, sim);
}

template<size_t _N>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements) {
    auto &mat = LinearElasticityND<_N>::
        template homogenousMaterial<Materials::Constant>();
    if (args.count("material")) mat.setFromFile(args["material"].as<string>());
    typename LinearElasticityND<_N>:: template
        HomogenousSimulator<Materials::Constant> sim(inElements, inVertices);

    JSFieldWriter<_N> *writer = NULL;
    if (args.count("output"))
        writer = new JSFieldWriter<_N>(args["output"].as<string>(), sim.mesh());

    std::vector<VField<_N>> w_ij;
    // solveCellProblems(w_ij, sim, writer);
    solveCellProblems(w_ij, sim, (JSFieldWriter<_N> *) NULL);

    // if (writer) {
    //     // Output fluctuation strains.
    //     for (size_t i = 0; i < w_ij.size(); ++i) {
    //         string name("w_ij ");
    //         name += to_string(i);
    //         writer->addField(string("w_ij ") + to_string(i), w_ij[i],
    //                         JSFieldWriter<_N>::PER_NODE);
    //     }
    // }

    ETensor<_N> Eh = homogenizedElasticityTensor(w_ij, sim);

    cout << setprecision(16) << endl;
    cout << "Homogenized elasticity tensor:" << endl;
    cout << Eh << endl << endl;

    ETensor<_N> S = Eh.inverse();
    vector<Real> moduli(flatLen(_N));

    // Shear moduli are multiplied by 4 in flattened compliance tensor...
    for (size_t i = 0; i < flatLen(_N); ++i)
        moduli[i] = ((i < _N) ? 1.0 : 0.25) / S.D(i, i);

    vector<Real> poisson;
    if (_N == 2) poisson = { -S.D(0, 1) / S.D(1, 1),   // v_yx
                             -S.D(1, 0) / S.D(0, 0) }; // v_xy
    else         poisson = { -S.D(0, 1) / S.D(1, 1),   // v_yx
                             -S.D(0, 2) / S.D(2, 2),   // v_zx
                             -S.D(1, 2) / S.D(2, 2),   // v_zy
                             -S.D(1, 0) / S.D(0, 0),   // v_xy
                             -S.D(2, 0) / S.D(0, 0),   // v_xz
                             -S.D(2, 1) / S.D(1, 1) }; // v_zy

    if (_N == 2)  {
        cout << "Approximate Young moduli:\t"  << moduli[0] << "\t" << moduli[1] << endl;
        cout << "Approximate shear modulus:\t" << moduli[2] << endl;

        cout << "v_yx, v_xy:\t" << poisson[0] << "\t" << poisson[1] << endl;
    }
    else {
        cout << "Approximate Young moduli:\t" << moduli[0] << "\t" << moduli[1] << "\t"
             << moduli[2] << endl;
        cout << "Approximate shear moduli:\t" << moduli[3] << "\t" << moduli[4] << "\t"
             << moduli[5] << endl;

        cout << "v_yx, v_zx, v_zy:\t" << poisson[0] << "\t" << poisson[1] << "\t" << poisson[2] << endl;
        cout << "v_xy, v_xz, v_yz:\t" << poisson[3] << "\t" << poisson[4] << "\t" << poisson[5] << endl;
    }

    if (writer)  {
        std::vector<ETensor<_N>> gradEh = homogenizedTensorGradient(w_ij, sim);
        ////////////////////////////////////////////////////////////////////////
        // Moduli Derivatives - for orthotropic only!
        ////////////////////////////////////////////////////////////////////////
        // Gradient of the compliance tensor--useful for moduli derivatives.
        std::vector<ETensor<_N>> gradEhinv;
        for (const auto &G : gradEh)
            gradEhinv.push_back(-S.doubleDoubleContract(G));

        size_t numBE = sim.mesh().numBoundaryElements();
        SField gradEx(numBE), gradEy(numBE), gradVyx(numBE), gradVxy(numBE),
               gradmu(numBE);
        for (size_t i = 0; i < numBE; ++i) {
             gradEx[i] = -moduli[0] * moduli[0] * gradEhinv[i].D(0, 0);
             gradEy[i] = -moduli[1] * moduli[1] * gradEhinv[i].D(1, 1);
            gradVyx[i] = -gradEy[i] * S.D(0, 1) - moduli[1] * gradEhinv[i].D(0, 1);
            gradVxy[i] = -gradEx[i] * S.D(1, 0) - moduli[0] * gradEhinv[i].D(1, 0);
             gradmu[i] =  gradEh[i].D(2, 2);
        }
        writer->addField( "gradEx",  gradEx, JSFieldWriter<_N>::PER_BDRY_ELEM);
        writer->addField( "gradEy",  gradEy, JSFieldWriter<_N>::PER_BDRY_ELEM);
        writer->addField("gradVyx", gradVyx, JSFieldWriter<_N>::PER_BDRY_ELEM);
        writer->addField("gradVxy", gradVxy, JSFieldWriter<_N>::PER_BDRY_ELEM);
        writer->addField( "gradmu",  gradmu, JSFieldWriter<_N>::PER_BDRY_ELEM);

        writeDescentVectors(*writer, "gradEx",  gradEx, sim);
        writeDescentVectors(*writer, "gradEy",  gradEy, sim);
        writeDescentVectors(*writer,"gradVyx", gradVyx, sim);
        writeDescentVectors(*writer,"gradVxy", gradVxy, sim);
        writeDescentVectors(*writer, "gradmu",  gradmu, sim);
    }
    // if (writer && args.count("parameterStep")) {
    //     Real parameterStep = args["parameterStep"].as<double>();
    //     cout << "Parameter step: " << parameterStep << endl;

    //     ETensor<_N> ETargetinv(S);
    //     Real currentPoisson = -ETargetinv.D(0, 1) / ETargetinv.D(1, 1);
    //     Real targetPoisson = currentPoisson + parameterStep * (-0.5 - currentPoisson);
    //     ETargetinv.D(0, 1) = -targetPoisson * ETargetinv.D(1, 1);
    //     writeTargetTensorShapeDerivative(*writer, "v_yx decreasing",
    //             ETargetinv.inverse(), Eh, gradEh, w_ij, sim);

    //     ETargetinv = S;
    //     currentPoisson = -ETargetinv.D(0, 1) / ETargetinv.D(0, 0);
    //     targetPoisson = currentPoisson + parameterStep * (-0.5 - currentPoisson);
    //     ETargetinv.D(0, 1) = -targetPoisson * ETargetinv.D(1, 1);
    //     writeTargetTensorShapeDerivative(*writer, "v_xy decreasing",
    //             ETargetinv.inverse(), Eh, gradEh, w_ij, sim);

    //     // Step a bit toward the tensor with Ex halved.
    //     ETargetinv = S;
    //     ETargetinv.D(0, 0) *= 2;
    //     ETargetinv.D(1, 0) *= ETargetinv.D(0, 0) / S.D(0, 0);
    //     // ETargetinv.D(0, 0) += parameterStep * ETargetinv.D(0, 0);
    //     // ETargetinv.D(1, 0) *= ETargetinv.D(0, 0) / S.D(0, 0);
    //     writeTargetTensorShapeDerivative(*writer, "Ex decreasing, v_xy fixed",
    //             ETargetinv.inverse(), Eh, gradEh, w_ij, sim);

    //     ETargetinv = S;
    //     ETargetinv.D(1, 1) *= 2;
    //     ETargetinv.D(0, 1) *= ETargetinv.D(1, 1) / S.D(1, 1);
    //     // ETargetinv.D(1, 1) += parameterStep * ETargetinv.D(1, 1);
    //     // ETargetinv.D(0, 1) *= ETargetinv.D(1, 1) / S.D(1, 1);
    //     writeTargetTensorShapeDerivative(*writer, "Ey decreasing, v_yx fixed",
    //             ETargetinv.inverse(), Eh, gradEh, w_ij, sim);

    //     ETargetinv = S;
    //     ETargetinv.D(2, 2) += parameterStep * ETargetinv.D(2, 2);
    //     writeTargetTensorShapeDerivative(*writer, "mu decreasing",
    //             ETargetinv.inverse(), Eh, gradEh, w_ij, sim);
    // 
    // }

    if (writer) delete writer;
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
    else    throw std::runtime_error("Mesh must be triangle or tet.");

    // Look up and run appropriate homogenizer instantiation.
    auto exec = (dim == 3) ? execute<3> : execute<2>;

    exec(args, inVertices, inElements);

    return 0;
}
