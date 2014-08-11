#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
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

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: PeriodicHomogenization_cli [options] dimension mesh output.msh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("mesh",       po::value<string>(),                     "input mesh")
        ("outputMSH",  po::value<string>()->default_value(""),  "output mesh")
        ;
    po::positional_options_description p;
    p.add("mesh",                1)
     .add("outputMSH",           1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("material",       po::value<string>()->default_value(""),   "base material")
        // ("parameterStep",  po::value<double>()->default_value(0.1),  "(fractional) ammount by which to attempt to change elastic coefficients")
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
void execute(const string &materialPath, const string &outMSH,
             const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements) {
    auto &mat = LinearElasticityND<_N>::
        template homogenousMaterial<Materials::Constant>();
    if (materialPath != "") mat.setFromFile(materialPath);

    typename LinearElasticityND<_N>:: template
        HomogenousSimulator<Materials::Constant> sim(inElements, inVertices);

    MSHFieldWriter *writer = NULL;
    if (outMSH != "") writer = new MSHFieldWriter(outMSH, sim.mesh());

    std::vector<typename LinearElasticityND<_N>::VField> w_ij;
    PeriodicHomogenization::solveCellProblems(w_ij, sim, writer);

    if (writer) {
        // Output fluctuation strains.
        for (size_t i = 0; i < w_ij.size(); ++i) {
            string name("w_ij ");
            name += to_string(i);
            writer->addField(string("w_ij ") + to_string(i), w_ij[i],
                            MSHFieldWriter::PER_NODE);
        }
    }

    typedef typename LinearElasticityND<_N>::ETensor ETensor;
    ETensor Eh = PeriodicHomogenization::homogenizedElasticityTensor(w_ij, sim);

    cout << setprecision(16) << endl;
    cout << "Homogenized elasticity tensor:" << endl;
    cout << Eh << endl << endl;

    ETensor Einv = Eh.inverse();
    auto moduli((1.0 / Einv.diag().array()).eval());
    if (_N == 2)  {
        cout << "Approximate Young moduli:\t"  << moduli[0] << "\t" << moduli[1] << endl;
        cout << "Approximate shear modulus:\t" << moduli[2] << endl;

        cout << "v_yx, v_xy:\t" << -Einv.D(0, 1) / Einv.D(1, 1) << "\t"
                                << -Einv.D(1, 0) / Einv.D(0, 0) << endl;
    }
    else {
        cout << "Approximate Young moduli:\t" << moduli[0] << "\t" << moduli[1] << "\t"
             << moduli[2] << endl;
        cout << "Approximate shear moduli:\t" << moduli[3] << "\t" << moduli[4] << "\t"
             << moduli[5] << endl;

        cout << "v_yx, v_zx, v_zy:\t" << -Einv.D(0, 1) / Einv.D(1, 1) << "\t"
                                      << -Einv.D(0, 2) / Einv.D(2, 2) << "\t"
                                      << -Einv.D(1, 2) / Einv.D(2, 2) << endl;
        cout << "v_xy, v_xz, v_yz:\t" << -Einv.D(1, 0) / Einv.D(0, 0) << "\t"
                                      << -Einv.D(2, 0) / Einv.D(0, 0) << "\t"
                                      << -Einv.D(2, 1) / Einv.D(1, 1) << endl;
    }
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

    string material = args["material"].as<string>();
    exec(material, args["outputMSH"].as<string>(), inVertices, inElements);

    return 0;
}
