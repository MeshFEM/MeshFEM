////////////////////////////////////////////////////////////////////////////////
// ConstStrainDisplacement_cli.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Generate a constant strain displacement for demonstration purposes.
//      Given a strain tensor, create a linear displacement field with that
//      strain. We choose the displacement field with u = 0 at the bounding box
//      center and with no infinitesimal rigid rotation component.
//
//      The no-rigid-rotation constraint makes integrating the strain simple
//      because it allows us to treat the strain tensor as a Jacobian:
//      J = 1/2 (J + J') + 1/2 (J - J') = strain + irot
//      irot = 0 ==> J = strain
//      This can also be interpreted componentwise: no rotation means, e.g.,
//      1/2 (u_x,y - u_y,x) = 0       \   ==> u_x,y = u_y,x = e_xy = e_yx
//      1/2 (u_x,y + u_y, x) = e_xy   /
//
//      If requested, the fluctuation displacement computed by periodic
//      homogenization can be added in.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/04/2014 01:18:22
////////////////////////////////////////////////////////////////////////////////
#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>
#include "Types.hh"
#include <SymmetricMatrix.hh>
#include "FEMMesh.hh"
#include "MeshIO.hh"
#include "LinearElasticity.hh"
#include "Materials.hh"
#include "PeriodicHomogenization.hh"
#include "MSHFieldWriter.hh"

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace PeriodicHomogenization;

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: ConstStrainDisplacement_cli [options] in.msh -s 'e_00 e_11 ...' out.msh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("mesh", po::value<string>(), "input mesh")
        ("outMesh", po::value<string>(), "output mesh")
        ;
    po::positional_options_description p;
    p.add("mesh",    1);
    p.add("outMesh", 1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("material,m", po::value<string>(), "base material")
        ("strain,s", po::value<string>(), "strain tensor")
        ("degree,d",   po::value<int>()->default_value(1), "degree of finite elements")
        ("addFluctuation,f",                "add fluctuation strains to the displacement")
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
    if (vm.count("outMesh") == 0) {
        cout << "Error: must specify input and output mesh" << endl;
        fail = true;
    }

    if (vm.count("strain") == 0) {
        cout << "Error: must specify strain tensor" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N>
using HMG = LinearElasticity::HomogenousMaterialGetter<Materials::Constant>::template Getter<_N>;

template<size_t _N, size_t _FEMDegree>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements)
{
    auto &mat = HMG<_N>::material;
    if (args.count("material")) mat.setFromFile(args["material"].as<string>());
    typedef LinearElasticity::Mesh<_N, _FEMDegree, HMG> Mesh;
    typedef LinearElasticity::Simulator<Mesh> Simulator;
    typedef typename Simulator::VField VField;
    Simulator sim(inElements, inVertices);
    const auto &mesh = sim.mesh();
    MSHFieldWriter writer(args["outMesh"].as<string>(), mesh);

    // Parse strain tensor.
    vector<string> strainComponents;
    string strainString = args["strain"].as<string>();
    boost::trim(strainString);
    boost::split(strainComponents, strainString, boost::is_any_of("\t "),
                 boost::token_compress_on);
    if (strainComponents.size() != flatLen(_N))
        throw runtime_error("Invalid strain tensor");
    SymmetricMatrixValue<Real, _N> strain;
    for (size_t i = 0; i < strainComponents.size(); ++i)
        strain[i] = stod(strainComponents[i]);

    auto bbox = mesh.boundingBox();
    VectorND<_N> center = 0.5 * (bbox.minCorner + bbox.maxCorner);
    VField cstrainDisp(mesh.numNodes());
    cstrainDisp.clear();
    for (size_t vi = 0; vi < mesh.numNodes(); ++vi) {
        VectorND<_N> p = mesh.node(vi)->p - center;
        for (size_t ci = 0; ci < _N; ++ci)
            for (size_t cj = 0; cj < _N; ++cj)
                cstrainDisp(vi)[ci] += strain(ci, cj) * p[cj];
    }


    if (args.count("addFluctuation")) {
        std::vector<VField> w_ij;
        solveCellProblems(w_ij, sim);

        // Remove rigid translation of fluctuation displacements relative to the
        // base cell (i.e. try to keep the fluctuation-displaced microstructure
        // "within" the base cell):
        // The no-rigid-motion constraint on fluctuation displacements
        // ensures the microstructure's center of mass doesn't move, but
        // this is not what we need. Instead, we need to ensure vertices on
        // periodic boundary do not move off the boundary. We enforce this in an
        // average sense for each cell face by translating so that the
        // corresponding displacement component's average over all vertices on
        // the face is zero.
        for (size_t i = 0; i < w_ij.size(); ++i) {
            VField &w = w_ij[i];
            VectorND<_N> translation(VectorND<_N>::Zero());
            vector<int> numAveraged(_N);
            
            for (size_t bni = 0; bni < mesh.numBoundaryNodes(); ++bni) {
                auto n = mesh.boundaryNode(bni).volumeNode();
                for (size_t d = 0; d < _N; ++d) {
                    if (std::abs(n->p[d] - bbox.minCorner[d]) < 1e-9) {
                        translation[d] += w(n.index())[d];
                        ++numAveraged[d];
                    }
                }
            }
            for (size_t d = 0; d < _N; ++d)
                translation[d] /= numAveraged[d];
            for (size_t n = 0; n < w.domainSize(); ++n)
                w(n) -= translation;
        }

        for (size_t i = 0; i < w_ij.size(); ++i) {
            VField tmp(w_ij[i]);
            tmp *= (((i < _N) ? 1.0 : 2.0) * strain[i]);
            cstrainDisp += tmp;

            writer.addField("w_ij" + to_string(i), w_ij[i], MSHFieldWriter::PER_NODE);
        }
    }

    writer.addField("u_cstrain", cstrainDisp, MSHFieldWriter::PER_NODE);

    writer.addField("stress", sim.averageStressField(cstrainDisp), MSHFieldWriter::PER_ELEMENT);
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
    auto type = load(args["mesh"].as<string>(), inVertices, inElements,
            MeshIO::FMT_GUESS, MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be triangle or tet.");

    // Look up and run appropriate instantiation.
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);
    exec(args, inVertices, inElements);
    return 0;
}
