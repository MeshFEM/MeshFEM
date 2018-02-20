////////////////////////////////////////////////////////////////////////////////
// selector.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Highlight a given vertex or element in a mesh by writing a indicator
//		scalar field.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/08/2014 17:30:11
////////////////////////////////////////////////////////////////////////////////
#include <MeshIO.hh>
#include <MSHFieldWriter.hh>
#include <MSHFieldParser.hh>
#include <CSGFEM/Fields.hh>
#include <stdexcept>
#include <vector>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: selector [options] in.msh out.msh" << endl;
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
        ("vertices,v", po::value<vector<int>>()->multitoken(), "highlight vertices")
        ("elements,e", po::value<vector<int>>()->multitoken(), "highlight elements")
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

    if (vm.count("vertices") + vm.count("elements") == 0) {
        cout << "Error: must specify geometry to highlight" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &vertices, 
             const vector<MeshIO::IOElement> &elements) {
    MSHFieldWriter writer(args["outMesh"].as<string>(), vertices, elements);

    if (args.count("vertices")) {
        std::vector<int> vidxs = args["vertices"].as<vector<int>>();
        ScalarField<Real> indicator(vertices.size());
        indicator.clear();
        for (size_t vtx : vidxs) {
            if (vtx >= vertices.size())
                throw runtime_error("invalid vertex index");
            indicator[vtx] = 1.0;
        }
        string name("vertices");
        for (size_t vtx : vidxs) name += " " + to_string(vtx);
        writer.addField(name, indicator, DomainType::PER_NODE);
        cout << "wrote field " << name << endl;
    }
    if (args.count("elements")) {
        std::vector<int> eidxs = args["elements"].as<vector<int>>();
        ScalarField<Real> indicator(elements.size());
        indicator.clear();
        for (size_t elem : eidxs) {
            if (elem >= elements.size())
                throw runtime_error("invalid element index");
            indicator[elem] = 1.0;
        }
        string name("elements");
        for (size_t elem : eidxs) name += " " + to_string(elem);
        writer.addField(name, indicator, DomainType::PER_ELEMENT);
        cout << "wrote field " << name << endl;
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

    vector<MeshIO::IOVertex > vertices;
    vector<MeshIO::IOElement> elements;

    auto type = load(args["mesh"].as<string>(), vertices, elements);
    if (elements.size() == 0) throw runtime_error("No elements read.");

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    // Look up and run appropriate instantiation.
    auto exec = (dim == 3) ? execute<3> : execute<2>;
    exec(args, vertices, elements);
    return 0;
}
