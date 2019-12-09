#include "argparse.hh"
#include <stdexcept>
#include <memory>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/optional.hpp>

namespace po = boost::program_options;
using namespace std;

[[ noreturn ]] void usage(int status, const po::options_description &visible_opts) {
    cout << "Usage: msh_processor in.msh [options]" << endl;
    cout << visible_opts << endl;
    exit(status);
}

std::tuple<std::string, std::vector<FilterInvocation>, boost::optional<size_t>> parseCmdLine(int argc, char *argv[]) {
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("msh", po::value<string>(), "input msh file")
        ;
    po::positional_options_description p;
    p.add("msh", 1);

    po::options_description parser_operations("Data source operations");
    parser_operations.add_options()
        ("forceDimension", po::value<size_t>(), "Force the mesh to be interpreted as embedded in a particular dimension instead of using the simplex dimension.")
        ("list",                                "List all fields in the msh")
        ("extract,e", po::value<string>(),      "Extract field(s) matching a given name (or name pattern)")
        ("extractAll",                          "Extract all fields")
        ("generate,g", po::value<string>(),     "Generate a mesh property field (valid arguments: 'x', 'volume', 'barycenter')")
        ("expression,E", po::value<string>(),   "Generate a per-node scalar/vector field from an expression (comma separated components")
        ("transferFieldsToPerElem,t", po::value<string>(), "Transfer fields on the stack to per-element fields on the passed target.msh (target.msh becomes current mesh/data source)")
        ("loadNewMSH,L",              po::value<string>(), "Load a new mesh/data source, replacing the current one. Must be of same spatial dimension as current mesh.")
        ("filterElements,F",                               "Filter out elements of a mesh using an indicator scalar field.")
        ;
    po::options_description stack_operations("Stack operations");
    stack_operations.add_options()
        ("dup",                            "Duplicate top of the stack")
        ("pop",                            "Pop top of the stack")
        ("pull",      po::value<string>(), "Pull a named value to the top of the stack")
        ("push",      po::value<string>(), "Push a scalar literal to the top of the stack")
        ("reverse",                        "Reverse the entire stack.")
        ("import_sfield", po::value<string>(), "Import a scalar field from an ascii file")
        ("import_vfield", po::value<string>(), "Import a vector field from an ascii file")
        ("noprint",                        "Suppress printing on exit")
        ("print,p",                        "Print top of stack")
        ("printName",                      "Print name of value at the top of the stack")
        ("rename,r",  po::value<string>(), "Rename the field(s) at the top of the stack (multiple names separated by commas)")
        ("setNodePositions",               "Overwrite the mesh node positions with the vector field at the top of the stack (popping the vector field)")
        ("outMSH,o",  po::value<string>(), "Output msh file with named fields for each entry in the stack")
        ;
    po::options_description modifiers("Modifiers: change the behavior of the next operation");
    modifiers.add_options()
        ("applyAll,A",                     "Apply next operation to entire stack instead of top")
        ("outerReduction,O",               "Apply the next reduction operation componentwise instead of the default (pointwise)")
        ;
    po::options_description unary_operations("Unary operations: act on each component (and at each point for fields)");
    unary_operations.add_options()
        ("abs,a",                          "Componentwise abs of scalar field or vector field")
        ("scale,s", po::value<string>(),   "Multiply the top of the stack by a scalar.")
        ("set",     po::value<string>(),   "Set every component of the top value to arg.")
        ;
    po::options_description reductions("(Partial) reduction operations: reduce vectors/matrices to scalars. "
                                       "By default, the reduction is along the innermost index of multi-indexed objects, e.g. Field<Vector> is reduced pointwise to Field<Scalar>, "
                                       "but you can prefix with -O to reduce along the outermost index");
    reductions.add_options()
        ("index,i", po::value<string>(),   "Indexed access into a value (flattened indexing for symmetric matrices)")
        ("min,m",                          "Min component")
        ("max,M",                          "Max component")
        ("minMag",                         "Min magnitude")
        ("maxMag",                         "Max magnitude")
        ("norm,n",                         "L2/Frobenius norm")
        ("sum,S",                          "Sum the components")
        ("mean",                           "Average the components")
        ;
    po::options_description matrix_operations("Special symmetric matrix (and symmetric matrix field) operations");
    matrix_operations.add_options()
        ("eigenvalues,l", "Eigenvalues")
        ("vonMises,v",    "von Mises stress measure (deviatoric part, weighted by sqrt(3/2))")
        ("frobeniusNorm", "Frobenius norm of tensor")
        // ("eigs",          "Eigenvalues + eigenvectors")
        ;
    po::options_description field_operations("Field operations: sampling of fields, fields of interpolants, and interpolants");
    field_operations.add_options()
        ("sample",  po::value<string>(),   "Sample a field/interpolant at a point. For fields, "
                                           "uses piecewise constant interpolation on Voronoi diagram of points/element barycenters. "
                                           "The point is specified as a comma-separated vector")
        ("elementAverage",                 "Averages the field over each element. Makes sense for nodal fields and interpolant fields.")
        ("smoothedElementField",           "Create a smoothed, piecewise constant field by averaging over each element's neighborhood.")
        // ("percentile", po::value<double>(), "extract a certain percentile of the field")
        ;
    po::options_description binary_operations("Component-wise binary operations: lower-dimensional types are implicitly promoted to higher dimension when it makes sense. "
                                              "For example, scalars can be added to vectors, matrices, or fields, but vectors cannot be added to symmetric matrices.");
    binary_operations.add_options()
        ("add", "Add      the top two values on the stack")
        ("sub", "Subtract the top two values on the stack (prev - top)")
        ("mul", "Multiply the top two values on the stack")
        ("div", "Divide   the top two values on the stack (prev / top)")
        ;

    po::options_description cli_opts;
    cli_opts.add_options()("help,h", "Produce this help message");
    cli_opts.add(parser_operations).add(stack_operations).add(modifiers)
            .add(reductions).add(unary_operations).add(binary_operations)
            .add(matrix_operations).add(field_operations)
            .add(hidden_opts);

    // Options visible in the help message.
    po::options_description visible_opts;
    visible_opts.add_options()("help,h", "Produce this help message");
    visible_opts.add(parser_operations).add(stack_operations).add(modifiers)
           .add(reductions).add(unary_operations).add(binary_operations)
           .add(matrix_operations).add(field_operations);

    unique_ptr<po::parsed_options> parsedOptions;
    try {
        parsedOptions = std::make_unique<po::parsed_options>(po::command_line_parser(argc, argv).
                            options(visible_opts).positional(p).run());
    }
    catch (std::exception &e) {
        cout << "Error: " << e.what() << endl << endl;
        usage(1, visible_opts);
    }

    int numMeshes = 0;
    bool helpReq = false;
    string mshFile;
    vector<FilterInvocation> filters;
    auto forcedDim = boost::make_optional(false, size_t()); // work around maybe-uninitialized GCC warning bug
    for (const auto &opt : parsedOptions->options) {
        if (opt.string_key == "msh") { ++numMeshes; mshFile = opt.value[0]; }
        else if (opt.string_key == "help") helpReq = true;
        else if (opt.string_key == "forceDimension") forcedDim = std::stod(opt.value.at(0));
        else                         { filters.push_back({opt.string_key, (opt.value.size() ? opt.value[0] : "")}); }
    }

    bool fail = false;
    if (numMeshes != 1) {
        cout << "Error: must specify one input msh file" << endl;
        fail = true;
    }

    if (fail || helpReq)
        usage(fail, visible_opts);

    return make_tuple(mshFile, filters, forcedDim);
}

int parseIntArg(const string &arg) {
    size_t end;
    int val = stoi(arg, &end);
    for (char c : arg.substr(end))
        if (!isspace(c)) throw runtime_error("Argument must be an integer");
    return val;
}

double parseRealArg(const string &arg) {
    size_t end;
    double val = stod(arg, &end);
    for (char c : arg.substr(end))
        if (!isspace(c)) throw runtime_error("Argument must be a real number");
    return val;
}

// Parse a comma-separated vector from a string
// Throws exception if the parsed size is not N.
template<size_t N>
VecHPCFix<N> parseVectorArg(const std::string &argImmutable)
{
    string arg(argImmutable);
    vector<string> argComponents;
    boost::trim(arg);
    boost::split(argComponents, arg, boost::is_any_of(","), boost::token_compress_on);
    if (argComponents.size() != N) throw std::runtime_error("Invalid vector argument size");
    VectorND<N> result;
    for (size_t i = 0; i < N; ++i)
        result[i] = std::stod(argComponents[i]);
    return result;
}

// Parse a semicolon-separated list of comma-separated vectors from a string
// Throws exception if any of the parsed vectors are not of size N.
template<size_t N>
std::vector<VecHPCFix<N>> parseVectorListArg(const std::string &argImmutable)
{
    std::vector<VecHPCFix<N>> result;
    string arg(argImmutable);
    vector<string> argComponents;
    boost::trim(arg);
    boost::split(argComponents, arg, boost::is_any_of(";"), boost::token_compress_on);
    if (argComponents.size() == 0) throw std::runtime_error("Vector list argument must contain at least one vector.");
    for (auto &vecArg : argComponents) {
        vector<string> vecComponents;
        boost::trim(vecArg);
        boost::split(vecComponents, vecArg, boost::is_any_of(","), boost::token_compress_on);
        if (vecComponents.size() != N) throw std::runtime_error("Invalid vector argument size");
        VectorND<N> vec;
        for (size_t i = 0; i < N; ++i)
            vec[i] = std::stod(vecComponents[i]);
        result.emplace_back(std::move(vec));
    }

    return result;
}

// Explicit instantiation
template VecHPCFix<2> parseVectorArg<2>(const std::string &arg);
template VecHPCFix<3> parseVectorArg<3>(const std::string &arg);

template std::vector<VecHPCFix<2>> parseVectorListArg<2>(const std::string &arg);
template std::vector<VecHPCFix<3>> parseVectorListArg<3>(const std::string &arg);
