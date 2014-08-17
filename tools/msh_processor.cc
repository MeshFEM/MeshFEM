////////////////////////////////////////////////////////////////////////////////
// msh_processor.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extracts and processes fields from a .msh file. The field processor is
//      essentually an RPN evaluator that maintains a stack of values and
//      applies the filters specified on the command line in order.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/16/2014 15:26:04
////////////////////////////////////////////////////////////////////////////////
#include <MSHFieldParser.hh>
#include <Types.hh>

#include <boost/program_options.hpp>
#include <regex>
#include <vector>
#include <stdexcept>
#include <memory>

using namespace MeshIO;

namespace po = boost::program_options;
using namespace std;

void usage(int status, const po::options_description &visible_opts) {
    cout << "Usage: msh_field_extractor [options] in.msh" << endl;
    cout << visible_opts << endl;
    exit(status);
}

po::parsed_options parseCmdLine(int argc, char *argv[]) {
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("msh", po::value<string>(), "input msh file")
        ;
    po::positional_options_description p;
    p.add("msh", 1);

    po::options_description visible_opts;
    visible_opts.add_options()("help,h", "Produce this help message")
        ("extract,e", po::value<string>(),  "extract field(s) matching a given name (or name pattern)")
        ("extractAll",                      "extract all fields")
        ("print,p",                         "print top of stack")
        ("rename,r",  po::value<string>(),  "rename the field(s) at the top of the stack (multiple names separated by commas)")
        ("outMSH,o",  po::value<string>(),  "output msh file with named fields for each entry in the stack")
        ("max,M",                           "max of scalar field (or element-wise for vector field)")
        ("min,m",                           "min of scalar field (or element-wise for vector field)")
        ("maxMag",                          "max magnitude of scalar field (or element-wise for vector field)")
        ("minMag",                          "min magnitude of scalar field (or element-wise for vector field)")
        ("abs,a",                           "componentwise abs of scalar field or vector field")
        ("norm,n",                          "L2 norm of scalar field (or element-wise for vector field)")
        ("eigenvalues,l",                   "eigenvalues for symmetric matrix field (vector field result)")
        ("percentile", po::value<double>(), "extract a certain percentile of the msh file")
        ;

    po::options_description cli_opts;
    cli_opts.add(visible_opts).add(hidden_opts);

    po::parsed_options *parsedOptions = NULL;
    try {
        parsedOptions = new po::parsed_options(po::command_line_parser(argc, argv).
                            options(cli_opts).positional(p).run());
    }
    catch (std::exception &e) {
        cout << "Error: " << e.what() << endl << endl;
        usage(1, visible_opts);
    }

    int numMeshes = 0;
    bool helpReq = false;
    for (const auto &opt : parsedOptions->options) {
        if (opt.string_key == "msh") ++numMeshes;
        if (opt.string_key == "help") helpReq = true;
    }

    bool fail = false;
    if (numMeshes != 1) {
        cout << "Error: must specify one input msh file" << endl;
        fail = true;
    }

    if (fail || helpReq)
        usage(fail, visible_opts);

    return *parsedOptions;
}

////////////////////////////////////////////////////////////////////////////////
// Types of values that can live on the stack.
////////////////////////////////////////////////////////////////////////////////
template<size_t N>
struct Value {
    Value(const string &n) : name(n) { }
    string name;
    virtual void print(std::ostream &os = std::cout) const = 0;
};

template<size_t N>
struct SFieldValue : public Value<N> {
    typedef Value<N> Base;
    typedef ScalarField<Real> value_type;
    value_type value;
    SFieldValue(const string &n, const value_type &v) : Base(n), value(v) { }
    virtual void print(std::ostream &os = std::cout) const {
        os << value << std::endl;
    }
};

template<size_t N>
struct VFieldValue : public Value<N> {
    typedef Value<N> Base;
    typedef VectorField<Real, N> value_type;
    value_type value;
    VFieldValue(const string &n, const value_type &v) : Base(n), value(v) { }
    virtual void print(std::ostream &os = std::cout) const {
        // os << value << std::endl;
    }
};

template<size_t N>
struct SMFieldValue : public Value<N> {
    typedef Value<N> Base;
    typedef SymmetricMatrixField<Real, N> value_type;
    value_type value;
    SMFieldValue(const string &n, const value_type &v) : Base(n), value(v) { }
    virtual void print(std::ostream &os = std::cout) const {
        // os << value << std::endl;
    }
};

template<size_t N>
struct ScalarValue : public Value<N> {
    typedef Value<N> Base;
    typedef Real value_type;
    value_type value;
    ScalarValue(const string &n, const value_type &v) : Base(n), value(v) { }
    virtual void print(std::ostream &os = std::cout) const {
        // os << value << std::endl;
    }
};

template<size_t N>
using VPtr = shared_ptr<Value<N> >; 

// Filter invocation: (name, argument string)
typedef pair<string, string> FilterInvocation;

////////////////////////////////////////////////////////////////////////////////
// Filters - operate on the stack.
// These are all template functions with the signature:
// template<size_t N>
// void f(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg)
////////////////////////////////////////////////////////////////////////////////
// Data source filters
// Extract field(s) matching the pattern in "arg", pushing them on the top of
// the stack.
template<size_t N>
void extract(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
             const string &arg) {
    std::regex pattern(arg);
    size_t numMatched = 0;
    for (const string &name : parser.scalarFieldNames()) {
        if (regex_match(name, pattern)) {
            stack.push_back(VPtr<N>(new SFieldValue<N>(name, parser.scalarField(name))));
            ++numMatched;
        }
    }
    for (const string &name : parser.vectorFieldNames()) {
        if (regex_match(name, pattern)) {
            stack.push_back(VPtr<N>(new VFieldValue<N>(name, parser.vectorField(name))));
            ++numMatched;
        }
    }
    for (const string &name : parser.symmetricMatrixFieldNames()) {
        if (regex_match(name, pattern)) {
            stack.push_back(VPtr<N>(new SMFieldValue<N>(name,
                        parser.symmetricMatrixField(name))));
            ++numMatched;
        }
    }
    if (numMatched == 0) throw runtime_error("No fields matched " + arg);
}

// Compute a value (e.g. element volume) from the mesh, pushing it on the top of
// the stack
// void compute

// Single operand filters
template<size_t N>
void max(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);
template<size_t N>
void min(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);
template<size_t N>
void sum(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);
template<size_t N>
void maxMag(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);
template<size_t N>
void minMag(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);
template<size_t N>
void abs(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);
template<size_t N>
void norm(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);
template<size_t N>
void eigenvalues(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);
template<size_t N>
void percentile(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);

// Multiple operand filters
// elementwise multiply of top two fields on the stack.
// void multiply()

// Report filters
// Print the top of the stack.
template<size_t N>
void print(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg)
{
}

template<size_t N>
void rename(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg)
{
    vector<string> names;
    boost::split(names, arg, boost::is_any_of(","));
    if (names.size() > stack.size()) {
        throw runtime_error("Too many names provided to rename");
    }
    size_t pos = stack.size();
    for(const auto &name : names)
        stack[--pos]->name = name;
}

// Print the entire stack.

template<size_t N>
void execute(const string &mshFile, const vector<FilterInvocation> &filters) {
    MSHFieldParser<N> parser(mshFile);

    map<string, function<void(vector<VPtr<N> > &,
                const MSHFieldParser<N> &, const string &)> > filterLUT = {
        {"extract", extract<N>}, {"print", print<N>}, {"rename", rename<N>} };
    vector<VPtr<N> > stack;
    for (const auto &f : filters) {
        std::cout << "running " << f.first << "\t" << f.second << std::endl;
        filterLUT.at(f.first)(stack, parser, f.second);
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    auto parsedOptions = parseCmdLine(argc, argv);

    MeshIO_MSH io;
    vector<IOVertex>  v;
    vector<IOElement> e;

    string mshFile;
    vector<FilterInvocation> filters;
    for (const auto &opt : parsedOptions.options) {
        if (opt.string_key == "msh") mshFile = opt.value[0];
        else filters.push_back(make_pair(opt.string_key,
                    (opt.value.size() ? opt.value[0] : "")));
    }

    ifstream infile(mshFile);
    if (!infile.is_open()) throw runtime_error("Couldn't open " + mshFile);
    MeshType type = io.load(infile, v, e, MESH_GUESS);
    infile.close();
    size_t dim = (type == MESH_TET) ? 3 : 2;

    if (dim == 3) execute<3>(mshFile, filters);
    else          execute<2>(mshFile, filters);
    return 0;
}
