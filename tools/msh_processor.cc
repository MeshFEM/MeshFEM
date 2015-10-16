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
#include <MSHFieldWriter.hh>
#include <Types.hh>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include <iomanip>
#include <regex>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <cmath>
#include <cctype>
#include <memory>
#include <functional>
#include <limits>

using namespace MeshIO;

namespace po = boost::program_options;
using namespace std;

void usage(int status, const po::options_description &visible_opts) {
    cout << "Usage: msh_field_extractor in.msh [options]" << endl;
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

    po::options_description parser_operations("Data source operations");
    parser_operations.add_options()
        ("list",                           "List all fields in the msh")
        ("extract,e", po::value<string>(), "Extract field(s) matching a given name (or name pattern)")
        ("extractAll",                     "Extract all fields")
        ;
    po::options_description stack_operations("Stack operations");
    stack_operations.add_options()
        ("dup,D",                          "Duplicate top of the stack")
        ("pop,P",                          "Pop top of the stack")
        ("pull",      po::value<string>(), "Pull a named value to the top of the stack")
        ("push",      po::value<string>(), "Push a scalar literal to the top of the stack")
        ("print,p",                        "Print top of stack")
        ("printName",                      "Print name of value at the top of the stack")
        ("rename,r",  po::value<string>(), "Rename the field(s) at the top of the stack (multiple names separated by commas)")
        ("outMSH,o",  po::value<string>(), "Output msh file with named fields for each entry in the stack")
        ;
    po::options_description unary_operations("Unary operations");
    unary_operations.add_options()
        ("applyAll,A",                       "Apply next filter to entire stack instead of top")
        ("max,M",                            "Max of scalar field (pointwise for vector field)")
        ("min,m",                            "Min of scalar field (pointwise for vector field)")
        ("maxMag",                           "Max magnitude of scalar field (pointwise for vector field)")
        ("minMag",                           "Min magnitude of scalar field (pointwise for vector field)")
        ("norm,n",                           "L2 norm of scalar field (pointwise for vector field)")

        ("component,c", po::value<string>(), "Extract component of vector (pointwise for vector field)")

        ("abs,a",                            "Componentwise abs of scalar field or vector field")
        ("scale,s", po::value<string>(),     "Multiply the top of the stack by a scalar.")
        ("set",     po::value<string>(),     "Set every component of the top value to arg.")
        ("sum,S",                            "Sum the components of a (scalar|vector) field or vector")
        ("mean",                             "Element average of a {scalar,vector} field or vector")
        ("eigenvalues,l",                    "Eigenvalues of sym matrix field (vector field result)")
        ("sample",  po::value<string>(),     "Sample the value of a scalar/vector field at a point, "
                                             "using as piecewise constant interpolation on Voronoi diagram of points/element barycenters. "
                                             "The point is specified as a comma-separated vector")
        ("sampleIndex", po::value<string>(), "Sample the value of a scalar/vector field at a particular vertex/element index.")
        // ("percentile", po::value<double>(), "extract a certain percentile of the msh file")
        ;
    po::options_description binary_operations("Binary operations");
    binary_operations.add_options()
        ("add", "Add      the top two values on the stack")
        ("sub", "Subtract the top two values on the stack (prev - top)")
        ("mul", "Multiply the top two values on the stack")
        ("div", "Divide   the top two values on the stack (prev / top)")
        ;

    po::options_description cli_opts;
    cli_opts.add_options()("help,h", "Produce this help message");
    cli_opts.add(parser_operations).add(stack_operations).add(unary_operations)
            .add(binary_operations).add(hidden_opts);

    // Options visible in the help message.
    po::options_description visible_opts;
    visible_opts.add_options()("help,h", "Produce this help message");
    visible_opts.add(parser_operations).add(stack_operations)
           .add(unary_operations).add(binary_operations);

    shared_ptr<po::parsed_options> parsedOptions;
    try {
        parsedOptions = std::make_shared<po::parsed_options>(po::command_line_parser(argc, argv).
                            options(visible_opts).positional(p).run());
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
typedef ScalarField<Real> SField;
template<size_t N> using  VField =          VectorField<Real, N>;
template<size_t N> using SMField = SymmetricMatrixField<Real, N>;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vector;

// Forward declarations
template<size_t _N> struct  Value;

template<size_t N>
using VPtr = shared_ptr<Value<N> >; 

// Polymorphic binary operator functors.
struct BinaryOperator {                virtual Real operator()(Real a, Real b) = 0; };
struct AddOp : public BinaryOperator { virtual Real operator()(Real a, Real b) { return a + b; } };
struct SubOp : public BinaryOperator { virtual Real operator()(Real a, Real b) { return a - b; } };
struct MulOp : public BinaryOperator { virtual Real operator()(Real a, Real b) { return a * b; } };
struct DivOp : public BinaryOperator { virtual Real operator()(Real a, Real b) { return a / b; } };

template<size_t _N>
struct Value {
    constexpr static size_t N = _N;
    Value(const string &n) : name(n) { }
    string name;
    virtual size_t numElems() const = 0;
    virtual void scale(Real s) = 0;
    virtual void setTo(Real s) = 0;
    virtual void applyAbs() = 0;
    virtual VPtr<N> clone() const = 0;
    virtual VPtr<N> binaryOp(BinaryOperator &op, VPtr<N> b) const = 0;
    virtual void print(std::ostream &os = std::cout) const = 0;
    virtual DomainType  domainType() const { throw std::runtime_error("Error: non-field value."); }
    virtual DomainType &domainType()       { throw std::runtime_error("Error: non-field value."); }
    virtual VPtr<N> valueAtIndex(size_t i) const { throw std::runtime_error("Attempted to index non-field value"); }
};

// Various in-place absolute value functions.
template<class T> T absoluteValue(const T &in) { return in.cwiseAbs(); }
  Real absoluteValue(const Real   &in) { return std::abs(in); }

template<class T>
void setConstant(T &inout, Real val) { inout.setConstant(val); }
void setConstant(Real &inout, Real val) { inout = val; }

template<class T, size_t N, class Subtype>
struct SpecificValue : public Value<N> {
    typedef Value<N> Base;
    typedef T value_type;
    value_type value;
    SpecificValue(const string &n, const value_type &v) : Base(n), value(v) { }
    void scale(Real s) { value *= s; }
    void setTo(Real s) { setConstant(value, s); }
    void applyAbs()    { value = absoluteValue(value); }
    VPtr<N> clone() const { return make_shared<Subtype>(static_cast<const Subtype &>(*this)); }
    virtual void print(std::ostream &os = std::cout) const {
        os << value;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Field types (scalar, vector, and symmetric matrix fields)
////////////////////////////////////////////////////////////////////////////////
// Implementation of binary operations for field types.
// (to be specialized for each field type)
template<class FieldValueType, size_t N>
struct _binaryOpFieldImpl {
    static VPtr<N> apply(BinaryOperator &op, const FieldValueType &a, VPtr<N> b);
};

// Implementation of indexed lookup for field types.
// (to be specialized for each field type)
template<class FieldValueType, size_t N>
struct _valueAtIndexImpl {
    static VPtr<N> lookup(size_t i, const FieldValueType &a);
};

template<class FieldType, size_t N>
struct FieldValue : public SpecificValue<FieldType, N, FieldValue<FieldType, N>> {
    typedef SpecificValue<FieldType, N, FieldValue<FieldType, N>> Base;

    FieldValue(const string &n, const FieldType &val, DomainType dt = DomainType::UNKNOWN) : Base(n, val), m_domainType(dt) { }
    virtual DomainType  domainType() const { return m_domainType; }
    virtual DomainType &domainType()       { return m_domainType; }
    virtual size_t numElems() const { return this->value.domainSize(); }
    virtual VPtr<N> binaryOp(BinaryOperator &op, VPtr<N> b) const {
        return _binaryOpFieldImpl<FieldValue, N>::apply(op, *this, b);
    }
    // gets value at the index as a 
    virtual VPtr<N> valueAtIndex(size_t i) const {
        if (i >= this->value.domainSize()) throw std::runtime_error("Field value index out of bounds");
        VPtr<N> result = _valueAtIndexImpl<FieldValue, N>::lookup(i, *this);
        result->name = this->name + "(" + std::to_string(i) + ")";
        return result;
    }
private:
    DomainType m_domainType;
};

// Type aliases and forward declarations needed for binary operations
template<size_t _N> using  SFieldValue = FieldValue< SField,     _N>;
template<size_t _N> using  VFieldValue = FieldValue< VField<_N>, _N>;
template<size_t _N> using SMFieldValue = FieldValue<SMField<_N>, _N>;
template<size_t _N> struct ScalarValue;
template<size_t _N> struct VectorValue;

////////////////////////////////////////////////////////////////////////////////
// Partial specializations of indexed lookup for the different field types
////////////////////////////////////////////////////////////////////////////////
template<size_t N>
struct _valueAtIndexImpl<SFieldValue<N>, N> {
    static VPtr<N> lookup(size_t i, const SFieldValue<N> &a) { return make_shared<ScalarValue<N>>("", a.value(i)); }
};
template<size_t N>
struct _valueAtIndexImpl<VFieldValue<N>, N> {
    static VPtr<N> lookup(size_t i, const VFieldValue<N> &a) { return make_shared<VectorValue<N>>("", a.value(i)); }
};
template<size_t N>
struct _valueAtIndexImpl<SMFieldValue<N>, N> {
    static VPtr<N> lookup(size_t i, const SMFieldValue<N> &a) { throw std::runtime_error("Symmetric matrix type not yet implemented."); }
};

////////////////////////////////////////////////////////////////////////////////
// Partial specializations of binary operation for the different field types
////////////////////////////////////////////////////////////////////////////////
// Partial specialization for SFieldValue
template<size_t N>
struct _binaryOpFieldImpl<SFieldValue<N>, N> {
static VPtr<N> apply(BinaryOperator &op, const SFieldValue<N> &a, VPtr<N> b) {
    runtime_error illegal("Illegal arguments for binary operation");
    runtime_error domainMismatch("Domain mismatch in binary operation");
    runtime_error mismatch("Size mismatch in binary operation");
    // Scalar field-scalar field op
    if (auto sfValue = dynamic_pointer_cast<SFieldValue<N>>(b)) {
        if (sfValue->domainType() != a.domainType()) throw domainMismatch;
        if (sfValue->numElems()   != a.numElems())   throw mismatch;
        auto result = make_shared<SFieldValue<N>>("result", SField(a.value), a.domainType());
        for (size_t i = 0; i < a.numElems(); ++i)
            result->value[i] = op(a.value[i], sfValue->value[i]);
        return result;
    }
    // Scalar field-vector field op
    else if (auto vfValue = dynamic_pointer_cast<VFieldValue<N>>(b)) {
        if (vfValue->domainType() != a.domainType()) throw domainMismatch;
        if (vfValue->numElems()   != a.numElems())   throw mismatch;
        auto result = make_shared<VFieldValue<N>>("result", VField<N>(vfValue->value), a.domainType());
        for (size_t i = 0; i < a.numElems(); ++i)
            for (size_t c = 0; c < vfValue->value.dim(); ++c)
                result->value(i)(c) = op(a.value[i], vfValue->value(i)(c));
        return result;
    }
    // Scalar field-matrix field op
    else if (auto smfValue = dynamic_pointer_cast<SMFieldValue<N>>(b)) {
        if (smfValue->domainType() != a.domainType()) throw domainMismatch;
        if (smfValue->numElems()   != a.numElems())   throw mismatch;
        auto result = make_shared<SMFieldValue<N>>("result", SMField<N>(smfValue->value), a.domainType());
        for (size_t i = 0; i < a.numElems(); ++i)
            for (size_t c = 0; c < smfValue->value.dim(); ++c)
                result->value(i)[c] = op(a.value[i], smfValue->value(i)[c]);
        return result;
    }
    // Scalar field-scalar op
    else if (auto sValue = dynamic_pointer_cast<ScalarValue<N>>(b)) {
        auto result = make_shared<SFieldValue<N>>("result", SField(a.value), a.domainType());
        for (size_t i = 0; i < a.numElems(); ++i)
            result->value[i] = op(a.value[i], sValue->value);
        return result;
    }

    throw illegal;
}
};

// Partial specialization for VFieldValue
template<size_t N>
struct _binaryOpFieldImpl<VFieldValue<N>, N> {
static VPtr<N> apply(BinaryOperator &op, const VFieldValue<N> &a, VPtr<N> b) {
    runtime_error illegal("Illegal arguments for binary operation");
    runtime_error domainMismatch("Domain mismatch in binary operation");
    runtime_error mismatch("Size mismatch in binary operation");
    auto result = make_shared<VFieldValue<N>>("result", VField<N>(a.value), a.domainType());
    // Vector field-scalar field op
    if (auto sfValue = dynamic_pointer_cast<SFieldValue<N>>(b)) {
        if (sfValue->domainType() != a.domainType()) throw domainMismatch;
        if (sfValue->numElems()   != a.numElems())   throw mismatch;
        for (size_t i = 0; i < a.numElems(); ++i)
            for (size_t c = 0; c < a.value.dim(); ++c)
                result->value(i)(c) = op(a.value(i)(c), sfValue->value[i]);
    }
    // Vector field-vector field op
    else if (auto vfValue = dynamic_pointer_cast<VFieldValue<N>>(b)) {
        if (vfValue->domainType() != a.domainType()) throw domainMismatch;
        if (vfValue->numElems()   != a.numElems())   throw mismatch;
        for (size_t i = 0; i < a.numElems(); ++i)
            for (size_t c = 0; c < vfValue->value.dim(); ++c)
                result->value(i)(c) = op(a.value(i)(c), vfValue->value(i)(c));
    }
    // Vector field-scalar op
    else if (auto sValue = dynamic_pointer_cast<ScalarValue<N>>(b)) {
        for (size_t i = 0; i < a.numElems(); ++i)
            for (size_t c = 0; c < a.value.dim(); ++c)
                result->value(i)(c) = op(a.value(i)(c), sValue->value);
    }

    throw illegal;
}
};

// Partial specialization for SMFieldValue
template<size_t N>
struct _binaryOpFieldImpl<SMFieldValue<N>, N> {
static VPtr<N> apply(BinaryOperator &op, const SMFieldValue<N> &a, VPtr<N> b) {
    runtime_error illegal("Illegal arguments for binary operation");
    runtime_error domainMismatch("Domain mismatch in binary operation");
    runtime_error mismatch("Size mismatch in binary operation");
    auto result = make_shared<SMFieldValue<N>>("result", SMField<N>(a.value), a.domainType());
    // matrix field-scalar field op
    if (auto sfValue = dynamic_pointer_cast<SFieldValue<N>>(b)) {
        if (sfValue->domainType() != a.domainType()) throw domainMismatch;
        if (sfValue->numElems()   != a.numElems())   throw mismatch;
        for (size_t i = 0; i < a.numElems(); ++i)
            for (size_t c = 0; c < a.value.dim(); ++c)
                result->value(i)[c] = op(a.value(i)[c], sfValue->value[i]);
        return result;
    }
    // matrix field-scalar op
    else if (auto sValue = dynamic_pointer_cast<ScalarValue<N>>(b)) {
        for (size_t i = 0; i < a.numElems(); ++i)
            for (size_t c = 0; c < a.value.dim(); ++c)
                result->value(i)[c] = op(a.value(i)[c], sValue->value);
        return result;
    }
    throw illegal;
}
};

////////////////////////////////////////////////////////////////////////////////
// Vector and Scalar types
////////////////////////////////////////////////////////////////////////////////
template<size_t N>
struct VectorValue : public SpecificValue<Vector, N, VectorValue<N>> {
    VectorValue(const string &n, const Vector &v = Vector())
        : SpecificValue<Vector, N, VectorValue>(n, v) { }
    virtual size_t numElems() const { return this->value.rows(); }
    virtual VPtr<N> binaryOp(BinaryOperator &op, VPtr<N> b) const {
        runtime_error illegal("Illegal arguments for binary operation");
        runtime_error mismatch("Size mismatch in binary operation");
        // Vector-vector field op
        if (auto vfValue = dynamic_pointer_cast<VFieldValue<N>>(b)) {
            auto result = make_shared<VFieldValue<N>>("result", Vector(this->value), vfValue->domainType());
            if (this->numElems() != vfValue->value.dim()) throw mismatch;
            for (size_t i = 0; i < vfValue->numElems(); ++i)
                for (size_t c = 0; c < this->numElems(); ++c)
                    result->value(i)[c] = op(this->value[c], vfValue->value(i)[c]);
        }
        // Vector-vector op
        else if (auto vValue = dynamic_pointer_cast<VectorValue<N>>(b)) {
            auto result = make_shared<VectorValue>("result", Vector(this->value));
            if (this->numElems() != vValue->numElems()) throw mismatch;
            for (size_t c = 0; c < this->numElems(); ++c)
                result->value[c] = op(this->value[c], vValue->value[c]);
        }
        // Vector-scalar op
        else if (auto sValue = dynamic_pointer_cast<ScalarValue<N>>(b)) {
            auto result = make_shared<VectorValue>("result", Vector(this->value));
            for (size_t c = 0; c < this->numElems(); ++c)
                result->value[c] = op(this->value[c], sValue->value);
            return result;
        }

        throw illegal;
    }
    virtual void print(std::ostream &os = std::cout) const { os << this->value << std::endl; }
};

template<size_t N>
struct ScalarValue : public SpecificValue<Real, N, ScalarValue<N>> {
    ScalarValue(const string &n, const Real &v = Real())
        : SpecificValue<Real, N, ScalarValue>(n, v) { }
    virtual size_t numElems() const { return 1; }
    virtual VPtr<N> binaryOp(BinaryOperator &op, VPtr<N> b) const {
        runtime_error illegal("Illegal arguments for binary operation");
        runtime_error mismatch("Size mismatch in binary operation");
        // Scalar-scalar op
        if (auto sValue = dynamic_pointer_cast<ScalarValue>(b)) {
            return make_shared<ScalarValue>("result",
                    op(this->value, sValue->value));
        }
        // Scalar-scalar field op
        if (auto sfValue = dynamic_pointer_cast<SFieldValue<N>>(b)) {
            auto result = make_shared<SFieldValue<N>>("result", SField(sfValue->value), sfValue->domainType());
            for (size_t i = 0; i < sfValue->numElems(); ++i)
                result->value[i] = op(this->value, sfValue->value[i]);
            return result;
        }
        // Scalar-vector field op
        else if (auto vfValue = dynamic_pointer_cast<VFieldValue<N>>(b)) {
            auto result = make_shared<VFieldValue<N>>("result", VField<N>(vfValue->value), vfValue->domainType());
            for (size_t i = 0; i < vfValue->numElems(); ++i)
                for (size_t c = 0; c < vfValue->value.dim(); ++c)
                    result->value(i)(c) = op(this->value, vfValue->value(i)(c));
            return result;
        }
        // Scalar-matrix field op
        else if (auto smfValue = dynamic_pointer_cast<SMFieldValue<N>>(b)) {
            auto result = make_shared<SMFieldValue<N>>("result", SMField<N>(smfValue->value), smfValue->domainType());
            for (size_t i = 0; i < smfValue->numElems(); ++i)
                for (size_t c = 0; c < smfValue->value.dim(); ++c)
                    result->value(i)[c] = op(this->value, smfValue->value(i)[c]);
            return result;
        }
        // Scalar-vector op
        else if (auto vValue = dynamic_pointer_cast<VectorValue<N>>(b)) {
            auto result = make_shared<VectorValue<N>>("result", Vector(vValue->value));
            for (size_t c = 0; c < vValue->numElems(); ++c)
                result->value[c] = op(this->value, result->value[c]);
            return result;
        }

        throw illegal;
    }
    virtual void print(std::ostream &os = std::cout) const { os << this->value << std::endl; }
};

////////////////////////////////////////////////////////////////////////////////
// Filters - operate on the stack.
// These are all template functions with the signature:
// template<size_t N>
// void f(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg)
////////////////////////////////////////////////////////////////////////////////
// Filter invocation: (name, argument string)
typedef pair<string, string> FilterInvocation;
// Data source filters
// Extract field(s) matching the pattern in "arg", pushing them on the top of
// the stack.
template<size_t N>
void extract(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
             const string &arg) {
    std::regex pattern(arg);
    size_t numMatched = 0;
    DomainType dtype;
    for (const string &name : parser.scalarFieldNames()) {
        if (regex_match(name, pattern)) {
            const auto &sf = parser.scalarField(name, DomainType::ANY, dtype);
            stack.push_back(VPtr<N>(new SFieldValue<N>(name, sf, dtype)));
            ++numMatched;
        }
    }
    for (const string &name : parser.vectorFieldNames()) {
        if (regex_match(name, pattern)) {
            const auto &vf = parser.vectorField(name, DomainType::ANY, dtype);
            stack.push_back(VPtr<N>(new VFieldValue<N>(name, vf, dtype)));
            ++numMatched;
        }
    }
    for (const string &name : parser.symmetricMatrixFieldNames()) {
        if (regex_match(name, pattern)) {
            const auto &smf = parser.symmetricMatrixField(name, DomainType::ANY, dtype);
            stack.push_back(VPtr<N>(new SMFieldValue<N>(name, smf, dtype)));
            ++numMatched;
        }
    }
    if (numMatched == 0) throw runtime_error("No fields matched '" + arg + "'");
}

template<size_t N>
void extractAll(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
             const string &arg) {
    DomainType dtype;
    for (const string &name : parser.scalarFieldNames()) {
        const auto &sf = parser.scalarField(name, DomainType::ANY, dtype);
        stack.push_back(VPtr<N>(new SFieldValue<N>(name, sf, dtype)));
    }
    for (const string &name : parser.vectorFieldNames()) {
        const auto &vf = parser.vectorField(name, DomainType::ANY, dtype);
        stack.push_back(VPtr<N>(new VFieldValue<N>(name, vf, dtype)));
    }
    for (const string &name : parser.symmetricMatrixFieldNames()) {
        const auto &smf = parser.symmetricMatrixField(name, DomainType::ANY, dtype);
        stack.push_back(VPtr<N>(new SMFieldValue<N>(name, smf, dtype)));
    }
}

template<size_t N>
VPtr<N> getValue(vector<VPtr<N> > &stack, size_t offset = 0) {
    if (stack.size() <= offset) throw std::runtime_error("Accessed out of stack bounds.");
    size_t idx = stack.size() - 1 - offset;
    return stack.at(idx);
}
template<size_t N>
VPtr<N> popValue(vector<VPtr<N> > &stack) {
    auto val = getValue(stack);
    stack.pop_back();
    return val;
}

template<typename T>
shared_ptr<T> getTypedValue(vector<VPtr<T::N> > &stack, size_t offset = 0) {
    VPtr<T::N> val = getValue(stack, offset);
    shared_ptr<T> tVal = dynamic_pointer_cast<T>(val);
    if (!tVal) { throw runtime_error("Invalid argument."); }
    return tVal;
}

template<typename T>
shared_ptr<T> popTypedValue(vector<VPtr<T::N> > &stack) {
    auto tVal = getTypedValue<T>(stack);
    stack.pop_back();
    return tVal;
}

double getDoubleArg(const string &arg) {
    size_t end;
    double factor = stod(arg, &end);
    for (char c : arg.substr(end))
        if (!isspace(c)) throw runtime_error("Argument must be a real number");
    return factor;
}

// Parse a comma-separated vector from a string
// Throws exception if the parsed size is not N.
template<size_t N>
VectorND<N> getVectorArg(string arg) {
    vector<string> argComponents;
    boost::trim(arg);
    boost::split(argComponents, arg, boost::is_any_of(","), boost::token_compress_on);
    if (argComponents.size() != N) throw std::runtime_error("Invalid vector argument size");
    VectorND<N> result;
    for (size_t i = 0; i < N; ++i)
        result[i] = std::stod(argComponents[i]);
    return result;
}

template<size_t N>
void dup(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
         const string &arg) { stack.push_back(getValue(stack)->clone()); }
template<size_t N>
void pop(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
         const string &arg) { popValue(stack); }
template<size_t N>
void push(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
         const string &arg) {
    double d = getDoubleArg(arg);
    stack.push_back(make_shared<ScalarValue<N>>(to_string(d), d));
}
template<size_t N>
void pull(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
         const string &arg) {
    for (auto it = stack.begin(); it != stack.end(); ++it) {
        if ((*it)->name == arg) {
            auto val = *it;
            stack.erase(it);
            stack.push_back(val);
            return;
        }
    }
    throw runtime_error("Couldn't find '" + arg + "' for pull.");
}

// Single operand filters

// partialReduction:
// Operations reducing vector fields to scalar fields, scalar fields to scalars,
// and vectors to scalars.
template<size_t N>
void partialReduction(const string &op, vector<VPtr<N> > &stack,
                      const MSHFieldParser<N> &parser, const string &arg) {
    auto top = popValue(stack);
    string name = op + "(" + top->name + ")";
    // NOTE: we could cast all quantities to ScalarField and use only the sfOps
    // table, but that would require a bunch of memory allocations/copies.
    static const map<string, function<Real(const SField &f)>> sfOps = {
        { "min",    [](const SField &f) -> Real { return f.min(); } },
        { "max",    [](const SField &f) -> Real { return f.max(); } },
        { "minMag", [](const SField &f) -> Real { return f.minMag(); } },
        { "maxMag", [](const SField &f) -> Real { return f.maxMag(); } },
        { "norm",   [](const SField &f) -> Real { return f.norm(); } } };
    static const map<string, function<Real(const Vector &)>> vOps = {
        { "min",    [](const Vector &v) -> Real { return v.minCoeff(); } },
        { "max",    [](const Vector &v) -> Real { return v.maxCoeff(); } },
        { "minMag", [](const Vector &v) -> Real { Real m = v.minCoeff(), M = v.maxCoeff(); return (std::abs(m) < M) ? m : M; } },
        { "maxMag", [](const Vector &v) -> Real { Real m = v.minCoeff(), M = v.maxCoeff(); return (std::abs(m) > M) ? m : M; } },
        { "norm",   [](const Vector &v) -> Real { return v.norm(); } } };
    if (auto sfVal = dynamic_pointer_cast<SFieldValue<N>>(top)) {
        stack.push_back(VPtr<N>(new ScalarValue<N>(name, sfOps.at(op)(sfVal->value))));
    }
    else if (auto vfVal = dynamic_pointer_cast<VFieldValue<N>>(top)) {
        auto vOp = vOps.at(op);
        auto r = new SFieldValue<N>(name, SField(vfVal->value.domainSize()), vfVal->domainType());
        for (size_t i = 0; i < vfVal->value.domainSize(); ++i)
            r->value[i] = vOp(vfVal->value(i));
        stack.push_back(VPtr<N>(r));
    }
    else if (auto vVal = dynamic_pointer_cast<VectorValue<N>>(top)) {
        stack.push_back(VPtr<N>(new ScalarValue<N>(name, vOps.at(op)(vVal->value))));
    }
    else { throw runtime_error("Invalid argument."); }
}

template<size_t N>
void binaryOperator(const string &op, vector<VPtr<N> > &stack,
                    const MSHFieldParser<N> &parser, const string &arg) {
    // Top of stack is the second operand, next in stack is the first
    auto b = popValue(stack);
    auto a = popValue(stack);
    static const map<string, shared_ptr<BinaryOperator>> opLUT = {
        { "+", make_shared<AddOp>()}, { "-", make_shared<SubOp>()},
        { "*", make_shared<MulOp>()}, { "/", make_shared<DivOp>()} };
    auto result = a->binaryOp(*opLUT.at(op), b);
    result->name = a->name + " " + op + " " + b->name;
    stack.push_back(result);
}

// Scalar/vector mean and sum
// Scalar field becomes scalar, vector field becomes vector,
// vector becomes scalar
template<size_t N>
void sum(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg) {
    auto top = popValue(stack);
    string name = "sum(" + top->name + ")";
    if (auto sfVal = dynamic_pointer_cast<SFieldValue<N>>(top))
        stack.push_back(VPtr<N>(new ScalarValue<N>(name, sfVal->value.sum())));
    else if (auto vfVal = dynamic_pointer_cast<VFieldValue<N>>(top)) {
        auto r = new VectorValue<N>(name);
        r->value.setZero(vfVal->value.dim());
        for (size_t i = 0; i < vfVal->value.domainSize(); ++i) {
            r->value += vfVal->value(i);
        }
        stack.push_back(VPtr<N>(r));
    }
    else if (auto vVal = dynamic_pointer_cast<VectorValue<N>>(top)) {
        stack.push_back(VPtr<N>(new ScalarValue<N>(name, vVal->value.sum())));
    }
    else { throw runtime_error("Invalid argument."); }
}

template<size_t N>
void mean(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg) {
    Real numElems = getValue(stack)->numElems();
    sum(stack, parser, arg);
    auto result = getValue(stack);
    result->scale(1 / numElems);
    result->name = result->name.replace(0, 3, "mean");
}

template<size_t N>
void scale(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
           const string &arg) {
    auto top = getValue(stack);
    Real factor = getDoubleArg(arg);
    top->scale(factor);
    top->name = to_string(factor) + " * (" + top->name + ")";
}

template<size_t N>
void setComponents(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser,
           const string &arg) {
    auto top = getValue(stack);
    size_t end;
    double factor = stod(arg, &end);
    for (char c : arg.substr(end))
        if (!isspace(c)) throw runtime_error("Set filter's argument must be a real number");
    top->setTo(factor);
    top->name = arg.substr(0, end);
}

// Component-wise abs
// Data types are unchanged.
template<size_t N>
void abs(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg) {
    auto top = getValue(stack);
    top->applyAbs();
    top->name = "abs(" + top->name + ")";
}

template<size_t N>
void eigenvalues(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg) {
    auto top = popTypedValue<SMFieldValue<N> >(stack);
    size_t numValues = top->value.domainSize();
    auto *result = new VFieldValue<N>("eigenvalues(" + top->name + ")", VField<N>(numValues), top->domainType());
    for (size_t i = 0; i < numValues; ++i)
        result->value(i) = top->value(i).eigenvalues();
    stack.push_back(shared_ptr<VFieldValue<N> >(result));
}

template<size_t N>
void component(vector<VPtr<N>> &stack, const MSHFieldParser<N> &parser, const string &arg) {
    size_t c = std::stoi(arg);
    if (c >= N) throw std::runtime_error("Component index out of range");

    VPtr<N> v = popValue(stack);
    if (auto vf = dynamic_pointer_cast<VFieldValue<N>>(v)) {
        SField result(vf->numElems());
        for (size_t i = 0; i < vf->numElems(); ++i)
            result[i] = vf->value(i)[c];
        stack.push_back(make_shared<SFieldValue<N>>(v->name + "[" + to_string(c) + "]", result, vf->domainType()));
    }
    else if (auto vector = dynamic_pointer_cast<VectorValue<N>>(v)) {
        stack.push_back(make_shared<ScalarValue<N>>(v->name + "[" + to_string(c) + "]", vector->value[c]));
    }
    else throw std::runtime_error("Component extraction only applies to vector and vector fields.");
}

// Sample a field at the vertex/element specified by index encoded in "arg".
template<size_t N>
void sampleIndex(vector<VPtr<N>> &stack, const MSHFieldParser<N> &parser, const string &arg) {
    size_t i = std::stoi(arg);
    VPtr<N> top = popValue(stack);
    stack.push_back(top->valueAtIndex(i));
}

// Sample a field at the point specified by vector encoded in "arg".
// The field is interpolated as piecewise constant on the Voronoi diagram of
// vertices (for per-vertex fields) or element barycenters (for per-element fields)
template<size_t N>
void sample(vector<VPtr<N>> &stack, const MSHFieldParser<N> &parser, const string &arg) {
    Point3D p = padTo3D(getVectorArg<N>(arg));
    VPtr<N> top = popValue(stack);
    DomainType dt = top->domainType();

    const auto &vertices = parser.vertices();
    const auto &elements = parser.elements();

    vector<Real> sqDistances;
    if (dt == DomainType::PER_NODE) {
        // Distance is to closest vertex
        sqDistances.reserve(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i)
            sqDistances.push_back((vertices[i].point - p).squaredNorm());
    }
    else if (dt == DomainType::PER_ELEMENT) {
        // Distance is to closest element barycenter
        sqDistances.reserve(elements.size());
        for (size_t i = 0; i < elements.size(); ++i) {
            Point3D barycenter(Point3D::Zero());
            size_t ncorners = elements[i].size();
            for (size_t ci = 0; ci < ncorners; ++ci)
                barycenter += vertices[elements[i][ci]].point;
            barycenter *= (1.0 / ncorners);
            sqDistances.push_back((barycenter - p).squaredNorm());
        }
    }
    else throw std::runtime_error("Illegal field domain type.");

    size_t sampleIndex = 0;
    Real closestSqDist = std::numeric_limits<Real>::max();
    Real secondClosestSqDist = closestSqDist;
    for (size_t i = 0; i < sqDistances.size(); ++i) {
        if (sqDistances[i] < closestSqDist) {
            sampleIndex = i;
            secondClosestSqDist = closestSqDist;
            closestSqDist = sqDistances[i];
        }
        else secondClosestSqDist = std::min(secondClosestSqDist, sqDistances[i]);
    }
    if (sqrt(closestSqDist) > 0.25 * sqrt(secondClosestSqDist))
        std::cerr << "WARNING: sampling far away from interpolation point." << std::endl;

    stack.push_back(top->valueAtIndex(sampleIndex));
}

template<size_t N>
void percentile(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg);

// Multiple operand filters
// component-wise multiply of top two fields on the stack.
// void multiply()

// Report filters
// List all fields parsed
template<size_t N>
void listNames(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg) {
    for (const string &name : parser.scalarFieldNames()) {
        cout << "s\t" << name << endl;
    }
    for (const string &name : parser.vectorFieldNames()) {
        cout << "v\t" << name << endl;
    }
    for (const string &name : parser.symmetricMatrixFieldNames()) {
        cout << "sm\t" << name << endl;
    }
}

// Print the top of the stack.
template<size_t N>
void print(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg)
{
    getValue(stack)->print();
}

template<size_t N>
void printName(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg) {
    cout << getValue(stack)->name << endl;
}

template<size_t N>
void outputMSH(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg) {
    MSHFieldWriter writer(arg, parser.vertices(), parser.elements());
    for (auto s : stack) {
        if      (auto  sfVal = dynamic_pointer_cast<SFieldValue<N>>(s))
            writer.addField(s->name,  sfVal->value, DomainType::GUESS);
        else if (auto  vfVal = dynamic_pointer_cast<VFieldValue<N>>(s))
            writer.addField(s->name,  vfVal->value, DomainType::GUESS);
        else if (auto smfVal = dynamic_pointer_cast<SMFieldValue<N>>(s))
            writer.addField(s->name, smfVal->value, DomainType::GUESS);
        else cout << "WARNING: ignored non-field value on stack" << endl;
    }
}

template<size_t N>
void rename(vector<VPtr<N> > &stack, const MSHFieldParser<N> &parser, const string &arg) {
    vector<string> names;
    boost::split(names, arg, boost::is_any_of(","));
    if (names.size() > stack.size()) {
        throw runtime_error("Too many names provided to rename");
    }
    size_t pos = stack.size();
    for(const auto &name : names)
        stack[--pos]->name = name;
}

template<size_t N>
void execute(const string &mshFile, const vector<FilterInvocation> &filters) {
    MSHFieldParser<N> parser(mshFile);

    cout << std::scientific << std::setprecision(16);

    using namespace std::placeholders;
    map<string, function<void(vector<VPtr<N> > &,
                const MSHFieldParser<N> &, const string &)> > filterLUT = {
        {"list", listNames<N>}, {"extract", extract<N>},
        {"extractAll", extractAll<N>}, {"print", print<N>},
        {"printName", printName<N>}, {"rename", rename<N>},
        {"eigenvalues", eigenvalues<N>},
        {"sample", sample<N>}, {"sampleIndex", sampleIndex<N>},
        {"min",    bind(partialReduction<N>, "min",    _1, _2, _3)},
        {"max",    bind(partialReduction<N>, "max",    _1, _2, _3)},
        {"norm",   bind(partialReduction<N>, "norm",   _1, _2, _3)},
        {"maxMag", bind(partialReduction<N>, "maxMag", _1, _2, _3)},
        {"minMag", bind(partialReduction<N>, "minMag", _1, _2, _3)},
        {"component", component<N>},
        {"sum", sum<N>}, {"mean", mean<N>}, {"abs", abs<N>},
        {"scale", scale<N>}, {"set", setComponents<N>},
        {"add", bind(binaryOperator<N>, "+", _1, _2, _3)},
        {"sub", bind(binaryOperator<N>, "-", _1, _2, _3)},
        {"mul", bind(binaryOperator<N>, "*", _1, _2, _3)},
        {"div", bind(binaryOperator<N>, "/", _1, _2, _3)},
        {"dup", dup<N>}, {"pop", pop<N>}, {"push", push<N>}, {"pull", pull<N>},
        {"outMSH", outputMSH<N>},
    };

    // The following commands suppress automatic output of stack at exit
    set<string> noImplicitPrint = { "noprint", "print", "outMSH" };

    vector<VPtr<N> > stack;
    for (size_t fi = 0; fi < filters.size(); ++fi) {
        const auto &f = filters[fi];
        try {
            // applyAll is special: apply next filter to each entry of stack
            if (f.first == "applyAll") {
                auto err = runtime_error("must be followed by a plain filter");
                if (fi == filters.size() - 1) throw err;
                const auto &nf = filters[++fi];
                if (filterLUT.count(nf.first) == 0) throw err;
                // Apply to all S elements on the stack by splitting into S auxiliary
                // stacks and recombining
                vector<VPtr<N>> auxiliaryStack, newStack;
                for (auto sval : stack) {
                    auxiliaryStack.assign(1, sval);
                    filterLUT.at(nf.first)(auxiliaryStack, parser, nf.second);
                    for (auto asval : auxiliaryStack) newStack.push_back(asval);
                }
                stack = newStack;
            }
            else filterLUT.at(f.first)(stack, parser, f.second);
        }
        catch (const exception &e) {
            cout << "Filter '" << f.first << "' failed: " << e.what() << endl;
            exit(-1);
        }
    }
    // implicit list when filters are empty
    if (filters.size() == 0) listNames<N>(stack, parser, "");
    else if (noImplicitPrint.count(filters.back().first) == 0) {
        if (stack.size() > 0) print<N>(stack, parser, "");
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
