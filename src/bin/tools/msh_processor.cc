////////////////////////////////////////////////////////////////////////////////
// msh_processor.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extracts and processes fields from a .msh file. The field processor is
//      essentually an RPN evaluator that maintains a stack of values and
//      applies the filters specified on the command line in order.
//
//      Component-wise Binary Operations:
//          add, sub, mul, div
//      Component-wise Unary Operations:
//          abs, scale, set
//      (Partial) reductions:
//          min, max, minMag, maxMag, norm, index, sum, mean
//      Reductions can be done in multiple wasy for multi-indexed objects
//      (e.g. vector fields, vector interpolants, fields of vector interpolants).
//      By default, reductions are done over the innermost index
//      ("pointwise" for fields/interpolants)
//          Field<NonScalar>       -> Field<Reduced>       (recursive)
//          Interpolant<NonScalar> -> Interpolant<Reduced> (recursive)
//          Field<Scalar>          -> Scalar
//          Interpolant<Scalar>    -> Scalar
//          PointValue             -> Scalar
//      When "outer reduction" mode is requested, reductions are done over the
//      outer index (per-component reduction for vector fields/interpolants).
//      Note: this is probably the more natural action for sum, mean of vector
//      fields.
//          Field<NonScalar>     -> Scalar
//          Field<Scalar>        -> Scalar
//          Interpolant<Scalar>  -> Scalar
//          Field<T>             -> T
//          Interpolant<T>       -> T
//          PointValue           -> Scalar
//      Warning: operations treat interpolants as vectors of nodal values, so
//      for interpolants with negative weights component-wise min/max won't
//      necessarily be the min/max over the simplex.
//
//      TODO: store element *index* on interpolant: binary operations can only
//      act on a pair of interpolants with matching element index.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/16/2014 15:26:04
////////////////////////////////////////////////////////////////////////////////
#include "argparse.hh"
#include "Sampler.hh"
#include "Values.hh"
#include "MeshConnectivity.hh"
#include <MeshFEM/ExpressionVector.hh>
#include <MeshFEM/filters/remove_dangling_vertices.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/SimplicialMesh.hh>
#include <MeshFEM/Types.hh>
#include <MeshFEM/VonMises.hh>

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
#include <sstream>

//using namespace MeshIO;
using namespace std;

// Global parsers used for 2 and 3D cases.
unique_ptr<MSHFieldParser<2>> g_parser2D;
unique_ptr<MSHFieldParser<3>> g_parser3D;

template<size_t N, typename... Args>
void parseMSH(Args&&... args) {
    static_assert(N == 2 || N == 3, "Invalid parser dimension");
    if (N == 2) g_parser2D = make_unique<MSHFieldParser<2>>(std::forward<Args>(args)...);
    else        g_parser3D = make_unique<MSHFieldParser<3>>(std::forward<Args>(args)...);
}

template<size_t N>
const MSHFieldParser<N> &getParser();
template<> const MSHFieldParser<2> &getParser<2>() { return *g_parser2D; }
template<> const MSHFieldParser<3> &getParser<3>() { return *g_parser3D; }

template<size_t N>
MSHFieldParser<N> &getMutableParser();
template<>       MSHFieldParser<2> &getMutableParser<2>() { return *g_parser2D; }
template<>       MSHFieldParser<3> &getMutableParser<3>() { return *g_parser3D; }

// Global lazily-constructed element samplers used for 2 and 3D cases.
unique_ptr<ElementSampler::Sampler<2>> g_sampler2D;
unique_ptr<ElementSampler::Sampler<3>> g_sampler3D;

template<size_t N>
const ElementSampler::Sampler<N> &getElementSampler();
template<> const ElementSampler::Sampler<2> &getElementSampler<2>() { if (g_sampler2D) return *g_sampler2D; g_sampler2D = make_unique<ElementSampler::Sampler<2>>(g_parser2D->vertices(), g_parser2D->elements()); return *g_sampler2D; }
template<> const ElementSampler::Sampler<3> &getElementSampler<3>() { if (g_sampler3D) return *g_sampler3D; g_sampler3D = make_unique<ElementSampler::Sampler<3>>(g_parser3D->vertices(), g_parser3D->elements()); return *g_sampler3D; }

// Global lazily-constructed mesh data structures for 2 and 3D cases.
unique_ptr<SimplicialMesh<2>> g_triMesh;
unique_ptr<SimplicialMesh<3>> g_tetMesh;

template<size_t N>
const SimplicialMesh<N> &getMeshDS();
template<> const SimplicialMesh<2> &getMeshDS<2>() { if (g_triMesh) return *g_triMesh; else g_triMesh = make_unique<SimplicialMesh<2>>(g_parser2D->elements(), g_parser2D->vertices().size()); return *g_triMesh; }
template<> const SimplicialMesh<3> &getMeshDS<3>() { if (g_tetMesh) return *g_tetMesh; else g_tetMesh = make_unique<SimplicialMesh<3>>(g_parser3D->elements(), g_parser3D->vertices().size()); return *g_tetMesh; }

////////////////////////////////////////////////////////////////////////////////
// Stack operations
////////////////////////////////////////////////////////////////////////////////
using Stack = vector<NamedValue>;

struct Modifiers {
    bool outerReduction = false;
    bool applyAll = false;
};

NamedValue &getValue(Stack &stack, size_t offset = 0) {
    if (stack.size() <= offset) throw std::runtime_error("Accessed out of stack bounds.");
    size_t idx = stack.size() - 1 - offset;
    return stack.at(idx);
}

NamedValue popValue(Stack &stack) {
    if (stack.empty()) throw std::runtime_error("Tried to pop from empty stack.");
    NamedValue val = std::move(getValue(stack));
    stack.pop_back();
    return val;
}

template<typename T>
TypedNamedValue<T> &getTypedValue(Stack &stack, size_t offset = 0) {
    return TypedNamedValue<T>(getValue(stack, offset));
}

template<typename T>
TypedNamedValue<T> popTypedValue(Stack &stack) {
    if (stack.empty()) throw std::runtime_error("Tried to pop from empty stack.");
    NamedValue val = std::move(stack.back());
    stack.pop_back();
    return val;
}

////////////////////////////////////////////////////////////////////////////////
// Filters - operate on the stack.
// These are all template functions with the signature:
// template<size_t N>
// size_t f(const string &op, const string &arg, Stack &stack, const Modifiers &m)
//
// Filters return the number of results they pushed onto the stack. Filters are
// considered to first pop all of their arguments then push "results" onto the
// stack. For example, the print operation that acts on the stack top but
// does not modify the stack is considered to have one result (it pops and
// pushes back the argument). The reverse operation, which acts on the whole
// stack, returns the stack size.
//
// Placed in their own namespace because their names overload those defined
// elsewhere, causing problems with the lookup table definition
////////////////////////////////////////////////////////////////////////////////
namespace Filter {
// Data source filters
// Extract field(s) matching the pattern in "arg", pushing them on the top of
// the stack.
size_t pushScalarField(Stack &stack, const string &name, const ScalarField<Real> &sf, const DomainType &dtype) {
    TypedNamedValue<FSValue> sfv(name, dtype, sf.domainSize());
    for (size_t i = 0; i < sf.domainSize(); ++i)
        sfv->value[i] = SValue(sf[i]);
    stack.push_back(std::move(sfv));
    return 1;
}

template<size_t N>
size_t pushVectorField(Stack &stack, const string &name, const VectorField<Real, N> &vf, const DomainType &dtype) {
    TypedNamedValue<FVValue> vfv(name, dtype, vf.domainSize());
    for (size_t i = 0; i < vf.domainSize(); ++i)
        vfv->value[i] = VValue(vf(i).eval());
    stack.push_back(std::move(vfv));
    return 1;
}

template<size_t N>
size_t pushSymmetricMatrixField(Stack &stack, const string &name, const SymmetricMatrixField<Real, N> &smf, const DomainType &dtype) {
    TypedNamedValue<FSMValue> smfv(name, dtype, smf.domainSize());
    for (size_t i = 0; i < smf.domainSize(); ++i)
        smfv->value[i] = SMValue(smf(i));
    stack.push_back(std::move(smfv));
    return 1;
}

template<class IFType, class RawType>
size_t pushInterpolantField(Stack &stack, const string &name, const RawType &raw_if, const DomainType &dtype) {
    TypedNamedValue<IFType> ifv(name, dtype, raw_if.size());
    for (size_t i = 0; i < raw_if.size(); ++i)
        ifv->value[i] = raw_if[i];
    stack.push_back(std::move(ifv));
    return 1;
}

template<size_t N>
size_t extract(const string &/*op*/, const string &arg, Stack &stack, const Modifiers &) {
    const auto &parser = getParser<N>();
    std::regex pattern(arg);
    size_t origSize = stack.size();
    DomainType dtype;
    for (const string &name : parser.scalarFieldNames()) {
        if (regex_match(name, pattern))
            pushScalarField(stack, name, parser.scalarField(name, DomainType::ANY, dtype), dtype);
    }
    for (const string &name : parser.vectorFieldNames()) {
        if (regex_match(name, pattern))
            pushVectorField(stack, name, parser.vectorField(name, DomainType::ANY, dtype), dtype);
    }
    for (const string &name : parser.symmetricMatrixFieldNames()) {
        if (regex_match(name, pattern))
            pushSymmetricMatrixField(stack, name, parser.symmetricMatrixField(name, DomainType::ANY, dtype), dtype);
    }
    for (const string &name : parser.scalarInterpolantFieldNames()) {
        if (regex_match(name, pattern))
            pushInterpolantField< FISValue>(stack, name, parser.         scalarInterpolantField(name, DomainType::ANY, dtype), dtype);
    }
    for (const string &name : parser.vectorInterpolantFieldNames()) {
        if (regex_match(name, pattern))
            pushInterpolantField< FIVValue>(stack, name, parser.         vectorInterpolantField(name, DomainType::ANY, dtype), dtype);
    }
    for (const string &name : parser.symmetricMatrixInterpolantFieldNames()) {
        if (regex_match(name, pattern))
            pushInterpolantField<FISMValue>(stack, name, parser.symmetricMatrixInterpolantField(name, DomainType::ANY, dtype), dtype);
    }

    assert(stack.size() >= origSize);
    if (stack.size() == origSize) throw runtime_error("No fields matched '" + arg + "'");
    return stack.size() - origSize;
}

template<size_t N>
size_t extractAll(const string &/*op*/, const string &/* arg */, Stack &stack, const Modifiers &) {
    size_t origSize = stack.size();
    const auto &parser = getParser<N>();
    DomainType dtype;
    for (const string &name : parser.scalarFieldNames())
        pushScalarField(stack, name, parser.scalarField(name, DomainType::ANY, dtype), dtype);
    for (const string &name : parser.vectorFieldNames())
        pushVectorField(stack, name, parser.vectorField(name, DomainType::ANY, dtype), dtype);
    for (const string &name : parser.symmetricMatrixFieldNames())
        pushSymmetricMatrixField(stack, name, parser.symmetricMatrixField(name, DomainType::ANY, dtype), dtype);

    assert(stack.size() >= origSize);
    return stack.size() - origSize;
}

template<size_t N>
size_t generate(const string &, const string &arg, Stack &stack, const Modifiers &) {
    const auto &parser = getParser<N>();
    const auto &vertices = parser.vertices();
    const auto &elements = parser.elements();
    if (arg == "x") {
        VectorField<Real, N> x(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i)
            x(i) = truncateFrom3D<VectorND<N>>(vertices[i].point);
        pushVectorField(stack, "x", x, DomainType::PER_NODE);
    }
    else if (arg == "volume") {
        const auto &sampler = getElementSampler<N>();
        ScalarField<Real> vol(elements.size());
        for (size_t i = 0; i < elements.size(); ++i)
            vol[i] = sampler.volume(i);
        pushScalarField(stack, "volume", vol, DomainType::PER_ELEMENT);
    }
    else if (arg == "barycenter") {
        VectorField<Real, N> c(elements.size());
        c.clear();
        for (size_t ei = 0; ei < elements.size(); ++ei) {
            const auto &e = elements[ei];
            for (size_t j : e)
                c(ei) += truncateFrom3D<VectorND<N>>(vertices.at(j).point);
            c(ei) *= 1.0 / e.size();
        }
        pushVectorField(stack, "barycenter", c, DomainType::PER_ELEMENT);
    }
    else throw std::runtime_error("Invalid mesh property name: " + arg);

    return 1;
}

template<size_t N>
size_t expression(const string &, const string &arg, Stack &stack, const Modifiers &) {
    const auto &parser = getParser<N>();
    const auto &vertices = parser.vertices();

    vector<string> components;
    boost::split(components, arg, boost::is_any_of(","));
    string name = "expr(" + arg + ")";

    ExpressionEnvironment env;
    BBox<VectorND<N>> bb(vertices);
    env.setVectorValue("mesh_size_", bb.dimensions());
    env.setVectorValue("mesh_min_",  bb.minCorner);
    env.setVectorValue("mesh_max_",  bb.maxCorner);

    if (components.size() == 1) {
        Expression expr(components[0]);
        ScalarField<Real> sf(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            env.setXYZ(vertices[i].point);
            sf[i] = expr.eval(env);
        }
        pushScalarField(stack, name, sf, DomainType::PER_NODE);
    }
    else if (components.size() == N) {
        ExpressionVector expr(components);
        VectorField<Real, N> vf(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            env.setXYZ(vertices[i].point);
            vf(i) = expr.eval<N>(env);
        }
        pushVectorField(stack, name, vf, DomainType::PER_NODE);
    }
    else throw std::runtime_error("Invalid number of components in vector-valued expression " + arg);

    return 1;
}

size_t     dup(const string &, const string &   , Stack &stack, const Modifiers &) { stack.emplace_back(getValue(stack)); return 2; } // Copies: NamedValue has value semantics
size_t     pop(const string &, const string &   , Stack &stack, const Modifiers &) { popValue(stack); return 0; }
size_t    push(const string &, const string &arg, Stack &stack, const Modifiers &) { double d = parseRealArg(arg); stack.push_back(TypedNamedValue<SValue>(to_string(d), d)); return 1; }
size_t reverse(const string &, const string &   , Stack &stack, const Modifiers &) { std::reverse(stack.begin(), stack.end()); return stack.size(); }
size_t pull(const string &, const string &arg, Stack &stack, const Modifiers &) {
    for (auto it = stack.begin(); it != stack.end(); ++it) {
        if ((*it).name == arg) {
            NamedValue val(std::move(*it));
            stack.erase(it);
            stack.emplace_back(std::move(val));
            return 1;
        }
    }
    throw runtime_error("Couldn't find '" + arg + "' for pull.");
}

////////////////////////////////////////////////////////////////////////////////
// Symmetric matrix operations
////////////////////////////////////////////////////////////////////////////////
struct Eigenvalues {
    // Returns a vector of eigenvalues.
    using value_type = VValue;
    static constexpr const char *name = "eigenvalues";
    static VValue apply(const SMValue &sm) { return VValue(sm.value.eigenvalues()); }
};

struct VonMises {
    // Returns a symmetric matrix
    using value_type = SMValue;
    static constexpr const char *name = "vonMises";
    static SMValue apply(const SMValue &sm) { return vonMises(sm.value); }
};

struct FrobeniusNorm {
    // Returns a scalar
    using value_type = SValue;
    static constexpr const char *name = "frobeniusNorm";
    static SValue apply(const SMValue &sm) { return sqrt(sm.value.frobeniusNormSq()); }
};

template<class Op>
size_t SMatrixOperation(const string &, const string &/* arg */, Stack &stack, const Modifiers &) {
    using VT = typename Op::value_type;
    auto val = popValue(stack);
    string name = std::string(Op::name) + "(" + val.name + ")";
    if (auto sm = dynamic_cast<SMValue *>(VPtr(val)))
        stack.push_back(TypedNamedValue<VT>(name, Op::apply(sm->value)));
    else if (auto ism = dynamic_cast<ISMValue *>(VPtr(val))) {
        auto result = make_unique<InterpolantValue<VT>>(ism->simplexDimension());
        for (size_t i = 0; i < result->dim(); ++i)
            (*result)[i] = VT(Op::apply((*ism)[i].value));
        stack.emplace_back(name, std::move(result));
    }
    else if (auto fsm = dynamic_cast<FSMValue *>(VPtr(val))) {
        auto result = make_unique<FieldValue<VT>>(fsm->size());
        result->domainType = fsm->domainType;
        for (size_t i = 0; i < result->size(); ++i) {
            (*result)[i] = VT(Op::apply((*fsm)[i].value));
        }
        stack.emplace_back(name, std::move(result));
    }
    else if (/*auto ifsm =*/ dynamic_cast<FISMValue *>(VPtr(val))) {
        throw std::runtime_error("Not yet implemented.");
    }
    else throw runtime_error("called on non-matrix type argument");

    return 1;
}

// // This matrix->vector operator unfortunately must be implemented manually in
// // the current framework...
// size_t eigenvaluesAndEigenvectors(const string &, const string &arg, Stack &stack, const Modifiers &) {
//     auto val = popValue(stack);
//     string name = "eigenvalues(" + val.name + ")";
//     if (auto sm = dynamic_cast<SMValue *>(VPtr(val)))
//         stack.push_back(TypedNamedValue<VValue>(name, sm->value.eigenvalues()));
//     else if (auto  ism = dynamic_cast<ISMValue *>(VPtr(val))) {
//         auto result = make_unique<IVValue>(ism->simplexDimension());
//         for (size_t i = 0; i < result->dim(); ++i) {
//             auto &sm = (*ism)[i];
//             (*result)[i] = VValue(sm.value.eigenvalues());
//         }
//         stack.emplace_back(name, std::move(result));
//     }
//     else if (auto  fsm = dynamic_cast<FSMValue *>(VPtr(val))) {
//         auto result = make_unique<FVValue>(fsm->size());
//         result->domainType = fsm->domainType;
//         for (size_t i = 0; i < result->size(); ++i) {
//             auto &sm = (*fsm)[i];
//             (*result)[i] = VValue(sm.value.eigenvalues());
//         }
//         stack.emplace_back(name, std::move(result));
//     }
//     else if (/*auto ifsm =*/ dynamic_cast<FISMValue *>(VPtr(val))) {
//         throw std::runtime_error("Not yet implemented.");
//     }
//     else throw runtime_error("called on non-matrix type argument");
//
//     return 1;
// }

template<size_t N>
size_t elementBarycenterFieldTransfer(const string &, const string &arg, Stack &stack, const Modifiers &) {
    auto &sampler = getElementSampler<N>();
    sampler.accelerate();

    MSHFieldParser<N> targetMesh(arg);

    PointND<N> center;
    std::vector<ElementSampler::Sample> samplePts; samplePts.reserve(targetMesh.elements().size());
    for (const auto &e : targetMesh.elements()) {
        center.setZero();
        for (size_t vi : e) center += truncateFrom3D<VectorND<N>>(targetMesh.vertices().at(vi));
        center *= 1.0 / e.size();
        samplePts.emplace_back(sampler(center));
    }

    auto &currentMesh = getMutableParser<N>();
    Stack xferStack;

    for (const auto &val : stack) {
        string name = "transfer(" + val.name + ")";
        xferStack.emplace_back(name, val->sample(samplePts,
                    currentMesh.meshDegree(), currentMesh.meshDimension(),
                    DomainType::PER_ELEMENT));
    }

    // Replace mesh/value stack with the new mesh, stack of transferred values.
    // Sampler is also invalidated
    stack = std::move(xferStack);
    currentMesh = std::move(targetMesh);
    g_sampler2D.reset();
    g_sampler3D.reset();

    return stack.size();
}

template<size_t N>
size_t loadNewMSH(const string &, const string &arg, Stack &, const Modifiers &) {
    auto &currentMesh = getMutableParser<N>();
    currentMesh = MSHFieldParser<N>(arg);
    g_sampler2D.reset();
    g_sampler3D.reset();
    return 0;
}

// Filter elements using an indicator scalar field.
// (elements with value > 0 are kept).
template<size_t N>
size_t filterElements(const string &, const string &, Stack &stack, const Modifiers &) {
    auto invalid = std::runtime_error("Invalid argument to filterElements. Argument must be a per-element scalar field.");
    try {
        auto &currentMesh = getMutableParser<N>();
        auto top = popTypedValue<FSValue>(stack);
        if (top->domainType != DomainType::PER_ELEMENT)
            throw invalid;
        auto verts = currentMesh.vertices();
        const auto &oldElems  = currentMesh.elements();
        std::vector<::MeshIO::IOElement> elems;
        for (size_t i = 0; i < oldElems.size(); ++i) {
            if (top->value[i].value > 0) elems.emplace_back(oldElems[i]);
        }
        remove_dangling_vertices(verts, elems);
        currentMesh.replaceMesh(elems, verts);
        stack.clear();
    }
    catch (...) {
        throw invalid;
    }
    return 0;
}

// Sample a field at the point(s) specified by vector list encoded in "arg".
// Nodal fields are interpolated using the mesh's finite element basis functions.
// Element fields are interpolated piecewise constant
// Interpolant fields are sampled at barycentric coordinates.
// Error is thrown for sample points outside the mesh (note: could happen on
// element boundaries if inside/outside check is not robust)
template<size_t N>
size_t sample(const string &, const string &arg, Stack &stack, const Modifiers &) {
    auto pts = parseVectorListArg<N>(arg);
    auto val = popValue(stack);
    // Determine element index, barycentric coordinates, and element node
    // indices of the containing element.
    const auto &sampler = getElementSampler<N>();
    const auto &parser = getParser<N>();

    for (const auto &p : pts) {
        stringstream ss;
        ss << p.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, "", ", ", "", "", "[", "]"));
        string name = "sample(" + val.name + ", " + ss.str() + ")";
        stack.emplace_back(name, val->sample(sampler(p), parser.meshDegree(),
                           parser.meshDimension()));
    }

    return pts.size();
}

// Average a field over each element.
template<size_t N>
size_t elementAverage(const string &, const string &/* arg */, Stack &stack, const Modifiers &) {
    auto val = popValue(stack);
    const auto &parser = getParser<N>();
    stack.emplace_back("elementAverage(" + val.name + ")",
                        val->elementAverage(parser.elements(), parser.meshDegree(), parser.meshDimension()));
    return 1;
}

// Create a per-element field by a volume-weighted averaging over each element's
// neighbors.
template<size_t N>
size_t smoothedElementField(const string &, const string &/* arg */, Stack &stack, const Modifiers &) {
    auto val = popValue(stack);
    const auto &parser = getParser<N>();

    // Compute element volumes
    const auto &sampler = getElementSampler<N>();
    size_t nelems = parser.elements().size();
    std::vector<Real> volumes(nelems);
    for (size_t i = 0; i < nelems; ++i)
        volumes[i] = sampler.volume(i);

    stack.emplace_back("smoothedElementField(" + val.name + ")",
                        val->smoothedElementField(parser.elements(), parser.meshDegree(), parser.meshDimension(),
                                                  volumes, MeshConnectivityImpl<SimplicialMesh<N>>(getMeshDS<N>())));
    return 1;
}

// Report filters
// List all fields parsed
template<size_t N>
size_t listNames(const string &, const string &/* arg */, Stack &, const Modifiers &) {
    const auto &parser = getParser<N>();
    for (const string &name : parser.         scalarFieldNames()) { cout << "s\t"  << name << endl; }
    for (const string &name : parser.         vectorFieldNames()) { cout << "v\t"  << name << endl; }
    for (const string &name : parser.symmetricMatrixFieldNames()) { cout << "sm\t" << name << endl; }

    for (const string &name : parser.         scalarInterpolantFieldNames()) { cout << "si\t"  << name << endl; }
    for (const string &name : parser.         vectorInterpolantFieldNames()) { cout << "vi\t"  << name << endl; }
    for (const string &name : parser.symmetricMatrixInterpolantFieldNames()) { cout << "smi\t" << name << endl; }
    return 0;
}

// Print the top of the stack.
size_t print    (const string &, const string &, Stack &stack, const Modifiers &) { getValue(stack)->print(); cout << endl; return 1; }
size_t printName(const string &, const string &, Stack &stack, const Modifiers &) { cout << getValue(stack).name << endl; return 1; }
size_t noprint  (const string &, const string &, Stack &     , const Modifiers &) { return 1; }

template<size_t N>
size_t importScalarField(const string &/* op */, const string &arg, Stack &stack, const Modifiers &) {
    const auto &parser = getParser<N>();
    const auto &vertices = parser.vertices();
    const auto &elements = parser.elements();
    ifstream inFile(arg);
    if (!inFile.is_open()) throw std::runtime_error("Couldn't open scalar field import file: " + arg);
    Real val;
    std::vector<Real> values;
    while (inFile >> val)
        values.push_back(val);
    if (values.size() == vertices.size())
        pushScalarField(stack, arg, values, DomainType::PER_NODE);
    else if (values.size() == elements.size())
        pushScalarField(stack, arg, values, DomainType::PER_ELEMENT);
    else throw std::runtime_error("Didn't recognize imported field size.");

    return 1;
}

// Import a flattened vector field (x0 y0 z0 x1 ...)
template<size_t N>
size_t importVectorField(const string &/* op */, const string &arg, Stack &stack, const Modifiers &) {
    const auto &parser = getParser<N>();
    const auto &vertices = parser.vertices();
    const auto &elements = parser.elements();
    ifstream inFile(arg);
    if (!inFile.is_open()) throw std::runtime_error("Couldn't open vector field import file: " + arg);
    Real val;
    std::vector<Real> values;
    while (inFile >> val)
        values.push_back(val);
    VectorField<Real, N> vfield(values);

    if (vfield.domainSize() == vertices.size())
        pushVectorField(stack, arg, vfield, DomainType::PER_NODE);
    else if (vfield.domainSize() == elements.size())
        pushVectorField(stack, arg, vfield, DomainType::PER_ELEMENT);
    else throw std::runtime_error("Didn't recognize imported field size.");

    return 1;
}

template<size_t N>
size_t outputMSH(const string &/* op */, const string &arg, Stack &stack, const Modifiers &) {
    const auto &parser = getParser<N>();
    // Note: there will be downsampling of interpolant fields if the output
    // mesh elements are linear. But in these cases, the original extracted
    // interpolant fields were linear as well.
    // TODO: rewrite as template code.
    MSHFieldWriter writer(arg, parser.vertices(), parser.elements(), parser.meshType());
    for (const auto &val : stack) {
        if (auto fs = dynamic_cast<const FSValue *>(CVPtr(val))) {
            ScalarField<Real> sf(fs->value.size());
            for (size_t i = 0; i < sf.domainSize(); ++i)
                sf[i] = fs->value[i].value;
            writer.addField(val.name, sf, fs->domainType);
        }
        else if (auto fv = dynamic_cast<const FVValue *>(CVPtr(val))) {
            VectorField<Real, N> vf(fv->value.size());
            for (size_t i = 0; i < vf.domainSize(); ++i)
                vf(i) = fv->value[i].value;
            writer.addField(val.name, vf, fv->domainType);
        }
        else if (auto fsm = dynamic_cast<const FSMValue *>(CVPtr(val))) {
            SymmetricMatrixField<Real, N> smf(fsm->value.size());
            for (size_t i = 0; i < smf.domainSize(); ++i)
                smf(i) = fsm->value[i].value;
            writer.addField(val.name, smf, fsm->domainType);
        }
        else if (auto fis = dynamic_cast<const FISValue *>(CVPtr(val))) {
            std::vector<typename InterpolantGetter<SValue, N>::storage_backed_type> isf(fis->value.size());
            for (size_t i = 0; i < fis->value.size(); ++i)
                isf[i] = InterpolantGetter<SValue, N>::get(fis->value[i]);
            writer.addField(val.name, isf, fis->domainType);
        }
        else if (auto fiv = dynamic_cast<const FIVValue *>(CVPtr(val))) {
            std::vector<typename InterpolantGetter<VValue, N>::storage_backed_type> ivf(fiv->value.size());
            for (size_t i = 0; i < fiv->value.size(); ++i)
                ivf[i] = InterpolantGetter<VValue, N>::get(fiv->value[i]);
            writer.addField(val.name, ivf, fiv->domainType);
        }
        else if (auto fism = dynamic_cast<const FISMValue *>(CVPtr(val))) {
            std::vector<typename InterpolantGetter<SMValue, N>::storage_backed_type> ismf(fism->value.size());
            for (size_t i = 0; i < fism->value.size(); ++i)
                ismf[i] = InterpolantGetter<SMValue, N>::get(fism->value[i]);
            writer.addField(val.name, ismf, fism->domainType);
        }
        else cout << "WARNING: ignored non-field value on stack: " << val.name << endl;
    }
    return stack.size();
}

size_t rename(const string &/* op */, const string &arg, Stack &stack, const Modifiers &) {
    vector<string> names;
    boost::split(names, arg, boost::is_any_of(","));
    if (names.size() > stack.size()) {
        throw runtime_error("Too many names provided to rename");
    }
    size_t pos = stack.size();
    for(auto &name : names)
        stack[--pos].name = std::move(name);
    return names.size();
}

template<size_t N>
size_t setNodePositions(const string &, const string &/* arg */, Stack &stack, const Modifiers &) {
    const auto &top = popTypedValue<FVValue>(stack);

    size_t nVals = top->size();
    std::vector<PointND<3>> newPositions(nVals);
    for (size_t i = 0; i < nVals; ++i)
        newPositions[i] = top->operator[](i).value;

    getMutableParser<N>().setNodePositions(newPositions);
    return 0;
}

template<class R>
size_t applyReduction(const string &op, const string &arg, Stack &stack, const Modifiers &m) {
    auto top = popValue(stack);
    string name = op + arg + "(" + top.name + ")";
    if (m.outerReduction) name = "outer_" + name;
    R r(arg);
    if (m.outerReduction) stack.emplace_back(name, top->outerReduction(r));
    else                  stack.emplace_back(name, top->innerReduction(r));
    return 1;
}

template<class UOp>
size_t applyUnaryOp(const string &op, const string &arg, Stack &stack, const Modifiers &) {
    auto top = popValue(stack);
    stack.emplace_back(op + arg + "(" + top.name + ")",
                       top->componentwiseUnaryOp(UOp(arg)));
    return 1;
}

template<class BOp>
size_t applyBinaryOp(const string &op, const string &arg, Stack &stack, const Modifiers &) {
    // Top of stack is the second operand, next in stack is the first
    auto b = popValue(stack);
    auto a = popValue(stack);
    if (arg.size() != 0) throw runtime_error("Did not expect binary op argument");
    stack.emplace_back(op + "(" + a.name + ", " + b.name + ")",
                       a->componentwiseBinaryOp(BOp(), b));
    return 1;
}

} // end namespace Filter

template<size_t N>
void execute(vector<FilterInvocation> &filters) {
    map<string, function<size_t(const string &, const string &, Stack &, const Modifiers &)>>
    filterImplementations = {
        // Reductions
        {"min",    Filter::applyReduction<ReductionMin   >},
        {"max",    Filter::applyReduction<ReductionMax   >},
        {"minMag", Filter::applyReduction<ReductionMinMag>},
        {"maxMag", Filter::applyReduction<ReductionMaxMag>},
        {"norm",   Filter::applyReduction<ReductionNorm  >},
        {"sum",    Filter::applyReduction<ReductionSum   >},
        {"mean",   Filter::applyReduction<ReductionMean  >},
        {"index",  Filter::applyReduction<ReductionIndex >},
        // Unary operations
        {"abs",    Filter::applyUnaryOp<AbsOp  >},
        {"scale",  Filter::applyUnaryOp<ScaleOp>},
        {"set",    Filter::applyUnaryOp<SetOp  >},
        // Binary operations
        {"add",    Filter::applyBinaryOp<AddOp>},
        {"sub",    Filter::applyBinaryOp<SubOp>},
        {"mul",    Filter::applyBinaryOp<MulOp>},
        {"div",    Filter::applyBinaryOp<DivOp>},
        // Custom value operations
        {"print",          Filter::print},
        {"noprint",        Filter::noprint},
        {"printName",      Filter::printName},
        {"eigenvalues",    Filter::SMatrixOperation<Filter::Eigenvalues>},
        {"vonMises",       Filter::SMatrixOperation<Filter::VonMises>},
        {"frobeniusNorm",  Filter::SMatrixOperation<Filter::FrobeniusNorm>},
        // {"eigs",           Filter::eigenvaluesAndEigenvectors},
        {"sample",         Filter::sample<N>},
        {"elementAverage", Filter::elementAverage<N>},
        {"smoothedElementField", Filter::smoothedElementField<N>},
        // Stack operations
        {"list",          Filter::listNames<N>},
        {"extract",       Filter::extract<N>},
        {"extractAll",    Filter::extractAll<N>},
        {"generate",      Filter::generate<N>},
        {"expression",    Filter::expression<N>},
        {"dup",           Filter::dup},
        {"pop",           Filter::pop},
        {"push",          Filter::push},
        {"pull",          Filter::pull},
        {"rename",        Filter::rename},
        {"import_sfield", Filter::importScalarField<N>},
        {"import_vfield", Filter::importVectorField<N>},
        {"reverse",       Filter::reverse},
        {"setNodePositions", Filter::setNodePositions<N>},
        {"outMSH",        Filter::outputMSH<N>},

        {"transferFieldsToPerElem", Filter::elementBarycenterFieldTransfer<N>},
        {"loadNewMSH", Filter::loadNewMSH<N>},
        {"filterElements", Filter::filterElements<N>}
    };

    // Classify the operations.
    set<string> reductions = { "min", "max", "minMag", "maxMag",
                               "norm", "sum", "mean", "index" };
    set<string> unaryOps  = { "abs", "scale", "set" };
    set<string> binaryOps = { "add", "sub", "mul", "div" };

    // The following commands suppress automatic output of stack at exit when performed last
    set<string> suppressImplicitPrint = { "noprint", "print", "outMSH", "list" };

    // Apply-all makes sense only for the following operations:
    set<string> acceptsApplyAll = { "print", "printName", "eigenvalues",
                                    "vonMises", "frobeniusNorm", "sample" };
    acceptsApplyAll.insert(reductions.begin(), reductions.end());
    acceptsApplyAll.insert(  unaryOps.begin(),   unaryOps.end());
    acceptsApplyAll.insert( binaryOps.begin(),  binaryOps.end());

    // Implicit list operation when filters are empty
    if (filters.size() == 0) filters.push_back({"list", ""});

    // Add an implicit print operation unless it is supressed
    if (!suppressImplicitPrint.count(filters.back().first)) filters.push_back({"print", ""});

    Stack stack;
    for (size_t fi = 0; fi < filters.size(); ++fi) {
        try {
            Modifiers m; // fresh modifier flags
            runtime_error missingOperation("Modifier specified without an operation.");
            // Note: applyAll should appear *first*
            if (filters[fi].first == "applyAll")       { m.applyAll       = true; ++fi; }
            if (fi >= filters.size()) throw missingOperation;
            if (filters[fi].first == "outerReduction") { m.outerReduction = true; ++fi; }
            if (fi >= filters.size()) throw missingOperation;
            const auto &f = filters[fi];

            // Validate modifiers.
            if (m.outerReduction && (reductions.count(f.first) == 0))
                throw runtime_error("--outerReduction must be followed by reduction");
            if (m.applyAll && (acceptsApplyAll.count(f.first) == 0))
                throw runtime_error("operation does not support apply all");

            // Perform the filter either once or once per stack value.
            if (m.applyAll) {
                Stack newStack;
                while (stack.size()) {
                    size_t n = filterImplementations.at(f.first)(f.first, f.second, stack, m);
                    // Move each result over to the new stack
                    for (size_t r = 0; r < n; ++r)
                        newStack.emplace_back(popValue(stack));
                }
                std::reverse(newStack.begin(), newStack.end());
                stack = std::move(newStack);
            }
            else filterImplementations.at(f.first)(f.first, f.second, stack, m);
        }
        catch (const exception &e) {
            if (fi < filters.size())
                cout << "Filter '" << filters[fi].first << "' failed: " << e.what() << endl;
            else
                cout << "Filter failed: " << e.what() << endl;
            exit(-1);
        }
    }
}

int main(int argc, char *argv[])
{
    cout << std::scientific << std::setprecision(16);
    MeshIO::MeshIO_MSH io;
    vector<MeshIO::IOVertex>  v;
    vector<MeshIO::IOElement> e;

    string mshFile;
    vector<FilterInvocation> filters;
    auto forcedDim = boost::make_optional(false, size_t()); // work around maybe-uninitialized GCC warning bug
    tie(mshFile, filters, forcedDim) = parseCmdLine(argc, argv);

    ifstream infile(mshFile);
    if (!infile.is_open()) throw runtime_error("Couldn't open " + mshFile);
	MeshIO::MeshType type = io.load(infile, v, e, MeshIO::MESH_GUESS);
    size_t meshDim = ::MeshIO::meshDimension(type);

    size_t dim = forcedDim ? *forcedDim : meshDim;
    if (dim < 2 || dim > 3) throw std::runtime_error("Unsupported dimension: " + std::to_string(dim));

    if (meshDim != dim)
        cerr << "Warning: some operations won't work properly on non-full-dimension meshes" << endl;

    if (dim == 3) parseMSH<3>(infile, type, std::move(e), std::move(v), io.binary(), meshDim != dim);
    else          parseMSH<2>(infile, type, std::move(e), std::move(v), io.binary(), meshDim != dim);

    if (dim == 3) execute<3>(filters);
    else          execute<2>(filters);

    return 0;
}
