////////////////////////////////////////////////////////////////////////////////
// Values.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      The dynamic value system for msh_processor, which consists of the
//      class hierarchy:
//  Value
//     ValueBase
//         PointValueTag
//             SValue
//             VValue
//             SMValue
//         CollectionValueTag
//             InterpolantValue<PointValue> (InterpolantValueTag)
//             FieldValue<ValueType>        (FieldValueTag)
//  Values are generally classified as "point values" (representing a
//  quantity that is defined at a single point) and "collection value"
//  (representing a value for each point).
//
//  Point values have the following useful typedefs:
//      raw_type                the type of data stored at the point
//      static_sized_raw_type   statically sized data type from which the point
//                              value can be constructed
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/29/2015 15:42:26
////////////////////////////////////////////////////////////////////////////////
#ifndef VALUES_HH
#define VALUES_HH

#include <memory>
#include <vector>
#include <string>
#include <array>
#include <stdexcept>
#include <cassert>
#include <type_traits>

#include <Eigen/Dense>
#include <MeshFEM/SymmetricMatrix.hh>
#include <MeshFEM/Simplex.hh>
#include <MeshFEM/Functions.hh>
#include <MeshFEM/Fields.hh>

#include "Sampler.hh"
#include "MeshConnectivity.hh"

class Value;
using UVPtr  = std::unique_ptr<      Value>;
using CUVPtr = std::unique_ptr<const Value>;
using VPtr  =       Value *;
using CVPtr = const Value *;

using ESample = ElementSampler::Sample;

////////////////////////////////////////////////////////////////////////////////
// Basic declarations for handling Binary/Unary Ops, Reductions, and printing
// (needed for Value classes.)
// The rest is in the respective .inl files.
////////////////////////////////////////////////////////////////////////////////
// Binary ops: applied componentwise to a pair of value
struct BinaryOp  {
    virtual ~BinaryOp() = default;
    virtual Real operator()(Real a, Real b) const = 0;
};
template<class T1> UVPtr dispatchCWiseBinaryOp(const BinaryOp &op, const T1 &a, CVPtr b);
// Unary ops: applied to each component of a single value
struct UnaryOp {
    UnaryOp(const std::string &arg = "") { setArg(arg); }
    virtual ~UnaryOp() = default;
    virtual Real operator()(Real a) const = 0;
    virtual void setArg(const std::string &arg) { if (arg.size()) throw std::runtime_error("Did not expect unary op argument"); }
};
template<class T> std::unique_ptr<T> applyUnaryOp(const UnaryOp &op, const T &a);
// Reductions: functors called on a sequence of scalar value (once per value).
struct Reduction {
    Reduction(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void setArg(const std::string &arg) { if (arg.size()) throw std::runtime_error("Did not expect reduction argument"); }
    virtual void reset() { m_acc = 0; }
    virtual void operator()(Real val) = 0;
    // Except in special cases, the accumulator usually holds the result.
    virtual Real result() const { return m_acc; }
    virtual ~Reduction() = default;
protected:
    Real m_acc;
};
template<class T, bool nested>     struct PrintImpl;
template<class T, typename = void> struct SampleImpl;
template<class T>                  struct MultipointSampleImpl;
template<class T, typename = void> struct ElmAvgImpl;
template<class T>                  struct SmoothedElmFldImpl;

////////////////////////////////////////////////////////////////////////////////
// Value type interface.
////////////////////////////////////////////////////////////////////////////////
class Value {
public:
    virtual void print(std::ostream &os = std::cout)                       const = 0;
    virtual UVPtr outerReduction(Reduction &r)                             const = 0;
    virtual UVPtr innerReduction(Reduction &r)                             const = 0;
    virtual UVPtr componentwiseBinaryOp(const BinaryOp &op, CVPtr b)       const = 0;
    virtual UVPtr componentwiseUnaryOp (const  UnaryOp &op)                const = 0;
    virtual UVPtr sample(const ESample &s, size_t meshDeg, size_t meshDim) const = 0;
    virtual UVPtr clone()                                                  const = 0;

    // Sample at a list of points, creating a per-sample-point field value.
    virtual UVPtr sample(const std::vector<ESample> &ss, size_t meshDeg, size_t meshDim, DomainType dt) const = 0;

    virtual UVPtr elementAverage(const std::vector<MeshIO::IOElement> &elems, size_t meshDeg, size_t meshDim) const = 0;
    virtual UVPtr smoothedElementField(const std::vector<MeshIO::IOElement> &elems, size_t meshDeg, size_t meshDim,
                                       const std::vector<Real> &volumes, const MeshConnectivity &connectivity) const = 0;

    virtual ~Value() = default;
};

// Value semantics, but without a copy assignment operator to make sure we
// avoid inadvertent expensive copies (move assignment is provided).
struct NamedValue {
    // Copy constructor clones the underlying value
    NamedValue(const NamedValue &v) : name(v.name), m_valptr(v->clone()) { }
    // Moving values moves the underlying pointer.
    NamedValue(NamedValue &&v) noexcept : name(std::move(v.name)), m_valptr(std::move(v.m_valptr)) { }
    // Creation from value pointer.
    NamedValue(const std::string &n, UVPtr &&ptr) : name(n), m_valptr(std::move(ptr)) { }

    CVPtr operator->() const { return m_valptr.get(); }
     VPtr operator->()       { return m_valptr.get(); }

    NamedValue &operator=(const NamedValue &b) = delete;
    NamedValue &operator=(NamedValue &&b) { m_valptr = std::move(b.m_valptr); name = std::move(b.name); return *this; }

    // Implicit casts to pointer type
    operator CVPtr() const { return m_valptr.get(); }
    operator  VPtr()       { return m_valptr.get(); }

    ~NamedValue() { }
    std::string name;
protected:
    UVPtr m_valptr;
};

// T must be a subclass of Value!
template<class T>
struct TypedNamedValue : public NamedValue {
    // Throw exceptions for invalid types.
    TypedNamedValue(NamedValue &&v) : NamedValue(std::move(v)) {
        if (dynamic_cast<const T *>(m_valptr.get()) == nullptr)
            throw std::runtime_error("Invalid argument");
    }
    // Allow typed construction
    template<typename... Args>
    TypedNamedValue(const std::string &n, Args&&... args) : NamedValue(n, std::make_unique<T>(std::forward<Args>(args)...)) { }

    TypedNamedValue &operator=(NamedValue &&b) {
        if (dynamic_cast<const T *>(CVPtr(b)) == nullptr)
            throw std::runtime_error("Invalid argument");
        NamedValue::operator=(b);
        return this;
    }

    // Work around the "potential null pointer dereference" warning.
    // Technically this dereference *could* be null if a TypedNamedValue
    // instance is assigned a value of a different type by using
    // a reference/pointer to the base class NamedValue.
    const T *guardedAccess() const {
        const T *result = dynamic_cast<const T *>(m_valptr.get());
        if (result) return result;
        throw std::runtime_error("Null pointer.");
    }
    T *guardedAccess() {
        T *result = dynamic_cast<T *>(m_valptr.get());
        if (result) return result;
        throw std::runtime_error("Null pointer.");
    }

    const T *operator->() const { return guardedAccess(); }
          T *operator->()       { return guardedAccess(); }
};

////////////////////////////////////////////////////////////////////////////////
// Forward declarations of all Value types.
////////////////////////////////////////////////////////////////////////////////
// "Point" Values
struct SValue;
struct VValue;
struct SMValue;

template<class ValueType>
struct FieldValue;
template<class PointValueType>
struct InterpolantValue;

// Value interpolants (triangles)
using  ISValue = InterpolantValue< SValue>;
using  IVValue = InterpolantValue< VValue>;
using ISMValue = InterpolantValue<SMValue>;

// Value fields
using  FSValue = FieldValue< SValue>;
using  FVValue = FieldValue< VValue>;
using FSMValue = FieldValue<SMValue>;

// Interpolant fields (triangles/tets)
using  FISValue = FieldValue< ISValue>;
using  FIVValue = FieldValue< IVValue>;
using FISMValue = FieldValue<ISMValue>;

// Mechanism for distinguishing general types of values in SFINAE
class PointValueTag      { };
class CollectionValueTag { };
class InterpolantValueTag : public CollectionValueTag { };
class FieldValueTag       : public CollectionValueTag { };

template<typename T> struct is_point_value       : public std::is_base_of<PointValueTag,       T> { };
template<typename T> struct is_collection_value  : public std::is_base_of<CollectionValueTag,  T> { };
template<typename T> struct is_interpolant_value : public std::is_base_of<InterpolantValueTag, T> { };
template<typename T> struct is_field_value       : public std::is_base_of<FieldValueTag,       T> { };

template<class T, typename T2 = void> struct enable_if_point_value           : public std::enable_if<       is_point_value<T>::value, T2> { };
template<class T, typename T2 = void> struct enable_if_collection_value      : public std::enable_if<  is_collection_value<T>::value, T2> { };
template<class T, typename T2 = void> struct enable_if_interpolant_value     : public std::enable_if< is_interpolant_value<T>::value, T2> { };
template<class T, typename T2 = void> struct enable_if_field_value           : public std::enable_if<       is_field_value<T>::value, T2> { };

template<class T, typename T2 = void> struct enable_if_not_point_value       : public std::enable_if<!      is_point_value<T>::value, T2> { };
template<class T, typename T2 = void> struct enable_if_not_collection_value  : public std::enable_if<! is_collection_value<T>::value, T2> { };
template<class T, typename T2 = void> struct enable_if_not_interpolant_value : public std::enable_if<!is_interpolant_value<T>::value, T2> { };
template<class T, typename T2 = void> struct enable_if_not_field_value       : public std::enable_if<!      is_field_value<T>::value, T2> { };

template<class Derived>
class ValueBase : public Value {
public:
    virtual void print(std::ostream &os = std::cout)      const { PrintImpl<Derived, false>::run(os, static_cast<const Derived&>(*this)); }
    virtual UVPtr outerReduction(Reduction &r)            const { return applyOuterReduction(r, static_cast<const Derived&>(*this)); }
    virtual UVPtr innerReduction(Reduction &r)            const { return applyInnerReduction(r, static_cast<const Derived&>(*this)); }
    virtual UVPtr componentwiseUnaryOp(const UnaryOp &op) const { return applyUnaryOp(op, static_cast<const Derived&>(*this)); };
    // Outer stage of double dispatch for component-wise binary operations
    virtual UVPtr componentwiseBinaryOp(const BinaryOp &op, CVPtr b)       const { return dispatchCWiseBinaryOp(op, static_cast<const Derived&>(*this), b); }
    virtual UVPtr clone()                                                  const { return std::make_unique<Derived>(static_cast<const Derived&>(*this)); }
    virtual UVPtr sample(const ESample &s, size_t meshDeg, size_t meshDim) const { return SampleImpl<Derived>::run(static_cast<const Derived&>(*this), s, meshDeg, meshDim); }
    virtual UVPtr elementAverage(const std::vector<MeshIO::IOElement> &elems, size_t meshDeg, size_t meshDim) const { return ElmAvgImpl<Derived>::run(static_cast<const Derived&>(*this), elems, meshDeg, meshDim); }
    virtual UVPtr smoothedElementField(const std::vector<MeshIO::IOElement> &elems, size_t meshDeg, size_t meshDim, const std::vector<Real> &volumes, const MeshConnectivity &conn) const { return SmoothedElmFldImpl<Derived>::run(static_cast<const Derived&>(*this), elems, meshDeg, meshDim, volumes, conn); }

    // Sample at a list of points, creating a per-sample-point field value.
    virtual UVPtr sample(const std::vector<ESample> &ss, size_t meshDeg, size_t meshDim, DomainType dt) const { return MultipointSampleImpl<Derived>::run(static_cast<const Derived&>(*this), ss, meshDeg, meshDim, dt); }
};

struct SValue : public ValueBase<SValue>, public PointValueTag {
    SValue(Real v = 0) : value(v) { }
    using raw_type = Real;
    template<size_t _N>
    using static_sized_raw_type = Real;
    static constexpr size_t dim() { return 1; }
    static constexpr size_t   N() { return 1; }

    Real  operator[](size_t i) const { if (i >= dim()) throw std::runtime_error("Out of bounds access"); return value; }
    Real &operator[](size_t i)       { if (i >= dim()) throw std::runtime_error("Out of bounds access"); return value; }

    // Convenience operations for compound op implementation
    SValue &operator*=(Real   v) { value *= v;       return *this; }
    SValue &operator+=(SValue b) { value += b.value; return *this; }

    raw_type value;
};

// Note, assignment can change the (dynamic) dimension!
// This this simplifies, e.g., the computation of vector field results.
struct VValue : public ValueBase<VValue>, public PointValueTag {
    using raw_type = Eigen::Matrix<Real, Eigen::Dynamic, 1, 0, 3, 1>; // Dynamic size up to 3x1
    template<int _N>
    using static_sized_raw_type = Eigen::Matrix<Real, _N, 1>;

    VValue(size_t N = 3) : value(raw_type::Zero(N)) { }
    VValue(const raw_type &v) : value(v) { }

    // Size-detecting constructor
    template<int _N>
    VValue(const static_sized_raw_type<_N> &v) : value(v) { validateSize(); }

    size_t dim() const { return value.rows(); }
    size_t N()   const { return value.rows(); }

    void validateSize() const { if ((dim() < 2) || (dim() > 3)) throw std::runtime_error("Invalid vector dimension"); }

    Real  operator[](size_t i) const { if (i >= dim()) throw std::runtime_error("Out of bounds access"); return value[i]; }
    Real &operator[](size_t i)       { if (i >= dim()) throw std::runtime_error("Out of bounds access"); return value[i]; }

    // Convenience operations for compound op implementation
    VValue &operator*=(Real          v) { value *= v;       return *this; }
    VValue &operator+=(const VValue &b) { value += b.value; return *this; }

    raw_type value;
};

// We'll need to distinguish between SMValue's copy/move constructors and
// constructors that forward to DynamicSymmetricMatrix
template<typename... Types>
struct is_smval_argument : public std::false_type { };
template<typename T>
struct is_smval_argument<T> { static constexpr bool value = std::is_same<typename std::decay<T>::type, SMValue>::value; };

struct SMValue : public ValueBase<SMValue>, public PointValueTag {
    using raw_type = DynamicSymmetricMatrix<Real>;
    template<size_t _N>
    using static_sized_raw_type = SymmetricMatrixValue<Real, _N>;

    // Allow copy and move construction.
    SMValue(const SMValue  &b) : value(b.value) { }
    SMValue(      SMValue &&b) : value(std::move(b.value)) { }

    // Forward everything else to DynamicSymmetricMatrix's constructors.
    template<typename... Args,
             typename std::enable_if<!is_smval_argument<Args...>::value, int>::type = 0>
    SMValue(Args&&... args) : value(std::forward<Args>(args)...) { }

    SMValue &operator=(const SMValue  &b) { value =           b.value ; return *this; }
    SMValue &operator=(      SMValue &&b) { value = std::move(b.value); return *this; }

    size_t dim() const { return value.flatSize(); }
    size_t N()   const { return value.size(); }

    Real  operator[](size_t i) const { if (i >= dim()) throw std::runtime_error("Out of bounds access"); return value[i]; }
    Real &operator[](size_t i)       { if (i >= dim()) throw std::runtime_error("Out of bounds access"); return value[i]; }

    // Convenience operations for compound op implementation
    SMValue &operator*=(Real           v) { value *= v;       return *this; }
    SMValue &operator+=(const SMValue &b) { value += b.value; return *this; }

    raw_type value;
};

// Interpolants are tricky because we need to treat them as a collection of
// PointValueType objects for recursive implementation of componentwise ops and
// reductions, but the Interpolant class expects a collection of raw values
// (providing the necessary arithmetic operator overloads).
// Also, we want to dynamically choose between two interpolants--triangle or
// tet--without the overhead of separate storage for both.
// Thankfully, the Interpolant class has a configurable storage policy, which
// allows us to solve both problems. We make a policy that references an array
// of PointValueType objects (stored externally in the InterpolantValue class)
// but whose [] accessor extracts the per-node data's raw value
//
// Note, assignment can change the (dynamic) simplex dimension!
// This this simplifies, e.g., the computation of fields of interpolants.

// Converting constructor implementation (specialized below).
template<class IV, class IRaw>
struct InterpolantConvertConstructImpl;

// Templated access to the underlying interpolant.
template<class PVT, size_t _K2> struct InterpolantGetter;
template<class PVT> struct InterpolantGetter<PVT, 2> {
    using type = typename InterpolantValue<PVT>::template interpolant_type<2>;
    using storage_backed_type = typename InterpolantValue<PVT>::template interpolant_sb_type<2>;
    static const type &get(const InterpolantValue<PVT> &iv) { return iv.getTriInterpolant(); }
};
template<class PVT> struct InterpolantGetter<PVT, 3> {
    using storage_backed_type = typename InterpolantValue<PVT>::template interpolant_sb_type<3>;
    using type = typename InterpolantValue<PVT>::template interpolant_type<3>;
    static const type &get(const InterpolantValue<PVT> &iv) { return iv.getTetInterpolant(); }
};

template<class PointValueType>
struct InterpolantValue : public ValueBase<InterpolantValue<PointValueType>>, public InterpolantValueTag {
    // Always use degree 2 interpolants for simplicity (in the future, this
    // would ideally be a template parameter, and multiplication
    // would respect this...).
    static constexpr size_t Deg = 2;

    // Enough storage for 2 or 3D interpolants.
    using raw_type = std::array<PointValueType, Simplex::numNodes(3, Deg)>;
    using raw_point_value_type = typename PointValueType::raw_type;

    // Copy constructor copies simplex dimension and values
    // (but not the interpolants: their pointers must point to our values!)
    InterpolantValue(const InterpolantValue &b) : value(b.value), triInterpolant(&value), tetInterpolant(&value) { setSimplexDimension(b.m_simplexDimension); }
    InterpolantValue(size_t K = 3) : triInterpolant(&value), tetInterpolant(&value) { setSimplexDimension(K); }
    // Note: move constructor of std::array still must loop over every entry,
    // but we gain if the contents are move constructible.
    InterpolantValue(InterpolantValue &&b) : value(std::move(b.value)), triInterpolant(&value), tetInterpolant(&value) { setSimplexDimension(b.m_simplexDimension); }

    // Converting constructor from an interpolant.
    template<typename T, size_t _K, size_t _Deg, template<typename, size_t, size_t> class _NSP>
    InterpolantValue(const Interpolant<T, _K, _Deg, _NSP> &b)
        : triInterpolant(&value), tetInterpolant(&value) {
        InterpolantConvertConstructImpl<InterpolantValue<PointValueType>,
                                        Interpolant<T, _K, _Deg, _NSP>>::run(*this, b);
    }

    size_t dim() const { return Simplex::numNodes(m_simplexDimension, Deg); }
    PointValueType  operator[](size_t i) const { if (i >= dim()) throw std::runtime_error("Index out of bounds"); return value[i]; }
    PointValueType &operator[](size_t i)       { if (i >= dim()) throw std::runtime_error("Index out of bounds"); return value[i]; }

    template<class Derived>
    PointValueType sampleBarycentric(const Eigen::DenseBase<Derived> &lambda) const {
        PointValueType result;
        if      (m_simplexDimension == 2) return PointValueType(triInterpolant(lambda[0], lambda[1], lambda[2]));
        else if (m_simplexDimension == 3) return PointValueType(tetInterpolant(lambda[0], lambda[1], lambda[2], lambda[3]));
        throw std::logic_error("Invalid simplex dimension");
    }

    PointValueType average() const {
        PointValueType result;
        if      (m_simplexDimension == 2) return PointValueType(triInterpolant.average());
        else if (m_simplexDimension == 3) return PointValueType(tetInterpolant.average());
        throw std::logic_error("Invalid simplex dimension");
    }

    // Note: changing the simplex type "invalidates" the stored data.
    // (it would be extra work to pad/restrict in a sensible way).
    void setSimplexDimension(size_t K) { if ((K < 2) || (K > 3)) throw std::runtime_error("Invalid simplex dimension"); m_simplexDimension = K; }
    size_t simplexDimension() const { return m_simplexDimension; }

    // Hack: m_simplexDimension acts like the "size" in the sense that it's
    // what must be passed to the constructor to create a new interpolant of
    // the same size/dim.
    size_t size() const { return m_simplexDimension; }

    // Storage policy: reference an InterpolantValue's data.
    template<typename T, size_t K, size_t Deg>
    struct InterpolantStoragePolicy {
        // We need the default constructor for compatibility with Interpolant,
        // but it must never be called.
        InterpolantStoragePolicy() : m_nodeVal(nullptr) { throw std::logic_error("InterpolantStoragePolicy default constructor unsupported"); }
        InterpolantStoragePolicy(raw_type *store) : m_nodeVal(store) { }
        static constexpr size_t numNodalValues = Simplex::numNodes(K, Deg);

        // Accessors needed for the interpolant class
        constexpr size_t size() const { return numNodalValues; }
        const T &operator[](size_t i) const { assert(i < numNodalValues); return (*m_nodeVal)[i].value; }
              T &operator[](size_t i)       { assert(i < numNodalValues); return (*m_nodeVal)[i].value; }
    private:
        raw_type *m_nodeVal;
    };

    friend void swap(InterpolantValue &a, InterpolantValue &b) throw() {
        std::swap(a.value, b.value);
        std::swap(a.m_simplexDimension, b.m_simplexDimension);
    }

    InterpolantValue &operator=(const InterpolantValue &b) {
        if (this != &b) {
            InterpolantValue copy(b);
            std::swap(*this, copy);
        }
        return *this;
    }

    InterpolantValue &operator=(InterpolantValue &&b) {
        value = std::move(b.value);
        m_simplexDimension = b.m_simplexDimension;
        return *this;
    }

    template<size_t _K>
    using interpolant_type    = Interpolant<raw_point_value_type, _K, Deg, InterpolantStoragePolicy>;
    // storage-backed type
    template<size_t _K>
    using interpolant_sb_type = Interpolant<raw_point_value_type, _K, Deg>;
    const interpolant_type<2> &getTriInterpolant() const { if (m_simplexDimension != 2) throw std::runtime_error("Wrong dimension interpolant getter called"); return triInterpolant; }
    const interpolant_type<3> &getTetInterpolant() const { if (m_simplexDimension != 3) throw std::runtime_error("Wrong dimension interpolant getter called"); return tetInterpolant; }

    raw_type value;
private:
    // Note: probably could have used boost::variant
    size_t m_simplexDimension;
    const interpolant_type<2> triInterpolant;
    const interpolant_type<3> tetInterpolant;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation of constructor converting from Interpolant to InterpolantValue
// (Used from InterpolantValue's copy constructor).
////////////////////////////////////////////////////////////////////////////////
// Deg 2 implementation
template<class IV, typename T, size_t K, template<typename, size_t, size_t> class NSP>
struct InterpolantConvertConstructImpl<IV, Interpolant<T, K, 2, NSP>> {
    static void run(IV &a, const Interpolant<T, K, 2, NSP> &b) {
        a.setSimplexDimension(K);
        assert(a.dim() == b.size());
        for (size_t i = 0; i < b.size(); ++i) {
            using value_type = typename std::remove_cv<typename std::remove_reference<decltype(a.value[i])>::type>::type; // hack around bug in Visual Studio...
            a.value[i] = value_type(b[i]);
        }
    }
};

// Deg 1 implementation
template<class IV, typename T, size_t K, template<typename, size_t, size_t> class NSP>
struct InterpolantConvertConstructImpl<IV, Interpolant<T, K, 1, NSP>> {
    static void run(IV &a, const Interpolant<T, K, 1, NSP> &b) {
        Interpolant<T, K, 2, NSP> bUpscale(b);
        InterpolantConvertConstructImpl<IV, Interpolant<T, K, 2, NSP>>::run(a, bUpscale);
    }
};

template<class ValueType>
struct FieldValue : public ValueBase<FieldValue<ValueType>>, public FieldValueTag {
    using raw_type = std::vector<ValueType>;
    FieldValue(size_t numElems = 0) { resize(numElems); }
    FieldValue(DomainType dt, size_t numElems) : domainType(dt) { resize(numElems); }

    size_t size() const { return value.size(); }
    void resize(size_t s) { value.resize(s); }
    size_t  dim() const { return size(); }
    const ValueType &operator[](size_t i) const { return value.at(i); }
          ValueType &operator[](size_t i)       { return value.at(i); }

    raw_type value;
    DomainType domainType;
};

////////////////////////////////////////////////////////////////////////////////
// One-dimensional Scalar value indexing adaptors
////////////////////////////////////////////////////////////////////////////////
// Point values have a different indexing interface from scalar collections
// (They return floating point types instead of SValue types).
template<class T> typename     enable_if_point_value<T, Real>::type
getScalarValueAtIndex(const T &val, size_t i) { return val[i]; }
template<class T> typename enable_if_not_point_value<T, Real>::type
getScalarValueAtIndex(const T &val, size_t i) { return val[i].value; }

template<class T> typename     enable_if_point_value<T, Real>::type
&getScalarValueAtIndex(T &val, size_t i) { return val[i]; }
template<class T> typename enable_if_not_point_value<T, Real>::type
&getScalarValueAtIndex(T &val, size_t i) { return val[i].value; }

#include "ValueOperations/Printing.inl"
#include "ValueOperations/BinaryOps.inl"
#include "ValueOperations/UnaryOps.inl"
#include "ValueOperations/Reductions.inl"
#include "ValueOperations/NodalInterpolant.inl"
#include "ValueOperations/Sampling.inl"
#include "ValueOperations/ElementAverage.inl"
#include "ValueOperations/Smoothing.inl"

#endif /* end of include guard: VALUES_HH */
