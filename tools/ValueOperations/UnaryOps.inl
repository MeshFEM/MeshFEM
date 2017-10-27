////////////////////////////////////////////////////////////////////////////////
// UnaryOps.inl
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements componetwise unary ops on values. To be included by Value.hh
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/28/2015 16:01:56
////////////////////////////////////////////////////////////////////////////////
#include "../argparse.hh"

struct AbsOp   : public UnaryOp { AbsOp(const std::string &arg) : UnaryOp(arg) { }; virtual Real operator()(Real a) const { return std::abs(a); } };
struct ScaleOp : public UnaryOp { virtual Real operator()(Real a) const { return m_s * a; };
    ScaleOp(const std::string &arg) { setArg(arg); }
    virtual void setArg(const std::string &arg) { m_s = parseRealArg(arg); }
private:
    Real m_s;
};
struct SetOp   : public UnaryOp { virtual Real operator()(Real /* a */) const { return m_s; };
    SetOp(const std::string &arg) { setArg(arg); }
    virtual void setArg(const std::string &arg) { m_s = parseRealArg(arg); }
private:
    Real m_s;
};

// Apply directly to all components of a non-compound value (e.g. point value)
template<class T>
typename std::enable_if<is_point_value<T>::value, void>::type
applyUnaryOpInPlace(const UnaryOp &op, T &a) {
    for (size_t i = 0; i < a.dim(); ++i)
        a[i] = op(a[i]);
}

// Recursively apply unary operation to each component of a compound value.
// (e.g. non point value)
template<class T>
typename std::enable_if<!is_point_value<T>::value, void>::type
applyUnaryOpInPlace(const UnaryOp &op, T &a) {
    for (size_t i = 0; i < a.dim(); ++i)
        applyUnaryOpInPlace(op, a[i]);
}

template<class T>
std::unique_ptr<T> applyUnaryOp(const UnaryOp &op, const T &a) {
    // Make copy, then perform in-place operation
    auto result = std::make_unique<T>(a);
    applyUnaryOpInPlace(op, *result);
    return result;
}
