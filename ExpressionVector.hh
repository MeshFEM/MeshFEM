////////////////////////////////////////////////////////////////////////////////
// ExpressionVector.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides a wrapper for libmatheval that evalutes vector-valued
//      expressions.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/26/2014 20:40:39
////////////////////////////////////////////////////////////////////////////////
#ifndef EXPRESSIONVECTOR_HH
#define EXPRESSIONVECTOR_HH
#include <matheval.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <memory>

// There's some sketchyness that has to happen here (like const_cast and
// mutable) to hack around some of libmatheval's unfortunate interface
// decisions.
struct ExpressionEnvironment {
    void setValue(const std::string &name, double value) {
        for (size_t i = 0; i < names.size(); ++i) {
            if (*names[i] == name) {
                values[i] = value;
                return;
            }
        }
        names.push_back(std::make_shared<std::string>(name));
        values.push_back(value);
        cnames.push_back(const_cast<char *>(names.back()->c_str()));
    }

    // Sets a value for each component with the names name1, name2...
    template<class _Vec>
    void setVectorValue(const std::string &name, const _Vec &v) {
        for (int i = 0; i < _Vec::RowsAtCompileTime; ++i)
            setValue(name + std::to_string(i), v[i]);
    }

    void setVectorValue(const std::string &name, const std::vector<Real> &v) {
        for (int i = 0; i < v.size(); ++i)
            setValue(name + std::to_string(i), v[i]);
    }

    template<class _Vec>
    void setXYZ(_Vec &v) {
        int N = _Vec::RowsAtCompileTime;
        if (!(N == 2 || N == 3)) throw std::runtime_error("Bad vector size");
        setValue("x", v[0]); setValue("y", v[1]);
        setValue("z", (N == 3) ? v[2] : 0);
    }

    char **getNames() const {
        return &cnames[0];
    }

    double *getValues() const {
        return &values[0];
    }

    size_t numVars() const { return names.size(); }
private:
    std::vector<std::shared_ptr<std::string>> names;
    mutable std::vector<double> values;
    mutable std::vector<char *> cnames;
};

// Expression wrapper handling destruction that should be wrapped in a smart
// pointer.
class Expression {
public:
    // Copies string for const correctness...
    // (libmatheval wants non-const pointers)
    Expression(std::string s) {
        m_eval = evaluator_create(const_cast<char *>(s.c_str()));
        if (!m_eval)
            throw std::runtime_error("Failed to parse expression '" + s + "'");
    }

    // DAANNGEROUS... get rid of it
    Expression &operator=(const Expression&e) = delete;

    double eval(const ExpressionEnvironment &e) const {
        if (e.numVars() == 0)
            throw std::runtime_error("Empty environment");
        return evaluator_evaluate(m_eval, e.numVars(), e.getNames(), e.getValues());
    }

    ~Expression() {
        if (m_eval) evaluator_destroy(m_eval);
    }
private:
    void *m_eval;
};

class ExpressionVector {
public:
    ExpressionVector() { }
    ExpressionVector(const std::vector<std::string> &componentExprs) {
        for (const auto &expr : componentExprs)
            add(expr);
    }

    void add(const std::string &expr) {
        m_evaluators.push_back(std::make_shared<Expression>(expr));
    }

    size_t size() const { return m_evaluators.size(); }

    template<size_t _N>
    VectorND<_N> eval(const ExpressionEnvironment &e) const {
        VectorND<_N> result;
        if (m_evaluators.size() != _N)
            throw std::runtime_error("Invalid evaluation size.");
        for (size_t i = 0; i < _N; ++i)
            result[i] = m_evaluators.at(i)->eval(e);

        return result;
    }

private:
    std::vector<std::shared_ptr<Expression>> m_evaluators;
};

#endif /* end of include guard: EXPRESSIONVECTOR_HH */
