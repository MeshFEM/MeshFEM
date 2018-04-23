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
#include <tinyexpr.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <memory>

// Expression environment for storing variable and their values
struct ExpressionEnvironment {
    void setValue(const std::string &name, double value) {
        for (auto & var : m_variables) {
            if (var->name == name) {
                var->value = value;
                return;
            }
        }
        auto var = std::make_shared<Variable>();
        var->name = name;
        var->value = value;
        m_variables.push_back(var);
        m_ptrs.push_back({ var->name.c_str(), &(var->value), TE_VARIABLE, 0 });
    }

    // Sets a value for each component with the names name1, name2...
    template<class _Vec>
    void setVectorValue(const std::string &name, const _Vec &v) {
        for (int i = 0; i < _Vec::RowsAtCompileTime; ++i)
            setValue(name + std::to_string(i), v[i]);
    }

    void setVectorValue(const std::string &name, const std::vector<Real> &v) {
        for (size_t i = 0; i < v.size(); ++i)
            setValue(name + std::to_string(i), v[i]);
    }

    template<class _Vec>
    void setXYZ(_Vec &v) {
        int N = _Vec::RowsAtCompileTime;
        if (!(N == 2 || N == 3)) throw std::runtime_error("Bad vector size");
        setValue("x", v[0]); setValue("y", v[1]);
        setValue("z", (N == 3) ? v[2] : 0);
    }

    size_t numVars() const { return m_variables.size(); }

    const te_variable * lookup() const { return m_ptrs.data(); }

private:
    struct Variable {
        std::string name;
        double value;
    };

    std::vector<std::shared_ptr<Variable>> m_variables;
    std::vector<te_variable> m_ptrs;
};

// Expression wrapper handling destruction that should be wrapped in a smart
// pointer.
class Expression {
public:
    // Copy the expression string (will be compiled lazily)
    Expression(std::string s)
        : m_str(s)
        , m_expr(nullptr)
        , m_lookup(nullptr)
    { }

    // DAANNGEROUS... get rid of it
    Expression &operator=(const Expression&e) = delete;

    // Not const because it lazily compiles the expression for the given environment
    double eval(const ExpressionEnvironment &e) {
        if (e.numVars() == 0) {
            throw std::runtime_error("Empty environment");
        }

        if (m_lookup != e.lookup()) {
            // Free previously compiled expression, if any
            if (m_expr) { te_free(m_expr); }

            // Lazy compilation of the expression + environment
            int error = 0;
            m_lookup = e.lookup();
            m_expr = te_compile(m_str.c_str(), m_lookup, e.numVars(), &error);
            if (error) {
                throw std::runtime_error("Failed to parse expression '" + m_str + "'");
            }
        }

        return te_eval(m_expr);
    }

    ~Expression() {
        if (m_expr) { te_free(m_expr); }
    }
private:
    std::string m_str;
    te_expr *m_expr;
    const te_variable *m_lookup;
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
