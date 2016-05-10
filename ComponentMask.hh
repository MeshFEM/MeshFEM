#ifndef COMPONENTMASK_HH
#define COMPONENTMASK_HH
#include <bitset>

class ComponentMask {
public:
    ComponentMask(const std::string &components = "") {
        setComponentString(components);
    }

    void setComponentString(const std::string &components) {
        m_active.reset();
        if (components.find("x") != std::string::npos) m_active.set(0);
        if (components.find("y") != std::string::npos) m_active.set(1);
        if (components.find("z") != std::string::npos) m_active.set(2);
        if (m_active.count() != components.size())
            throw std::runtime_error("invalid component specifier: '" + components + "'");
    }

    bool has(size_t c) const { return m_active.test(c); }
    bool hasX()        const { return m_active[0]; }
    bool hasY()        const { return m_active[1]; }
    bool hasZ()        const { return m_active[2]; }
    bool hasAny(size_t dim) const { return count(dim) > 0; }
    bool hasAll(size_t dim) const { return count(dim) == dim; }
    // Number of active components for dimension (2 or 3)
    size_t count(size_t dim) const {
        if (dim == 3)      return m_active.count();
        else if (dim == 2) return m_active.count() - (hasZ() ? 1 : 0);
        else throw std::runtime_error("Illegal dimension");
    }

    void set()           { m_active.set(); }
    void set(size_t c)   { m_active.set(c); }
    void clear()         { m_active.reset(); }
    void clear(size_t c) { m_active.reset(c); }

    bool operator==(const ComponentMask &b) const { return m_active == b.m_active; }
    bool operator!=(const ComponentMask &b) const { return m_active != b.m_active; }

    // Apply the mask to a vector, clearing any component not set in the mask.
    template<int N>
    VectorND<N> apply(const VectorND<N> &v) const {
        VectorND<N> result(v);
        for (size_t c = 0; c < N; ++c)
            if (!has(c)) result[c] = 0;
        return result;
    }

    std::string componentString() const {
        std::string result;
        if (hasX()) result += "x";
        if (hasY()) result += "y";
        if (hasZ()) result += "z";
        return result;
    }

    friend std::ostream &operator<<(std::ostream &os, const ComponentMask &cm) {
        if (cm.hasX()) os << "x";
        if (cm.hasY()) os << "y";
        if (cm.hasZ()) os << "z";
        return os;
    }

private:
    std::bitset<3> m_active;
};

#endif /* end of include guard: COMPONENTMASK_HH */
