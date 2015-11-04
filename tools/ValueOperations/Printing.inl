// "Nested" bool allows us to customize printing style when the value is part
// of a collection.
template<bool nested>
struct PrintImpl<SValue, nested> {
    static void run(std::ostream &os, const SValue &v) { os << v.value; }
};

// Print vectors on a single row
// (more compact, and multiple vector values appear distinct)
template<bool nested>
struct PrintImpl<VValue, nested> {
    static void run(std::ostream &os, const VValue &v) {
        for (size_t i = 0; i < v.dim(); ++i) {
            if (i) os << "\t";
            os << v[i];
        }
    }
};

// Print symmetric matrices on a single line
template<bool nested>
struct PrintImpl<SMValue, nested> {
    static void run(std::ostream &os, const SMValue &v) {
        os << "SMatrix{";
        for (size_t i = 0; i < v.dim(); ++i) {
            if (i) os << "\t";
            os << v[i];
        }
        os << "}";
    }
};

// Print fields one value per line
template<typename T, bool nested>
struct PrintImpl<FieldValue<T>, nested> {
    static void run(std::ostream &os, const FieldValue<T> &v) {
        for (size_t i = 0; i < v.dim(); ++i) {
            if (i) os << std::endl;
            PrintImpl<T, true>::run(os, v[i]);
        }
    }
};

// Print interpolants on a single line
template<typename T, bool nested>
struct PrintImpl<InterpolantValue<T>, nested> {
    static void run(std::ostream &os, const InterpolantValue<T> &v) {
        os << "Interpolant" << v.simplexDimension() << "{";
        for (size_t i = 0; i < v.dim(); ++i) {
            if (i) os << ", ";
            PrintImpl<T, true>::run(os, v[i]);
        }
        os << "}";
    }
};
