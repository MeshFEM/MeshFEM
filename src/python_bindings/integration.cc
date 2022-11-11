#include "MeshFEM/Types.hh"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <MeshFEM/MonteCarloIntegration.hh>
#include <MeshFEM/GaussQuadrature.hh>
#include "BindingInstantiations.hh"

template<class VecN, class VecBC, class Element>
VecN ptForBarycoords(const VecBC &x, const Element &e) {
    VecN pt = VecN::Zero();
    for (auto v : e.vertices())
        pt += x[v.localIndex()] * v.node()->p;
    return pt;
}

struct MeshBinder {
    template<class Mesh>
    static void bind(py::module &bm, py::module &em) {
        static constexpr size_t K = Mesh::K;
        static constexpr size_t N = Mesh::EmbeddingDimension;

        using VecBC = VecN_T<Real, K + 1>;
        using VecN  = VecN_T<Real, N>;

        ////////////////////////////////////////////////////////////////////////
        // Barycentric versions: function `f` takes barycentric coordinates.
        ////////////////////////////////////////////////////////////////////////
        bm.def("monteCarlo", [](size_t ns, const Mesh &m,
                                std::function<Real(size_t, const VecBC &)> f) {
                Real result = 0;
                for (auto e : m.elements()) {
                    result += monteCarloIntegration<K>([&](const EvalPt<K> &x) {
                                return f(e.index(), Eigen::Map<const VecBC>(x.data()));
                            }, ns, e->volume());
                }
                return result;
            }, py::arg("ns"), py::arg("m"), py::arg("f"),
            "Approximately integrate scalar field f(ei, barycoords) using `ns` samples over each element `ei` in mesh `m`, returning the total integral.");
        bm.def("monteCarloElement", [](size_t ns, const Mesh &m, size_t ei,
                                     std::function<Real(VecN_T<Real, Mesh::K + 1>)> f) {
                if (ei > m.numElements()) throw std::runtime_error("Mesh element index out of bounds");
                return monteCarloIntegration<K>([&](const EvalPt<K> &x) {
                            return f(Eigen::Map<const VecBC>(x.data()));
                        }, ns, m.element(ei)->volume());
            }, py::arg("ns"), py::arg("m"), py::arg("ei"), py::arg("f"),
            "Approximately integrate scalar field f(barycoords) over element `ei` of mesh `m` using `ns` samples.");

        bm.def("monteCarloElementAverage", [](size_t ns, const Mesh &m, size_t ei,
                                     std::function<Real(VecN_T<Real, Mesh::K + 1>)> f) {
                if (ei > m.numElements()) throw std::runtime_error("Mesh element index out of bounds");
                return monteCarloIntegration<K>([&](const EvalPt<K> &x) {
                            return f(Eigen::Map<const VecBC>(x.data()));
                        }, ns, 1.0);
            }, py::arg("ns"), py::arg("m"), py::arg("ei"), py::arg("f"),
            "Approximately average scalar field f(barycoords) over element `ei` of mesh `m` using `ns` samples.");

        bm.def("gaussQuadrature", [](size_t deg, const Mesh &m, std::function<Real(size_t, const VecBC &)> f) {
                Real result = 0;
                for (auto e : m.elements()) {
                    if      (deg <= 1) result += Quadrature<K, 1>::integrate([&](const EvalPt<K> &x) { return f(e.index(), Eigen::Map<const VecBC>(x.data())); }, e->volume());
                    else if (deg == 2) result += Quadrature<K, 2>::integrate([&](const EvalPt<K> &x) { return f(e.index(), Eigen::Map<const VecBC>(x.data())); }, e->volume());
                    else if (deg == 3) result += Quadrature<K, 3>::integrate([&](const EvalPt<K> &x) { return f(e.index(), Eigen::Map<const VecBC>(x.data())); }, e->volume());
                    else if (deg == 4) result += Quadrature<K, 4>::integrate([&](const EvalPt<K> &x) { return f(e.index(), Eigen::Map<const VecBC>(x.data())); }, e->volume());
                    else throw std::runtime_error("Unimplemented degree");
                }
                return result;
            }, py::arg("deg"), py::arg("m"), py::arg("f"),
            "Exact integration of a scalar field f(barycoords) of degree `deg` over each element `ei` in mesh `m`, returning the total integral.");

        bm.def("gaussQuadratureElement", [](size_t deg, const Mesh &m, size_t ei, std::function<Real(size_t, const VecBC &)> f) {
                if (ei > m.numElements()) throw std::runtime_error("Mesh element index out of bounds");
                if (deg <= 1) return Quadrature<K, 1>::integrate([&](const EvalPt<K> &x) { return f(ei, Eigen::Map<const VecBC>(x.data())); }, m.element(ei)->volume());
                if (deg == 2) return Quadrature<K, 2>::integrate([&](const EvalPt<K> &x) { return f(ei, Eigen::Map<const VecBC>(x.data())); }, m.element(ei)->volume());
                if (deg == 3) return Quadrature<K, 3>::integrate([&](const EvalPt<K> &x) { return f(ei, Eigen::Map<const VecBC>(x.data())); }, m.element(ei)->volume());
                if (deg == 4) return Quadrature<K, 4>::integrate([&](const EvalPt<K> &x) { return f(ei, Eigen::Map<const VecBC>(x.data())); }, m.element(ei)->volume());
                throw std::runtime_error("Unimplemented degree");
            }, py::arg("deg"), py::arg("m"), py::arg("ei"), py::arg("f"),
            "Exact integration of a scalar field f(barycoords) of degree `deg` over each element `ei` in mesh `m`, returning the total integral.");

        ////////////////////////////////////////////////////////////////////////
        // Euclidean versions: function `f` takes Euclidean coordinates.
        ////////////////////////////////////////////////////////////////////////
        em.def("monteCarlo", [](size_t ns, const Mesh &m,
                                std::function<Real(size_t, VecN)> f) {
                Real result = 0;
                for (auto e : m.elements()) {
                    result += monteCarloIntegration<K>([&](const EvalPt<K> &x) { return f(e.index(), ptForBarycoords<VecN>(x, e)); }, ns, e->volume());
                }
                return result;
            }, py::arg("ns"), py::arg("m"), py::arg("f"),
            "Approximately integrate scalar field f(ei, x) over each element `ei` in mesh `m`, returning the total integral.");
        em.def("monteCarloElement", [](size_t ns, const Mesh &m, size_t ei,
                                     std::function<Real(VecN)> f) {
                if (ei > m.numElements()) throw std::runtime_error("Mesh element index out of bounds");
                auto e = m.element(ei);
                return monteCarloIntegration<K>([&](const EvalPt<K> &x) { return f(ptForBarycoords<VecN>(x, e)); }, ns, e->volume());
            }, py::arg("ns"), py::arg("m"), py::arg("ei"), py::arg("f"),
            "Approximately integrate scalar field f(x) over element `ei` of mesh `m`.");

        em.def("monteCarloElementAverage", [](size_t ns, const Mesh &m, size_t ei,
                                     std::function<Real(VecN)> f) {
                if (ei > m.numElements()) throw std::runtime_error("Mesh element index out of bounds");
                auto e = m.element(ei);
                return monteCarloIntegration<K>([&](const EvalPt<K> &x) { return f(ptForBarycoords<VecN>(x, e)); }, ns, 1.0);
            }, py::arg("ns"), py::arg("m"), py::arg("ei"), py::arg("f"),
            "Approximately average scalar field f(x) over element `ei` of mesh `m`.");

        em.def("gaussQuadrature", [](size_t deg, const Mesh &m, std::function<Real(size_t, const VecN &)> f) {
                Real result = 0;
                for (auto e : m.elements()) {
                    if      (deg <= 1) result += Quadrature<K, 1>::integrate([&](const EvalPt<K> &x) { return f(e.index(), ptForBarycoords<VecN>(x.data(), e)); }, e->volume());
                    else if (deg == 2) result += Quadrature<K, 2>::integrate([&](const EvalPt<K> &x) { return f(e.index(), ptForBarycoords<VecN>(x.data(), e)); }, e->volume());
                    else if (deg == 3) result += Quadrature<K, 3>::integrate([&](const EvalPt<K> &x) { return f(e.index(), ptForBarycoords<VecN>(x.data(), e)); }, e->volume());
                    else if (deg == 4) result += Quadrature<K, 4>::integrate([&](const EvalPt<K> &x) { return f(e.index(), ptForBarycoords<VecN>(x.data(), e)); }, e->volume());
                    else throw std::runtime_error("Unimplemented degree");
                }
                return result;
            }, py::arg("deg"), py::arg("m"), py::arg("f"),
            "Exact integration of a scalar field f(x) of degree `deg` over each element `ei` in mesh `m`, returning the total integral.");

        em.def("gaussQuadratureElement", [](size_t deg, const Mesh &m, size_t ei, std::function<Real(size_t, const VecN &)> f) {
                if (ei > m.numElements()) throw std::runtime_error("Mesh element index out of bounds");
                if (deg <= 1) return Quadrature<K, 1>::integrate([&](const EvalPt<K> &x) { return f(ei, ptForBarycoords<VecN>(x.data(), m.element(ei))); }, m.element(ei)->volume());
                if (deg == 2) return Quadrature<K, 2>::integrate([&](const EvalPt<K> &x) { return f(ei, ptForBarycoords<VecN>(x.data(), m.element(ei))); }, m.element(ei)->volume());
                if (deg == 3) return Quadrature<K, 3>::integrate([&](const EvalPt<K> &x) { return f(ei, ptForBarycoords<VecN>(x.data(), m.element(ei))); }, m.element(ei)->volume());
                if (deg == 4) return Quadrature<K, 4>::integrate([&](const EvalPt<K> &x) { return f(ei, ptForBarycoords<VecN>(x.data(), m.element(ei))); }, m.element(ei)->volume());
                throw std::runtime_error("Unimplemented degree");
            }, py::arg("deg"), py::arg("m"), py::arg("ei"), py::arg("f"),
            "Exact integration of a scalar field f(x) of degree `deg` over each element `ei` in mesh `m`, returning the total integral.");
    }
};

PYBIND11_MODULE(integration, m)
{
    m.doc() = "MeshFEM quadrature and Monte Carlo integration bindings";

    py::module barycentric_module = m.def_submodule("barycentric");
    py::module   euclidean_module = m.def_submodule("euclidean");

    barycentric_module.doc() = "Barycentric coordinate versions of integration routines";
      euclidean_module.doc() = "Euclidean coordinate versions of integration routines";

    generateMeshSpecificBindings<MeshBinder>(barycentric_module, euclidean_module, MeshBinder());
}
