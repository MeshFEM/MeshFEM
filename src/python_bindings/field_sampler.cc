#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/FieldSampler.hh>

#include "BindingInstantiations.hh"

template<class PyFS>
struct SamplingMeshBinder {
    SamplingMeshBinder(PyFS &pyFS) : m_pyFS(pyFS) { }

    template<class Mesh>
    void bind(py::module &/* m */, py::module &/* detail_module */) {
        m_pyFS.def(py::init([](std::shared_ptr<const Mesh> mesh) {
                        return FieldSampler::construct(mesh);
                    }), py::arg("mesh"))
        ;
    }

private:
    PyFS &m_pyFS;
};

PYBIND11_MODULE(field_sampler, m)
{
    py::class_<FieldSampler, std::unique_ptr<FieldSampler>> pyFS(m, "FieldSampler");
    pyFS.def(py::init([](Eigen::Ref<const Eigen::MatrixXd> V,
                         Eigen::Ref<const Eigen::MatrixXi> F) {
                    return FieldSampler::construct(V, F);
                }), py::arg("V"), py::arg("F"))
        .def("closestElementAndPoint", [](const FieldSampler &s, Eigen::Ref<const Eigen::MatrixXd> P) {
                using RType = std::tuple<Eigen::VectorXi,  // I
                                         Eigen::MatrixXd>; // C
                Eigen::VectorXd sq_dists;
                RType result;
                s.closestElementAndPoint(P, sq_dists, std::get<0>(result), std::get<1>(result));
                return result;
            }, py::arg("P"))
        .def("closestElementAndBaryCoords", [](const FieldSampler &s, Eigen::Ref<const Eigen::MatrixXd> P) {
                using RType = std::tuple<Eigen::VectorXi,  // I
                                         Eigen::MatrixXd>; // B
                RType result;
                s.closestElementAndBaryCoords(P, std::get<0>(result), std::get<1>(result));
                return result;
            }, py::arg("P"))
        .def("closestNodeAndSqDist", [](const FieldSampler &s, Eigen::Ref<const Eigen::MatrixXd> P) {
                using RType = std::tuple<Eigen::VectorXi,  // NI
                                         Eigen::VectorXd>; // sqDist
                RType result;
                s.closestNodeAndSqDist(P, std::get<0>(result), std::get<1>(result));
                return result;
            }, py::arg("P"))
        .def("contains", [](const FieldSampler &s,
                          Eigen::Ref<const Eigen::MatrixXd> P, double eps) {
                return s.contains(P, eps);
            }, py::arg("P"), py::arg("eps") = 1e-10)
        .def("sample", [](const FieldSampler &s,
                          Eigen::Ref<const Eigen::MatrixXd> P,
                          Eigen::Ref<const Eigen::MatrixXd> fieldValues) {
                return s.sample(P, fieldValues);
            }, py::arg("P"), py::arg("fieldValues")) // Piecewise linear field
        ;

    generateMeshSpecificBindings(m, m, SamplingMeshBinder<decltype(pyFS)>(pyFS));
}
