#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/libigl_aabb/point_simplex_squared_distance.h>

#include <MeshFEM/FieldSampler.hh>
#include <MeshFEM/EmbeddedElement.hh>
#include <MeshFEM/Parallelism.hh>
#include <MeshFEM/algorithms/marching_tetrahedra.hh>

#include "BindingInstantiations.hh"
#include <MeshFEM/Utilities/MeshConversion.hh>

// `marching_tetrahedra` is specific to tetrahedral meshes...
template<class Mesh>
struct BindMarchingTetrahedra {
    static void bind(py::module &/* m */) { };
};

template<size_t _Deg, class _EmbeddingSpace>
struct BindMarchingTetrahedra<FEMMesh<3, _Deg, _EmbeddingSpace>> {
    using Mesh = FEMMesh<3, _Deg, _EmbeddingSpace>;
    static void bind(py::module &m) {
        m.def("marching_tetrahedra", [](const Mesh &m, const Eigen::VectorXd &f, bool sublevelset, bool lerp) {
            std::vector<MeshIO::IOVertex> outVertices;
            std::vector<MeshIO::IOElement> outElements;
            std::vector<ContourSamplePtInfo> outSampleInfo;
            size_t numContourTris = marching_tetrahedra(m, f, outVertices, outElements, outSampleInfo, sublevelset, lerp);
            return std::make_tuple(getV(outVertices), getF(outElements), outSampleInfo, numContourTris);
        }, py::arg("mesh"), py::arg("f"), py::arg("sublevelset") = true, py::arg("linearInterpolation") = true)
        ;
    };
};

template<class PyFS>
struct SamplingMeshBinder {
    SamplingMeshBinder(PyFS &pyFS) : m_pyFS(pyFS) { }

    template<class Mesh>
    void bind(py::module &m, py::module &/* detail_module */) {
        m_pyFS.def(py::init([](std::shared_ptr<const Mesh> mesh) {
                        return FieldSampler::construct(mesh);
                    }), py::arg("mesh"))
        ;
        BindMarchingTetrahedra<Mesh>::bind(m);
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

    m.def("closestPointsInTetrahedra" ,
        [](Eigen::Ref<const Eigen::MatrixXd> V,
           Eigen::Ref<const Eigen::MatrixXi> T,
           const Eigen::Vector3d &p) {
            if (T.cols() != 4) throw std::runtime_error("Expected tetrahedra");
            const size_t ne = T.rows();
            using AES = AffineEmbeddedSimplex<3, Eigen::Vector3d>;
            AES aes;
            std::pair<Eigen::VectorXd, Eigen::MatrixXd> result;

            Eigen::VectorXd &dists = result.first;
            Eigen::MatrixXd &closestPts = result.second;

            dists.resize(ne);
            closestPts.resize(ne, 3);

            parallel_for_range(0, ne, [&](size_t ei) {
                aes.embed_indexed(V, T, ei);
                if (aes.contains(p)) {
                    dists[ei] = 0.0;
                    closestPts.row(ei) = p.transpose();
                    return;
                }
                Eigen::RowVector4i t = T.row(ei);
                double min_d = std::numeric_limits<double>::max();
                Eigen::Vector3d closest_c;
                for (size_t face = 0; face < 4; ++face) {
                    double d;
                    Eigen::Vector3d c;
                    iglaabb::point_simplex_squared_distance<3>(p, V, t.leftCols(3).eval(), 0, d, c);
                    t = (Eigen::RowVector4i() << t.rightCols(3), t[0]).finished(); // move to next face (orientation irrelevant)
                    if (d < min_d) {
                        min_d = d;
                        closest_c = c;
                    }
                }
                dists[ei] = min_d;
                closestPts.row(ei) = closest_c;
            });

            return result;
        },
        "Find the closest point to `p` separately for each tetrahedra of mesh (V, T) along with its closest distance"
    );

    generateMeshSpecificBindings(m, m, SamplingMeshBinder<decltype(pyFS)>(pyFS));
}
