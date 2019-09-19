#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <Eigen/Dense>
#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Utilities/TemplateName.hh>
#include <MeshFEM/Meshing.hh>

template<size_t _Dimension, size_t _Degree>
std::string
getMeshName()
{
    return (getFEMName<_Degree>() + "FEM" + std::to_string(_Dimension) + "DMesh").c_str();
}

/**
 *  Helper struct to allow partial specialization of template function.
 */
template<size_t _Dimension, size_t _Degree>
struct ParameterSpecificMeshBinding
{
    using Mesh = FEMMesh<_Dimension, _Degree, Eigen::Matrix<double, _Dimension, 1>>;
    // static void bind(py::module& module, py::class_<Mesh>& mesh_bindings) {}
};

template<size_t _Degree>
struct ParameterSpecificMeshBinding<2, _Degree>
{
    static constexpr size_t Dimension = 2;
    using Mesh = FEMMesh<Dimension, _Degree, Eigen::Matrix<double, Dimension, 1>>;
    static void bind(py::module& module, py::class_<Mesh>& mesh_bindings)
    {
        module.def(
          ("triangularizePolygonSet" + getFEMName<_Degree>() + "FEM").c_str(),
          [](const std::vector<Eigen::Vector2d>& points,
             const std::vector<std::vector<std::pair<size_t, size_t>>>& polygons,
             const std::vector<Eigen::Vector2d>& holes,
             Real target_area,
             Real strong_connections) {
              using Triangulation =
                PolygonSetTriangulation<double, Eigen::Vector2d, std::pair<size_t, size_t>>;
              Triangulation triangulation = Triangulation(points, polygons, holes, target_area, strong_connections);
              return Mesh(triangulation.getElements(), triangulation.getVertices());
          });

        mesh_bindings.def("rotate", [&](Mesh& mesh, double angle) {
            Eigen::Rotation2D<double> rotation(angle);
            std::vector<Eigen::Vector3d> new_positions(mesh.numNodes());
            for (const auto& node : mesh.nodes())
            {
                new_positions[node.index()] = padTo3D(rotation * node->p);
            }
            mesh.setNodePositions(new_positions);
        });
    }
};

template<size_t _Degree>
struct ParameterSpecificMeshBinding<3, _Degree>
{
    static constexpr size_t Dimension = 3;
    using Mesh = FEMMesh<Dimension, _Degree, Eigen::Matrix<double, Dimension, 1>>;
    using Vector = typename Mesh::EmbeddingSpace;
    static void bind(py::module& /* module */, py::class_<Mesh>& mesh_bindings)
    {
        mesh_bindings.def("rotate", [&](Mesh& mesh, double angle, const Vector& axis) {
            Eigen::AngleAxis<double> rotation(angle, axis);
            std::vector<Vector> new_positions(mesh.numNodes());
            for (const auto& node : mesh.nodes())
            {
                new_positions[node.index()] = rotation * node->p;
            }
            mesh.setNodePositions(new_positions);
        });
    }
};

template<size_t _Dimension, size_t _Degree>
void
bindMesh(py::module& module)
{
    using Mesh = FEMMesh<_Dimension, _Degree, Eigen::Matrix<double, _Dimension, 1>>;

    py::class_<Mesh> mesh_bindings(module, getMeshName<_Dimension, _Degree>().c_str());
    mesh_bindings
      .def(py::init([](const std::string& path) {
               std::vector<MeshIO::IOVertex> vertices;
               std::vector<MeshIO::IOElement> elements;
               MeshIO::load(path, vertices, elements);
               return Mesh(elements, vertices);
           }),
           py::arg("path"))
      .def("vertices",
           [&](const Mesh& m) {
               Eigen::Matrix<double, Eigen::Dynamic, _Dimension> V(m.numVertices(), _Dimension);
               for (const auto& v : m.vertices())
                   V.row(v.index()) = v.node()->p;
               return V;
           })
      .def("elements",
           [&](const Mesh& m) {
               std::vector<std::array<size_t, _Dimension + 1>> elements;
               elements.reserve(m.numElements());
               std::array<size_t, _Dimension + 1> current_element;
               for (const auto& e : m.elements())
               {
                   for (const auto& v : e.vertices())
                   {
                       current_element[v.localIndex()] = v.index();
                   }
                   elements.push_back(current_element);
               }
               return elements;
           })
      .def("boundary_elements",
           [&](const Mesh& m) {
               std::vector<std::array<size_t, _Dimension>> elements;
               elements.reserve(m.numBoundaryElements());
               std::array<size_t, _Dimension> current_element;
               for (const auto& e : m.boundaryElements())
               {
                   for (const auto& v : e.vertices())
                   {
                       current_element[v.localIndex()] = v.volumeVertex().index();
                   }
                   elements.push_back(current_element);
               }
               return elements;
           })
      .def("numVertices", &Mesh::numVertices)
      .def("numElements", &Mesh::numElements)
      .def("numNodes",    &Mesh::numNodes)
      .def("save", [&](const Mesh& m, const std::string& path) { return MeshIO::save(path, m); })
      .def("is_tet_mesh", [&](const Mesh& m) { return (m.element(0).vertices().size() == 4); })
      .def_property_readonly("volume", [](const Mesh& m) { return m.boundingBox().volume(); })
      .def_property_readonly_static("degree", [](py::object) { return _Degree; })
      .def_property_readonly_static("dimension", [](py::object) { return _Dimension; });

    ParameterSpecificMeshBinding<_Dimension, _Degree>::bind(module, mesh_bindings);
}

template<size_t _Dimension>
void
bindPeriodicCondition(py::module& module)
{
    using LinearMesh = FEMMesh<_Dimension, 1, Eigen::Matrix<double, _Dimension, 1>>;
    using QuadraticMesh = FEMMesh<_Dimension, 2, Eigen::Matrix<double, _Dimension, 1>>;

    py::class_<PeriodicCondition<_Dimension>>(
      module, ("PeriodicCondition" + std::to_string(_Dimension) + "D").c_str())
      .def(py::init<const LinearMesh&, double, bool, std::vector<size_t>>(), py::arg("mesh"), py::arg("eps") = 1e-7, py::arg("ignore_mismatch") = false, py::arg("ignore_dims") = std::vector<size_t>())
      .def(py::init<const QuadraticMesh&, double, bool, std::vector<size_t>>(), py::arg("mesh"), py::arg("eps") = 1e-7, py::arg("ignore_mismatch") = false, py::arg("ignore_dims") = std::vector<size_t>())
      .def(py::init<const LinearMesh&, const std::string&>(),
           py::arg("mesh"),
           py::arg("periodic_condition_file"))
      .def(py::init<const QuadraticMesh&, const std::string&>(),
           py::arg("mesh"),
           py::arg("periodic_condition_file"))
      .def("periodicDoFsForNodes", &PeriodicCondition<_Dimension>::periodicDoFsForNodes);
}

PYBIND11_MODULE(mesh, m)
{
    bindMesh<3, 1>(m);
    bindMesh<2, 1>(m);
    bindMesh<3, 2>(m);
    bindMesh<2, 2>(m);

    bindPeriodicCondition<2>(m);
    bindPeriodicCondition<3>(m);

    using PSetTriangulation = PolygonSetTriangulation<
        double, Eigen::Vector2d, std::pair<size_t, size_t>>;

    py::class_<PSetTriangulation>(m, "PolygonSetTriangulation")
        .def(py::init<
                const std::vector<Eigen::Vector2d>&,
                const std::vector<std::vector<std::pair<size_t, size_t>>>&,
                const std::vector<Eigen::Vector2d>&,
                double, double>())
        .def("getLinearMesh", [](const PSetTriangulation& triangulation)
                {
                    return std::make_unique<FEMMesh<2, 1, Eigen::Vector2d>>(
                            triangulation.getElements(),
                            triangulation.getVertices());
                })
        .def("getQuadraticMesh", [](const PSetTriangulation& triangulation)
                {
                    return std::make_unique<FEMMesh<2, 2, Eigen::Vector2d>>(
                            triangulation.getElements(),
                            triangulation.getVertices());
                });
}
