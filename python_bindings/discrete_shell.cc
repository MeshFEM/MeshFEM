#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "../DiscreteShell.hh"
#include "../3rdparty//MeshFEM/src/python_bindings/MeshEntities.hh"
#include "../HingeBendingEnergy.hh"
#include "../HingePanelizationEnergy.hh"
#include <MeshFEM/Loads/Gravity.hh>
#include <MeshFEM/Loads/Springs.hh>

using APC = Loads::AttachmentPointCoordinate<double>;

PYBIND11_MODULE(discrete_shell, m)
{
    using EO = ElasticObject<double>;
    using DS = DiscreteShell<HingePanelizationEnergy>;

    using Mesh = DS::Mesh;

    py::class_<DS, EO, std::shared_ptr<DS>>(m, "DiscreteShell")
          .def_property_readonly_static("dimension", [](py::object /* self */) { return 3; })
          .def(py::init<const std::shared_ptr<Mesh> &, double, double>(),
                  py::arg("mesh"), py::arg("youngModulus") = 200, py::arg("poissonRatio") = 0.3)
          .def("mesh", [](const DS &ds) { return ds.mesh(); })
          .def("numHinges", &DS::numHinges)
          .def("visualizationGeometry", [](const DS &ds, double normalCreaseAngle) {
                Mesh visMesh(getF(ds.mesh()), ds.deformedPositions().topRows(ds.numVertices()));
                return getVisualizationGeometry(visMesh, normalCreaseAngle);
             }, py::arg("normalCreaseAngle") = M_PI)
          .def("visualizationField", [](const DS &ds, const Eigen::VectorXd &f) { return getVisualizationField(ds.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("visualizationField", [](const DS &ds, const Eigen::MatrixXd &f) { return getVisualizationField(ds.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def_readwrite("bendingStiffness", &DS::bendingStiffness)
          .def_readwrite("h",                &DS::h)
          .def_property("delta", &DS::getDelta, &DS::setDelta)
          ;

    using GLoad = Loads::Gravity<DS>;
    using Load = Loads::Load<double>;
    py::class_<GLoad, Load, std::shared_ptr<GLoad>>(m, "Gravity")
          .def(py::init([&](const std::shared_ptr<DS> &obj, double rho, const Eigen::Vector3d &g) {
                return std::make_shared<GLoad>(obj, rho, g);
            }), py::arg("obj"), py::arg("rho"), py::arg("g"))
          .def_property("rho", &GLoad::get_rho, &GLoad::set_rho)
          ;
      
      using Springs = Loads::Springs<DS>;
        using VXd  = Eigen::VectorXd;
        py::class_<Springs, Load, std::shared_ptr<Springs>>(m, "Springs")
            .def(py::init<const std::shared_ptr<DS> &, const std::vector<APC> &, const std::vector<APC> &, double>(),
            py::arg("obj"), py::arg("coordsA"), py::arg("coordsB"), py::arg("stiffness"))
            .def("getStiffnesses", &Springs::getStiffnesses)
            .def("setStiffnesses", [](Springs &s, double     val ) { s.setStiffnesses(val ); }, py::arg("val"))
            .def("setStiffnesses", [](Springs &s, const VXd &vals) { s.setStiffnesses(vals); }, py::arg("vals"))
            ;
}
