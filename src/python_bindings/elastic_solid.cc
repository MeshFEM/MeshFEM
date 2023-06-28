#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/ElasticSolid.hh>
#include <MeshFEM/ElasticSolidRotExtrap.hh>
#include <MeshFEM/MassMatrix.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include "MeshEntities.hh"

#include "BindingInstantiations.hh"

template<size_t NewDeg, class ES>
py::object toDegree(const ES &es) {
    return py::cast(new ElasticSolid<ES::K, NewDeg, typename ES::EmbeddingSpace, typename ES::Energy>(es),
                    py::return_value_policy::take_ownership);
}

struct ElasticSolidBinder {
    template<class ES>
    static void bind(py::module &module, py::module &detail_module) {
        static constexpr size_t K   = ES::K;
        static constexpr size_t N   = ES::K;
        static constexpr size_t Deg = ES::Deg;
        using Real   = typename ES::Real;
        using EO     = ElasticObject<Real>;
        using VM     = typename EO::VariableMask;
        using Energy = typename ES::Energy;
        using MXNd   = Eigen::Matrix<Real, Eigen::Dynamic, N>;
        using Mesh   = typename ES::Mesh;
        using EmbeddingSpace = typename Mesh::EmbeddingSpace;

        module.def("ElasticSolid", [](std::shared_ptr<Mesh> m, const Energy &e) { return std::make_shared<ES>(e, m); }, py::arg("mesh"), py::arg("energy"));

        const std::string name = getElasticSolidName<Energy, K, Deg, VecN_T<Real, N>>();
        py::class_<ES, EO, std::shared_ptr<ES>> pyES(detail_module, name.c_str());
        pyES
          .def_property_readonly_static("dimension",   [](py::object /* self */) { return N; })
          .def_property_readonly_static("degree",      [](py::object /* self */) { return Deg; })
          .def_property_readonly_static("energy_name", [](py::object /* self */) { return getEnergyName<Energy>(); })
          .def("mesh",                      &ES::mesh)
          .def("numElements",               &ES::numElements)
          .def("setDeformedPositions",      &ES::setDeformedPositions)
          .def("applyRigidTransform",       &ES::applyRigidTransform, py::arg("R"), py::arg("t"))
          .def("prepareRigidMotionPins",    &ES::prepareRigidMotionPins)
          .def("filterRMPinArtifacts",      &ES::filterRMPinArtifacts, py::arg("pinVertices"))
          .def("getDeformedPositions",      &ES::deformedPositions)
          .def("getRestPositions",          &ES::restNodePositions)
          .def("getNodeDisplacements",      &ES::nodeDisplacements)
          .def("getEnergyDensity",          &ES::getEnergyDensity, py::arg("ei"), py::return_value_policy::reference_internal)
          .def("greenStrain",               [](const ES &es, size_t ei) { return es.greenStrain(ei); }, py::arg("ei"))
          .def("greenStrain",               [](const ES &es, size_t ei, const typename ES::EvalPtK &baryCoords) { return es.greenStrain(ei, baryCoords); }, py::arg("ei"), py::arg("baryCoords"))
          .def("vertexGreenStrains",        &ES::vertexGreenStrains)
          .def("cauchyStress",              [](const ES &es, size_t ei) { return es.cauchyStress(ei); }, py::arg("ei"))
          .def("cauchyStress",              [](const ES &es, size_t ei, const typename ES::EvalPtK &baryCoords) { return es.cauchyStress(ei, baryCoords); }, py::arg("ei"), py::arg("baryCoords"))
          .def("vertexCauchyStresses",      &ES::vertexCauchyStresses)
          .def("surfaceStressLpNorm",       &ES::surfaceStressLpNorm, py::arg("p"))
          .def("visualizationGeometry", [](const ES &obj, double normalCreaseAngle) {
                FEMMesh<Mesh::K, 1, EmbeddingSpace> visMesh(getF(obj.mesh()), obj.deformedPositions().topRows(obj.numVertices()));
                return getVisualizationGeometry(visMesh, normalCreaseAngle);
             }, py::arg("normalCreaseAngle") = M_PI)
          .def("visualizationField", [](const ES &es, const Eigen::VectorXd &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("visualizationField", [](const ES &es, const MXNd            &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("toDegree", [](const ES &es, const size_t degree) {
                  if (degree == 1) return toDegree<1>(es);
                  if (degree == 2) return toDegree<2>(es);
                  throw std::runtime_error("Only degree 1 and 2 are supported");
            }, py::arg("degree"), "Upgrade/downgrade the degree of the FEM discretization")
          .def("minDeformedEdgeLen", [](const ES &es) {
                BENCHMARK_SCOPED_TIMER_SECTION timer("minDeformedEdgeLen");
                Real result = std::numeric_limits<Real>::max();
                for (auto he : es.mesh().halfEdges()) {
                    result = std::min(result, (es.deformedPositions().row(he.tip().index()) -
                                               es.deformedPositions().row(he.tail().index())).norm());
                }
                return result;
              }, "Useful for detecting collapsed elements...")
          .def("deformedElementVolumes", &ES::deformedElementVolumes, "Numerical approximation of each element's volume in the deformed config.")
          .def("hessianBlockSparsityPattern", &ES::hessianBlockSparsityPattern, py::arg("val") = 0, py::arg("vmask") = VM::Defo)
         ;
        if constexpr (K == 3) {
            pyES.def("shrunkenTetVisualizationGeometry", [](const ES &obj, double tetShrinkFactor) {
                FEMMesh<Mesh::K, 1, EmbeddingSpace> visMesh(getF(obj.mesh()), obj.deformedPositions().topRows(obj.numVertices()));
                return getShrunkenTetVisualizationGeometry(visMesh, tetShrinkFactor);
            }, py::arg("tetShrinkFactor"))
            .def("shrunkenTetVisualizationField", [](const ES &obj, const Eigen::MatrixXd &f) {
                return getShrunkenTetVisualizationField(obj.mesh(), f);
            }, py::arg("f"))
            ;
        }


        using ESRE = ElasticSolidRotExtrap<K, Deg, EmbeddingSpace, Energy>;
        module.def("ElasticSolidRotExtrap", [](std::shared_ptr<Mesh> m, const Energy &e) { return std::make_shared<ESRE>(e, m); }, py::arg("mesh"), py::arg("energy"));

        py::class_<ESRE, EO, std::shared_ptr<ESRE>> pyESRE(detail_module, (name + "RotExtrap").c_str());

        using Method = typename ESRE::Method;
        py::enum_<Method>(pyESRE, "Method")
            .value("ElementExtrapolation", Method::ElementExtrapolation)
            .value("ModalWarping",         Method::ModalWarping        )
            ;

        pyESRE
            .def_property_readonly("elasticSolid", [](const ESRE &esre) -> const ES & { return esre.elasticSolid(); }, py::return_value_policy::reference_internal)
            .def_property_readonly("source_x",     &ESRE::source_x)
            .def_property("method", &ESRE::getMethod, &ESRE::setMethod)
            .def("visualizationGeometry", [](const ESRE &obj, double normalCreaseAngle) {
                  FEMMesh<Mesh::K, 1, EmbeddingSpace> visMesh(getF(obj.mesh()), obj.elasticSolid().deformedVertices());
                  return getVisualizationGeometry(visMesh, normalCreaseAngle);
               }, py::arg("normalCreaseAngle") = M_PI)
            .def("visualizationField", [](const ESRE &es, const Eigen::VectorXd &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
            .def("visualizationField", [](const ESRE &es, const MXNd            &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
            ;
    }
};

PYBIND11_MODULE(elastic_solid, m)
{
    py::module detail_module = m.def_submodule("detail");

    py::module::import("mesh");
    py::module::import("energy");
    py::module::import("sparse_matrices");
    py::module::import("py_newton_optimizer");
    py::module::import("loads");
    py::module::import("elastic_object");

    generateElasticSolidBindings(m, detail_module, ElasticSolidBinder());
}
