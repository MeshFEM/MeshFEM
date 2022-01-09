#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <MeshFEM/ElasticSolid.hh>
#include <MeshFEM/MassMatrix.hh>
#include <MeshFEM/EnergyDensities/LinearElasticEnergy.hh>
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/CorotatedLinearElasticity.hh>
#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/Utilities/NameMangling.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include "MeshEntities.hh"

#include "EquilibriumBinding.hh"
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
        using Vector = VectorND<N>;
        using Energy = typename ES::Energy;
        using MXNd   = Eigen::Matrix<Real, Eigen::Dynamic, N>;
        using Mesh = typename ES::Mesh;
        using EmbeddingSpace = typename Mesh::EmbeddingSpace;

        module.def("ElasticSolid", [](std::shared_ptr<Mesh> m, const Energy &e) { return std::make_shared<ES>(e, m); }, py::arg("mesh"), py::arg("energy"));

        const std::string name = getElasticSolidName<Energy, K, Deg, Vector>();
        py::class_<ES, std::shared_ptr<ES>> pyES(detail_module, name.c_str());
        pyES
          .def_property_readonly_static("dimension",   [](py::object /* self */) { return N; })
          .def_property_readonly_static("degree",      [](py::object /* self */) { return Deg; })
          .def_property_readonly_static("energy_name", [](py::object /* self */) { return getEnergyName<Energy>(); })
          .def("mesh",                      &ES::mesh)
          .def("numVars",                   &ES::numVars)
          .def("numElements",               &ES::numElements)
          .def("setIdentityDeformation",    &ES::setIdentityDeformation)
          .def("getVars",                   &ES::getVars)
          .def("setVars",                   &ES::setVars, py::arg("vars"))
          .def("setDeformedPositions",      &ES::setDeformedPositions)
          .def("applyRigidTransform",       &ES::applyRigidTransform, py::arg("R"), py::arg("t"))
          .def("prepareRigidMotionPins",    &ES::prepareRigidMotionPins)
          .def("filterRMPinArtifacts",      &ES::filterRMPinArtifacts, py::arg("pinVertices"))
          .def("energy",                    &ES::energy)
          .def("gradient",                  &ES::gradient)
          .def("hessian",                   [](const ES &es, bool projectionMask) { return es.hessian(projectionMask); }, py::arg("projectionMask") = false)
          .def("hessianSparsityPattern",    &ES::hessianSparsityPattern)
          .def("massMatrix",                &ES::massMatrix, py::arg("lumped") = false)
          .def("sobolevInnerProductMatrix", &ES::sobolevInnerProductMatrix, py::arg("Mscale") = 1.0)
          .def("getDeformedPositions",      &ES::deformedPositions)
          .def("getRestPositions",          &ES::restPositions)
          .def("getNodeDisplacements",      &ES::nodeDisplacements)
          .def("getEnergyDensity",          &ES::getEnergyDensity, py::arg("ei"), py::return_value_policy::reference)
          .def("greenStrain",               [](const ES &es, size_t ei) { return es.greenStrain(ei); }, py::arg("ei"))
          .def("greenStrain",               [](const ES &es, size_t ei, const typename ES::EvalPtN &baryCoords) { return es.greenStrain(ei, baryCoords); }, py::arg("ei"), py::arg("baryCoords"))
          .def("vertexGreenStrains",        &ES::vertexGreenStrains)
          .def("cauchyStress",              [](const ES &es, size_t ei) { return es.cauchyStress(ei); }, py::arg("ei"))
          .def("cauchyStress",              [](const ES &es, size_t ei, const typename ES::EvalPtN &baryCoords) { return es.cauchyStress(ei, baryCoords); }, py::arg("ei"), py::arg("baryCoords"))
          .def("vertexCauchyStresses",      &ES::vertexCauchyStresses)
          .def("surfaceStressLpNorm",       &ES::surfaceStressLpNorm, py::arg("p"))
          .def("visualizationGeometry", [](const ES &obj, double normalCreaseAngle) {
                FEMMesh<Mesh::K, 1, EmbeddingSpace> visMesh(getF(obj.mesh()), obj.deformedVertices());
                return getVisualizationGeometry(visMesh, normalCreaseAngle);
             }, py::arg("normalCreaseAngle") = M_PI)
          .def("visualizationField", [](const ES &es, const Eigen::VectorXd &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("visualizationField", [](const ES &es, const MXNd            &f) { return getVisualizationField(es.mesh(), f); }, "Convert a per-vertex or per-element field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
          .def("toDegree", [](const ES &es, const size_t degree) {
                  if (degree == 1) return toDegree<1>(es);
                  if (degree == 2) return toDegree<2>(es);
                  throw std::runtime_error("Only degree 1 and 2 are supported");
            }, py::arg("degree"), "Upgrade/downgrade the degree of the FEM discretization")
          .def("referenceConfigSampler",   &ES::referenceConfigSampler)
          .def("deformationSamplerMatrix", &ES::deformationSamplerMatrix)
         ;

        addComputeEquilibriumBinding<ES>(pyES, detail_module, name);

#if 0 // For debugging
        using SVP = SingleVertexOptProblem<ES>;
        py::class_<SVP>(detail_module, ("SingleVertexOptProblem" + name).c_str())
            .def("numVars",  [](const SVP &svp) { return svp.numVars(); })
            .def("getVars",  &SVP::getVars)
            .def("setVars",  &SVP::setVars)
            .def("energy",   &SVP::energy)
            .def("gradient", &SVP::gradient)
            .def("hessian",  &SVP::hessian)
            ;
        module.def("SingleVertexOptProblem", [](ES &es, size_t vi) {
                return std::make_unique<SVP>(es, vi);
            }, py::arg("es"), py::arg("vi"));
#endif
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

    generateElasticSolidBindings(m, detail_module, ElasticSolidBinder());
}
