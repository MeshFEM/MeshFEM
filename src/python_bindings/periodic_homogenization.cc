#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include <Eigen/Dense>
#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/SymmetricMatrix.hh>
#include <MeshFEM/PeriodicHomogenization.hh>
#include <MeshFEM/OrthotropicHomogenization.hh>
#include <MeshFEM/LinearElasticity.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

template<typename Mesh>
using ETensor = ElasticityTensor<typename Mesh::Real, Mesh::EmbeddingDimension>;

template<typename Mesh>
struct HomogenizationResult {
    static constexpr size_t Dim = Mesh::EmbeddingDimension;
    using Real    = typename Mesh::Real;
    using VField  = Eigen::Matrix<Real, Eigen::Dynamic, Dim>;
    using SMField = SymmetricMatrixField<Real, Dim>;

    ETensor<Mesh>        Ch;
    std::vector<VField>  w_ij;
    std::vector<SMField> strain_w_ij;
};

template<size_t _N>
using HMG = LinearElasticity::HomogenousMaterialGetter<Materials::Constant>::template Getter<_N>;

template<typename _Mesh>
HomogenizationResult<_Mesh> runHomogenization(
        const _Mesh &mesh, const ETensor<_Mesh> &Cbase, bool orthotropicCell,
        const std::string &manualPeriodicVerticesFile, bool centerFluctuationDisplacements,
        bool ignorePeriodicMismatch) {
    using Real = typename _Mesh::Real;
    static constexpr size_t N = _Mesh::EmbeddingDimension;
    using LEMesh = LinearElasticity::Mesh<N, _Mesh::Deg, HMG>;
    HMG<N>::material.setTensor(Cbase);
    LinearElasticity::Simulator<LEMesh> sim(getF(mesh), getV(mesh));

    HomogenizationResult<_Mesh> result;
    std::vector<VectorField<Real, N>> w_ij;

    // Compute fluctuation displacements and homogenized elasticity tensor
    std::unique_ptr<PeriodicCondition<N>> pc;
    if (!manualPeriodicVerticesFile.empty())
        pc = Future::make_unique<PeriodicCondition<N>>(sim.mesh(), manualPeriodicVerticesFile);
    if (orthotropicCell) {
        auto systems = PeriodicHomogenization::Orthotropic::solveCellProblems(w_ij, sim);
        result.Ch = PeriodicHomogenization::Orthotropic::homogenizedElasticityTensorDisplacementForm(w_ij, sim);
    }
    else {
        PeriodicHomogenization::solveCellProblems(w_ij, sim, 1e-7, ignorePeriodicMismatch, std::move(pc));
        result.Ch = PeriodicHomogenization::homogenizedElasticityTensorDisplacementForm(w_ij, sim);
    }

    const size_t numCellProblems = w_ij.size();

    if (centerFluctuationDisplacements) {
        for (size_t i = 0; i < numCellProblems; ++i) {
            auto &w = w_ij[i];
            VectorND<N> total(VectorND<N>::Zero());
            for (size_t ii = 0; ii < w.domainSize(); ++ii) total += w(ii);
            total *= 1.0 / w.domainSize();
            for (size_t ii = 0; ii < w.domainSize(); ++ii) w(ii) -= total;
        }
    }

    // Convert to numpy-compatible output
    result.w_ij.resize(numCellProblems);
    for (size_t i = 0; i < numCellProblems; ++i) {
        const auto &w = w_ij[i];
        result.w_ij[i].resize(w.domainSize(), N);
        for (size_t ii = 0; ii < w.domainSize(); ++ii)
            result.w_ij[i].row(ii) = w(ii).transpose();
    }

    // Compute fluctuation strains
    result.strain_w_ij.resize(numCellProblems);
    for (size_t i = 0; i < numCellProblems; ++i)
        result.strain_w_ij[i] = sim.averageStrainField(w_ij[i]);

    return result;
}

template<class _Mesh, class HR, class SMValue>
std::tuple<typename HR::VField, typename HR::SMField>
getProbeResult(const _Mesh &mesh, const HR &homogenizationResult, const SMValue &macroStrain) {
    const HR &hr = homogenizationResult;
    const size_t numCellProblems = hr.w_ij.size();
    constexpr size_t N = SMValue::N;
    using Vec = VectorND<N>;

    typename HR::VField         w = macroStrain[0] * hr.w_ij[0];
    typename HR::SMField strain_w = macroStrain[0] * hr.strain_w_ij[0];

    for (size_t i = 1; i < numCellProblems; ++i) {
        typename HR::Real shearDoubler = (i < N) ? 1.0 : 2.0;
        w        += shearDoubler * macroStrain[i] * hr.w_ij[i];
        strain_w += shearDoubler * macroStrain[i] * hr.strain_w_ij[i];
    }

    // Remove rigid translation of fluctuation displacement relative to the
    // base cell (i.e. try to keep the fluctuation-displaced microstructure
    // "within" the base cell):
    // We need to ensure vertices on periodic boundary do not move off the
    // boundary. We enforce this in an average sense for each cell face by
    // translating so that the corresponding displacement component's average
    // over all vertices on the face is zero. (Note: this is different from
    // preventing the center of mass from moving.)
    {
        auto bbox = mesh.boundingBox();
        Vec translation(Vec::Zero());
        Vec numAveraged(Vec::Zero());

        for (auto bn : mesh.boundaryNodes()) {
            auto n = bn.volumeNode();
            for (size_t d = 0; d < N; ++d) {
                if (std::abs(n->p[d] - bbox.minCorner[d]) < 1e-9) {
                    translation[d] += w(n.index(), d);
                    numAveraged[d] += 1.0;
                }
            }
        }
        translation.array() /= numAveraged.array();
        w.rowwise() -= translation.transpose();
    }

    // Add in the linear term
    auto        u = w;
    auto strain_u = strain_w;
    for (auto n : mesh.nodes())
        u.row(n.index()) += macroStrain.contract(n->p);
    for (size_t i = 0; i < strain_u.domainSize(); ++i)
        strain_u(i) += macroStrain;

    return std::make_tuple(u, strain_u);
}

template<typename _Mesh>
void bindHomogenization(py::module &m, py::module &detail_module) {
    using Real = typename _Mesh::Real;
    static constexpr size_t N = _Mesh::EmbeddingDimension;
    using HR = HomogenizationResult<_Mesh>;
    using SMValue = SymmetricMatrixValue<Real, N>;

    py::class_<HR>(detail_module, ("HomogenizationResult" + NameMangler<_Mesh>::name()).c_str())
        .def_readonly("Ch",          &HR::Ch)
        .def_readonly("w_ij",        &HR::w_ij)
        .def_readonly("strain_w_ij", &HR::strain_w_ij)
        ;

    m.def("homogenize", runHomogenization<_Mesh>,
          py::arg("mesh"), py::arg("Cbase"), py::arg("orthotropicCell") = false, py::arg("manualPeriodicVerticesFile") = std::string(),
          py::arg("centerFluctuationDisplacements") = true, py::arg("ignorePeriodicMismatch") = false)
     .def("probe", getProbeResult<_Mesh, HR, SMValue>, py::arg("mesh"), py::arg("homogenizationResult"), py::arg("macroStrain"))
     .def("probe", [](const _Mesh &mesh, const ETensor<_Mesh> &Cbase, const SMValue &macroStrain,
                      bool orthotropicCell, const std::string &manualPeriodicVerticesFile,
                      bool ignorePeriodicMismatch) {
                auto hr = runHomogenization(mesh, Cbase, orthotropicCell, manualPeriodicVerticesFile, false,
                                            ignorePeriodicMismatch);
                return getProbeResult(mesh, hr, macroStrain);
            }, py::arg("mesh"), py::arg("Cbase"), py::arg("macroStrain"), py::arg("orthotropicCell") = false, py::arg("manualPeriodicVerticesFile") = std::string(),
               py::arg("ignorePeriodicMismatch") = false)
     ;
}

template<typename _Real>
void addBindings(py::module &m, py::module &detail_module) {
    using V3d = Eigen::Matrix<_Real, 3, 1>;
    using V2d = Eigen::Matrix<_Real, 2, 1>;

    bindHomogenization<FEMMesh<3, 1, V3d>>(m, detail_module); // linear    tet mesh in 3d
    bindHomogenization<FEMMesh<3, 2, V3d>>(m, detail_module); // quadratic tet mesh in 3d
    bindHomogenization<FEMMesh<2, 1, V2d>>(m, detail_module); // linear    tri mesh in 2d
    bindHomogenization<FEMMesh<2, 2, V2d>>(m, detail_module); // quadratic tri mesh in 2d
}

PYBIND11_MODULE(periodic_homogenization, m) {
    m.doc() = "Periodic Homogenization";

    py::module detail_module = m.def_submodule("detail");

    py::module::import("mesh");
    py::module::import("tensors");

    addBindings<double>(m, detail_module);
// #if MESHFEM_BIND_LONG_DOUBLE
//     addBindings<long double>(m, detail_module);
// #endif
}
