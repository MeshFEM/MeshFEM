#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <Eigen/Dense>
#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/Fields.hh>
#include <MeshFEM/SymmetricMatrix.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/VonMises.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

template<typename _Real, size_t N>
void bindTensors(py::module& module, py::module& detail_module) {
    using ETensor = ElasticityTensor    <_Real, N>;
    using SMValue = SymmetricMatrixValue<_Real, N>;
    using SMF     = SymmetricMatrixField<_Real, N>;

    auto py_et = py::class_<ETensor>(module, NameMangler<ETensor>::name().c_str())
        .def(py::init<>())
        .def(py::init([](const std::string& material_file) { return Materials::Constant<N>(material_file).getTensor(); }), py::arg("material_file"))
        .def(py::init<_Real, _Real>(), py::arg("E"), py::arg("nu"))
        .def("setIsotropic", &ETensor::setIsotropic, py::arg("E"), py::arg("nu"))
        .def("setIdentity",  &ETensor::setIdentity)

        .def("getOrthotropicParameters", py::overload_cast<>(&ETensor::getOrthotropicParameters, py::const_))
        .def("anisotropy", &ETensor::anisotropy)

        .def("__call__", [](const ETensor &E, size_t i, size_t j, size_t k, size_t l) {
                    if ((i >= N) || (j >= N) || (k >= N) || (l >= N))
                        throw std::runtime_error("Index out of bounds");
                    return E(i, j, k, l);
                })
        .def_property_readonly("D", [](const ETensor &E) {
                    typename ETensor::DType D;
                    for (int i = 0; i < D.rows(); ++i) {
                        for (int j = 0; j < D.cols(); ++j) {
                            D(i, j) = E.D(i, j);
                        }
                    }
                    return D;
                })
        .def("doubleContract", [](const ETensor &E, const SMValue &smat) { return E.doubleContract(smat); }, py::arg("smat"))
        .def("doubleContract", [](const ETensor &E, const SMF     & smf) { 
                    SMF result(smf.domainSize());
                    for (size_t i = 0; i < smf.domainSize(); ++i)
                        result(i) = E.doubleContract(smf(i));
                    return result;
                }, py::arg("smat"))
        // .def("doubleContract",    [](const ETensor &E, const ETensor &Eother) { return E.doubleContract(Eother);    }, py::arg("E")) // this produces a non-major-symmetric result, which we haven't bound yet
        .def("quadrupleContract", [](const ETensor &E, const ETensor &Eother) { return E.quadrupleContract(Eother); }, py::arg("E"))

        .def("computeEigenstrains", &ETensor::computeEigenstrains)

        .def("inverse",         &ETensor::inverse)
        .def("pseudoinverse",   &ETensor::inverse)
        .def("frobeniusNormSq", &ETensor::frobeniusNormSq)
        .def("transform",       &ETensor::transform, py::arg("R"), "Apply a *orthogonal* change of coordinates to this tensor")
        .def("__sub__", [](const ETensor &E, const ETensor &Eother) { return E - Eother; })
        .def("__repr__", [](const ETensor &E) {
                std::stringstream ss;
                ss << N << "D elasticity tensor with orthotropic moduli: ";
                E.printOrthotropic(ss);
                return ss.str(); })
        ;

    if (N == 3) {
        py_et.def("setOrthotropic",
            &ETensor::setOrthotropic3D,
            py::arg("Ex"),   py::arg("Ey"),   py::arg("Ez"),
            py::arg("nuYX"), py::arg("nuZX"), py::arg("nuZY"),
            py::arg("muYZ"), py::arg("myZX"), py::arg("muXY"));
    }

    if (N == 2) {
        py_et.def("setOrthotropic",
            &ETensor::setOrthotropic2D,
            py::arg("Ex"),   py::arg("Ey"),
            py::arg("nuYX"), py::arg("muXY"));
    }

    py::class_<SMValue>(detail_module, NameMangler<SMValue>::name().c_str())
        .def("__call__", [](const SMValue &sm, size_t i, size_t j) { if ((i >= N) || (j >= N)) throw std::runtime_error("Index out of bounds"); return sm(i, j); }, py::arg("i"), py::arg("j"))
        .def("toMatrix", [](const SMValue &sm) { return sm.toMatrix(); })
        .def("eigenvalues",        &SMValue::eigenvalues)
        .def("eigenDecomposition", &SMValue::eigenDecomposition)
        ;

    py::class_<SMF>(detail_module, ("SymmetricMatrixField" + std::to_string(N) + "D" + floatingPointTypeSuffix<_Real>()).c_str())
        .def("vonMises", [](const SMF &smf) {
                SMF smf_vm = vonMises(smf);
                Eigen::VectorXd result(smf_vm.domainSize());
                for (size_t i = 0; i < smf_vm.domainSize(); ++i)
                    result[i] = std::sqrt(smf_vm(i).frobeniusNormSq());
                return result;
            })
        .def("eigendecomposition", [](const SMF &smf) {
                std::vector<SMEigenDecompositionType<_Real, N>> result;
                result.reserve(smf.domainSize());
                for (size_t i = 0; i < smf.domainSize(); ++i)
                    result.push_back(smf(i).eigenDecomposition());
                return result;
            })
        .def("__call__", [](const SMF &smf, size_t i) { if (i >= smf.domainSize()) throw std::runtime_error("Index out of bounds."); return SMValue(smf(i)); })
        ;

    module.def("SymmetricMatrix", [](const Eigen::Matrix<_Real, flatLen(N), 1> &flatValues) { return SMValue(flatValues); }, py::arg("flatValues"));
    module.def("SymmetricMatrix", [](const Eigen::Matrix<_Real, N, N>          &mat)        { return SMValue(       mat); }, py::arg("mat"));
}

template<typename _Real>
void addBindings(py::module &m) {
    py::module detail_module = m.def_submodule("detail");
    py::class_<ETensorEigenDecomposition>(detail_module, "ETensorEigenDecomposition")
        .def_readonly("eigenstrains", &ETensorEigenDecomposition::strains) // flattened symmetric matrix field
        .def_readonly("eigenvalues",  &ETensorEigenDecomposition::lambdas)
        ;
    bindTensors<_Real, 2>(m, detail_module);
    bindTensors<_Real, 3>(m, detail_module);
}

PYBIND11_MODULE(tensors, m) {
    m.doc() = "Tensors and tensor fields used for elasticity simulations";

    addBindings<double>(m);
}
