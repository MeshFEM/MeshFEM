#include "ElasticSheetBinding.hh"
#include <MeshFEM/Elements/DihedralAngle.hh>

using namespace MeshFEM;

PYBIND11_MODULE(elastic_sheet, m)
{
    py::module detail_module = m.def_submodule("detail");
    py::module::import("mesh");
    py::module::import("energy");
    py::module::import("sparse_matrices");
    py::module::import("py_newton_optimizer");
    py::module::import("loads");
    py::module::import("elastic_object");

    generateElasticSheetBindings(m, detail_module, ElasticSheetBinder());

    // // Standalone binding of PlateBendingElement for validation.
    // using PBE = PlateBending<double, AngleFunctionSin>;
    // using CPos = typename PBE::CornerPositions;
    // using V3d = Eigen::Vector3d;
    // auto elementData = [](const CPos &x) {
    //     using LEE = LinearlyEmbeddedElement<Simplex::Triangle, 1, V3d>;
    //     LEE e;
    //     e.embed(x.row(0).transpose(), x.row(1).transpose(), x.row(2).transpose());
    //     return elements::EmbeddedMembraneElementData<LEE, LEE>(e);
    // };
    // py::class_<PBE>(m, "PlateBendingElement")
    //     .def(py::init<double>(), py::arg("thickness"))
    //     .def("energy", [&elementData](const PBE &e, const ET &C, const CPos &X, const CPos &x, const Eigen::Vector3d &gamma) {
    //                 const auto ref_edata = elementData(X);
    //                 return e.energy(C, e.computeII(x, gamma, ref_edata), Eigen::Matrix2d::Zero(), ref_edata);
    //             }, py::arg("C"), py::arg("X"), py::arg("x"), py::arg("gamma"))
    //     .def("gradient", [&elementData](const PBE &e, const ET &C, const CPos &X, const CPos &x, const Eigen::Vector3d &gamma) {
    //                 const auto ref_edata = elementData(X);
    //                 return e.gradient(C, x, gamma, e.computeII(x, gamma, ref_edata), Eigen::Matrix2d::Zero(), ref_edata);
    //             }, py::arg("C"), py::arg("X"), py::arg("x"), py::arg("gamma"))
    //     .def("hessian", [&elementData](const PBE &e, const ET &C, const CPos &X, const CPos &x, const Eigen::Vector3d &gamma) {
    //                 const auto ref_edata = elementData(X);
    //                 return e.hessian(C, x, gamma, e.computeII(x, gamma, ref_edata), Eigen::Matrix2d::Zero(), ref_edata);
    //             }, py::arg("C"), py::arg("X"), py::arg("x"), py::arg("gamma"))
    // ;

    // Standalone binding of DihedralAngle for validation.
    using DA = elements::DihedralAngle<double>;
    py::class_<DA>(m, "DihedralAngle")
        .def(py::init<>())
        .def("configure", &DA::configure, py::arg("stencilPts"))
        .def("value",    &DA::value)
        .def("gradient", &DA::gradient)
        .def("hessian",  &DA::hessian)
    ;
}
