#include "MSHFieldWriter_bindings.hh"
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/Future.hh>

void bindMSHFieldWriter(py::module &m) {
    py::class_<MSHFieldWriter> field_writer(m, "MSHFieldWriter");

    py::enum_<DomainType>(field_writer, "DomainType")
        .value("PER_ELEMENT", DomainType::PER_ELEMENT)
        .value("PER_NODE",    DomainType::PER_NODE)
        .value("GUESS",       DomainType::GUESS)
        .value("ANY",         DomainType::ANY)
        .value("UNKNOWN",     DomainType::UNKNOWN)
        ;

    field_writer
        .def(py::init([](const std::string &path,
                    const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, bool binary) {
                    auto mio = getMeshIO(V, F);
                    return Future::make_unique<MSHFieldWriter>(path, mio.first, mio.second, MeshIO::MESH_GUESS, binary);
                }), py::arg("path"), py::arg("V"), py::arg("F"), py::arg("binary") = true)
        .def("addField", [](MSHFieldWriter &writer, const std::string &name, const Eigen::MatrixXd &field, DomainType dtype) {
                    if (field.cols() == 1) {
                        ScalarField<double> sf(field);
                        writer.addField(name, sf, dtype);
                    }
                    else if (field.cols() == 2) {
                        VectorField<double, 2> vf;
                        size_t n = field.rows();
                        vf.resizeDomain(n);
                        for (size_t i = 0; i < n; ++i)
                            vf(i) = field.row(i).transpose();
                        writer.addField(name, vf, dtype);
                    }
                    else {
                        VectorField<double, 3> vf;
                        size_t n = field.rows();
                        vf.resizeDomain(n);
                        for (size_t i = 0; i < n; ++i)
                            vf(i) = field.row(i).transpose();
                        writer.addField(name, vf, dtype);
                    }
                }, py::arg("name"), py::arg("field"), py::arg("dtype") = DomainType::GUESS)
        ;
}
