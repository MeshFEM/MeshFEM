#include "MSHFieldWriter_bindings.hh"
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/Future.hh>

template<size_t N>
void bindMSHFieldParserDimSpecific(py::module &m) {
    using MFP = MSHFieldParser<N>;

    py::class_<MFP>(m, ("MSHFieldParser" + std::to_string(N)).c_str())
        .def(py::init<const std::string &, bool>(), py::arg("mshPath"), py::arg("permitDimMismatch") = true)
        .def("vertices", [](const MFP &mfp) { return getV(mfp.vertices()); })
        .def("elements", [](const MFP &mfp) { return getF(mfp.elements()); })
        .def("meshDegree",    &MFP::meshDegree)
        .def("meshDimension", &MFP::meshDimension)
        .def("numElements",   &MFP::numElements)
        .def("numVertices",   &MFP::numVertices)

        .def(         "vectorField", [](const MFP &mfp, const std::string &name, DomainType dtype = DomainType::ANY) { auto  vf = mfp.         vectorField(name, dtype); return  vf.data().transpose().eval(); }, py::arg("name"), py::arg("domainType") = DomainType::ANY)
        .def(         "scalarField", [](const MFP &mfp, const std::string &name, DomainType dtype = DomainType::ANY) { auto  sf = mfp.         scalarField(name, dtype); return  sf.values();                  }, py::arg("name"), py::arg("domainType") = DomainType::ANY)
        .def("symmetricMatrixField", [](const MFP &mfp, const std::string &name, DomainType dtype = DomainType::ANY) { auto smf = mfp.symmetricMatrixField(name, dtype); return smf.data().transpose().eval(); }, py::arg("name"), py::arg("domainType") = DomainType::ANY)


        .def(                    "vectorFieldNames", &MFP::                    vectorFieldNames, py::arg("domainType") = DomainType::ANY)
        .def(                    "scalarFieldNames", &MFP::                    scalarFieldNames, py::arg("domainType") = DomainType::ANY)
        .def(           "symmetricMatrixFieldNames", &MFP::           symmetricMatrixFieldNames, py::arg("domainType") = DomainType::ANY)
        .def(         "vectorInterpolantFieldNames", &MFP::         vectorInterpolantFieldNames, py::arg("domainType") = DomainType::ANY)
        .def(         "scalarInterpolantFieldNames", &MFP::         scalarInterpolantFieldNames, py::arg("domainType") = DomainType::ANY)
        .def("symmetricMatrixInterpolantFieldNames", &MFP::symmetricMatrixInterpolantFieldNames, py::arg("domainType") = DomainType::ANY)
        ;

}

py::object mshFieldParserFactory(const std::string &path, bool permitDimMismatch) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    std::ifstream mshFile(path);
    if (!mshFile.is_open()) throw std::runtime_error(std::string("Couldn't open input file ") + path);

    auto mio = dynamic_cast<MeshIO::MeshIO_MSH *>(getMeshIO(MeshIO::FMT_MSH));
    auto mtype = mio->load(mshFile, vertices, elements, MeshIO::MESH_GUESS);
    const size_t elem_size = elements.at(0).size();
    if (mtype == MeshIO::MESH_TRI) {
        return py::cast(new MSHFieldParser<2>(mshFile, mtype, std::move(elements), std::move(vertices), mio->binary(), permitDimMismatch),
                        py::return_value_policy::take_ownership);
    }
    if (mtype == MeshIO::MESH_TET) {
        return py::cast(new MSHFieldParser<3>(mshFile, mtype, std::move(elements), std::move(vertices), mio->binary(), permitDimMismatch),
                        py::return_value_policy::take_ownership);
    }
    throw std::runtime_error("Unexpected element size " + std::to_string(elem_size));
}

void bindMSHFieldParser(py::module &m) {
    bindMSHFieldParserDimSpecific<2>(m);
    bindMSHFieldParserDimSpecific<3>(m);

    m.def("MSHFieldParser", &mshFieldParserFactory, py::arg("path"), py::arg("permitDimMismatch") = true);
}
