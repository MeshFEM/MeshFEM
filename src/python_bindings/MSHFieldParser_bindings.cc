#include "MSHFieldWriter_bindings.hh"
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/Future.hh>

template<size_t N>
void bindMSHFieldParserDimSpecific(py::module &m) {
    using MFP = MSHFieldParser<N>;

    py::class_<MFP>(m, ("MSHFieldParser" + std::to_string(N)).c_str())
        .def(py::init<const std::string &, bool>(), py::arg("mshPath"), py::arg("permitDimMismatch") = false)
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

void bindMSHFieldParser(py::module &m) {
    bindMSHFieldParserDimSpecific<2>(m);
    bindMSHFieldParserDimSpecific<3>(m);
}
