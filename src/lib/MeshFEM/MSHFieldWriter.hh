////////////////////////////////////////////////////////////////////////////////
// MSHFieldWriter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Write scalar/vector/matrix fields in MSH format for viewing with Gmsh.
//      For higher order FEM, two modes are allowed:
//          1) (m_linearSubsample = true) A piecewise linear mesh is output,
//             with higher degree per-node fields subsampled at the vertices.
//             This allows piecewise linear fields to be output regardless of
//             mesh degree.
//          2) (m_linearSubsample = false) A full-degree mesh including all
//             element nodes is output, requiring all per-node fields to be full
//             degree.
//      Currently, outputing a per-vertex field on a higher degree FEM mesh
//      is unsupported (the calling code must first manually create a higher
//      degree field that evaluates the linear field at all mesh nodes.)
//      field interpolating the linear
//
//      Also, to subsample the higher degree fields (case 1), we require the
//      vertex nodes to be a prefix of the full node list.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/26/2013 17:30:04
////////////////////////////////////////////////////////////////////////////////
#ifndef MSHFIELDWRITER_HH
#define MSHFIELDWRITER_HH
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>
#include <type_traits>

#include <MeshFEM/Fields.hh>
#include <MeshFEM/Flattening.hh>
#include <MeshFEM/Functions.hh>
#include <MeshFEM/MeshIO.hh>

class MSHFieldWriter {
protected:
    MSHFieldWriter() { } // Allow subclass MSHBoundaryFieldWriter to do all the
                         // construction work when it wants..
public:
    // Note: this constructor cannot be used in linear subsample mode (since we
    // don't know enough to distinguish nodes from vertices).
    MSHFieldWriter(const std::string &mshPath,
                   const std::vector<MeshIO::IOVertex>  &nodes,
                   const std::vector<MeshIO::IOElement> &elements,
                   MeshIO::MeshType meshType = MeshIO::MESH_GUESS,
                   bool binary = true)
        : m_linearSubsample(false),
          m_outStream(mshPath), m_numVertices(nodes.size()),
          m_numNodes(nodes.size()), m_numElements(elements.size()),
          m_binary(binary)
    {
        m_numOutputNodesPerElement.reserve(m_numElements);
        for (const auto &e : elements)
            m_numOutputNodesPerElement.push_back(e.size());
        if (!m_outStream.is_open()) {
            std::cout << "Failed to open output file '"
                      << mshPath << '\'' << std::endl;
        }
        else {
            MeshIO::MeshIO_MSH io;
            io.setBinary(binary);
            io.save(m_outStream, nodes, elements, meshType);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Construct a field writer for a mesh data structure.
    //  @param[in]  mshPath           msh output file path
    //  @param[in]  mesh              mesh to output
    //  @param[in]  linearSubsample   whether to always write piecewise linear
    //                                per-vertex fields regardless of mesh
    //                                degree (this is the default to save space)
    //  @param[in]  meshType          Type of mesh elements
    //  @param[in]  binary            Whether to use the binary MSH format.
    *///////////////////////////////////////////////////////////////////////////
    template<typename Mesh>
    MSHFieldWriter(const std::string &mshPath, const Mesh &mesh,
                   bool linearSubsample = true,
                   MeshIO::MeshType meshType = MeshIO::MESH_GUESS,
                   bool binary = true)
        : m_linearSubsample(linearSubsample),
          m_outStream(mshPath), m_numVertices(mesh.numVertices()),
          m_numNodes(mesh.numNodes()), m_numElements(mesh.numElements()),
          m_binary(binary)
    {
        if (!m_outStream.is_open()) {
            std::cout << "Failed to open output file '"
                      << mshPath << '\'' << std::endl;
        }
        else {
            std::vector<MeshIO::IOVertex>  outNodes;
            std::vector<MeshIO::IOElement> outElements;
            outElements.reserve(m_numElements);
            m_numOutputNodesPerElement.reserve(m_numElements);
            if (m_linearSubsample) {
                for (auto v : mesh.vertices())
                    outNodes.emplace_back(v.node()->p.template cast<double>().eval());
                for (auto e : mesh.elements()) {
                    outElements.emplace_back(e.numVertices());
                    m_numOutputNodesPerElement.push_back(e.numVertices());
                    for (size_t c = 0; c < e.numVertices(); ++c)
                        outElements.back()[c] = e.vertex(c).index();
                }
            }
            else  {
                for (auto n : mesh.nodes())
                    outNodes.emplace_back(n->p.template cast<double>().eval());
                for (auto e : mesh.elements()) {
                    outElements.emplace_back(e.numNodes());
                    m_numOutputNodesPerElement.push_back(e.numNodes());
                    for (size_t c = 0; c < e.numNodes(); ++c)
                        outElements.back()[c] = e.node(c).index();
                }
            }

            MeshIO::MeshIO_MSH io;
            io.setBinary(binary);
            io.save(m_outStream, outNodes, outElements, meshType);
        }
    }

    // General fields
    template<typename Field>
    void addField(const std::string &name, const Field &f,
                  DomainType type = DomainType::GUESS) {
        std::string sectionHeader;
        std::runtime_error invalidSize("Invalid field domain size.");
        std::runtime_error invalidDim("Invalid field dimension.");

        // We might be writing a subset of the domainSize() entries.
        size_t numEntries = 0;
        m_determineDomainTypeAndNumEntries(f.domainSize(), type, numEntries);

        if      (type == DomainType::PER_ELEMENT) sectionHeader = "ElementData";
        else if (type == DomainType::PER_NODE)    sectionHeader = "NodeData";
        else throw std::runtime_error("Unsupported DomainType");

        size_t dim = f.dim(), paddedDim = f.dim();
        switch (f.fieldType()) {
            case FIELD_SCALAR:
                if (dim != 1) throw invalidDim;
                break;
            case FIELD_VECTOR:
                // 2-vectors are padded to 3-vectors for GMSH compatibility.
                if (dim == 2) paddedDim = 3;
                if (paddedDim != 3) throw invalidDim;
                break;
            case FIELD_MATRIX:
                if ((f.N() != 2) && (f.N() != 3)) throw invalidDim;
                // for GMSH compatibility, 2x2 matrices are padded to 3x3,
                // which are output as a 9-vector in scanline
                paddedDim = 9;
                break;
            default:
                throw std::runtime_error("Invalid field type.");
        }

        m_outStream << '$' << sectionHeader << std::endl
                    << '1' << std::endl // One string tag: field name
                    << '"' << name << '"' << std::endl
                    << '0' << std::endl // No real tags
                    << '3' << std::endl // 3 Integer tags:
                    << '0' << std::endl // Time step 0 (ignored)
                    << paddedDim << std::endl // dimension
                    << numEntries << std::endl;
        for (size_t i = 1; i <= numEntries; ++i) {
            auto val = f(i - 1);
            if (m_binary) { int out = int(i); m_outStream.write((char *) &out, sizeof(int)); }
            else          m_outStream << i;
            if (f.fieldType() == FIELD_MATRIX) {
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        // Pad to 3x3
                        double value = (((k < f.N()) && (l < f.N())) ?
                                        val[flattenIndices(f.N(), k, l)] : 0);
                        if (m_binary) m_outStream.write((char *) &value, sizeof(double));
                        else          m_outStream << ' ' << value;
                    }
                }
            }
            else {
                for (size_t c = 0; c < paddedDim; ++c) {
                    double value = ((c < dim) ? val[c] : 0);
                    if (m_binary) m_outStream.write((char *) &value, sizeof(double));
                    else          m_outStream << ' ' << value;
                }
            }
            if (!m_binary)
                m_outStream << std::endl;
        }
        m_outStream << "$End" << sectionHeader << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Vector of interpolant suppport.
    ////////////////////////////////////////////////////////////////////////////
    // We need to distinguish scalar, vector, and symmetric vector types
    // Traits-like class for wrapping interpolant types.
    // Specializations of this class must provide:
    //    - constructor taking a value (needed for runtime Eigen vector
    //                                  dimension query)
    //    - component(value, i): function for indexing into (double) value.
    //    - paddedDim:           number of components to output; this could
    //                           differ from the underlying value's dimension
    //                           because of padding
    template<typename T, typename = void>
    struct InterpolantTypeWrapper;

    // Scalar (arithmetic) field type
    template<typename T>
    struct InterpolantTypeWrapper<T, typename std::enable_if<std::is_arithmetic<T>::value, void>::type> {
        InterpolantTypeWrapper(const T &) { }
        size_t dim = 1, paddedDim = 1;
        double component(const T &val, size_t i) const {
            (void) (i);
            assert(i < paddedDim);
            return val;
        }
    };

    // Vector (Eigen) field type
    template<int RowsAtCompileTime, int Options, int MaxRowsAtCompileTime, int MaxColsAtCompileTime>
    struct InterpolantTypeWrapper<Eigen::Matrix<Real, RowsAtCompileTime, 1, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime>, void> {
        using Vector = Eigen::Matrix<Real, RowsAtCompileTime, 1, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime>;
        InterpolantTypeWrapper(const Vector &v) {
            paddedDim = dim = v.rows();
            // 2-vectors are padded to 3-vectors for GMSH compatibility.
            if (dim == 2) paddedDim = 3;
            if (paddedDim != 3) throw std::runtime_error("Invalid vector field dimensions.");
        }
        double component(const Vector &v, size_t i) const {
            assert(i < paddedDim);
            return v[i];
        }
        size_t dim, paddedDim;
    };

    // Symmetric matrix type.
    template<class SMatrix>
    struct InterpolantTypeWrapper<SMatrix, typename std::enable_if<is_symmetric_matrix<SMatrix>::value, void>::type> {
        InterpolantTypeWrapper(const SMatrix &sm) {
            dim = sm.flatSize();
            N = sm.size();
            if (dim > 9) throw std::runtime_error("Invalid output matrix size");
        }

        double component(const SMatrix &sm, size_t i) const {
            if (sm.size() != N) throw std::runtime_error("Variable size output matrix field.");
            // Output in scanline order, padding to 3x3
            assert(i < paddedDim);
            size_t k = i / 3, l = i % 3;
            return ((k < N) && (l < N)) ? sm(k, l) : 0.0;
        }

        size_t N, dim, paddedDim = 9;
    };

    // Vector of interpolants. (ElementNode field only)
    template<typename _Interpolant, typename std::enable_if<is_interpolant<_Interpolant>::value, int>::type = 0>
    void addField(const std::string &name,
                  const std::vector<_Interpolant> &f,
                  DomainType type = DomainType::GUESS) {
        size_t numEntries = 0;
        type = m_determineDomainTypeAndNumEntries(f.size(), type, numEntries);
        if (type != DomainType::PER_ELEMENT)
            throw std::runtime_error("Vector-of-interpolants must be per-element.");

        InterpolantTypeWrapper<typename _Interpolant::value_type> wrapper(f.at(0)[0]);
        // InterpolantTypeWrapper<decltype(f.at(0)[0])> wrapper(f.at(0));

        m_outStream << "$ElementNodeData" << std::endl
                    << '1' << std::endl // One string tag: field name
                    << '"' << name << '"' << std::endl
                    << '0' << std::endl // No real tags
                    << '3' << std::endl // 3 Integer tags:
                    << '0' << std::endl // Time step 0 (ignored)
                    << wrapper.paddedDim << std::endl // dimension
                    << numEntries << std::endl;

        // Format: elem_idx  nodesPerElem values
        // there are nodesPerElem * dim wrapper.paddedDim values.
        for (size_t i = 1; i <= numEntries; ++i) {
            size_t numNodesPerElem = m_numOutputNodesPerElement.at(i - 1);
            if (m_binary) { int out[2] = {int(i), int(numNodesPerElem)}; m_outStream.write((char *) out, 2 * sizeof(int)); }
            else          { m_outStream << i << ' ' << numNodesPerElem; }
            const auto &val = f.at(i - 1);
            if (val.size() < numNodesPerElem)  // allow subsampling of higher-degree val
                throw std::runtime_error("Interpolant has too few nodes");
            for (size_t n = 0; n < numNodesPerElem; ++n) { // for each node
                const auto &nval = val[n];
                for (size_t c = 0; c < wrapper.paddedDim; ++c) {
                    double value = wrapper.component(nval, c);
                    if (m_binary) m_outStream.write((char *) &value, sizeof(double));
                    else          m_outStream << ' ' << value;
                }
            }
            if (!m_binary)
                m_outStream << std::endl;
        }

        m_outStream << "$EndElementNodeData" << std::endl;
    }

    size_t numVertices() const { return m_numVertices; }
    size_t numElements() const { return m_numElements; }

    // Type cast to bool checks if the output file is open and ready
    operator bool() const {
        return m_outStream.is_open();
    }

    ~MSHFieldWriter() {
        m_outStream.close();
    }

private:
    ////////////////////////////////////////////////////////////////////////////
    /*! Validate/guess domain's type based on its size.
    //  @param[in]    domainSize  used for guessing domain type.
    //  @param[inout] type        in: domain type to be validated (or PER_GUESS)
    //                            out: validated/guessed domain type
    //  @param[out]   numEntries  number of entries to be output
    //                            (possibly less than domainSize)
    *///////////////////////////////////////////////////////////////////////////
    DomainType m_determineDomainTypeAndNumEntries(size_t domainSize,
                                DomainType &type, size_t &numEntries) const {
        std::runtime_error invalidSize("Invalid field domain size.");
        if (type == DomainType::GUESS) {
            if (domainSize == m_numElements)
                type = DomainType::PER_ELEMENT;
            else if ((domainSize == m_numVertices) || (domainSize == m_numNodes))
                type = DomainType::PER_NODE;
            else throw invalidSize;
        }

        std::set<size_t> validSizes;
        if (type == DomainType::PER_ELEMENT) {
            numEntries = m_numElements;
            validSizes = { m_numElements };
        }
        else if (type == DomainType::PER_NODE) {
            numEntries = m_linearSubsample ? m_numVertices : m_numNodes;
            validSizes = { m_numNodes };
            if (m_linearSubsample) { validSizes.insert(m_numVertices); }
        }
        if (validSizes.find(domainSize) == validSizes.end()) throw invalidSize;
        return type;
    }

protected:
    bool m_linearSubsample;
    std::ofstream m_outStream;
    size_t m_numVertices, m_numNodes, m_numElements;
    // needed for validation/output of ElementNodeData fields
    std::vector<size_t> m_numOutputNodesPerElement;
    bool m_binary;
};

class MSHBoundaryFieldWriter : public MSHFieldWriter {
public:
    ////////////////////////////////////////////////////////////////////////////
    /*! Construct a field writer for ** the boundary of ** a mesh data structure
    //  @param[in]  mshPath           msh output file path
    //  @param[in]  mesh              mesh to output
    //  @param[in]  linearSubsample   whether to always write piecewise linear
    //                                per-vertex fields regardless of mesh
    //                                degree (this is the default to save space)
    //  @param[in]  meshType          Type of mesh elements
    //  @param[in]  binary            Whether to use the binary MSH format.
    *///////////////////////////////////////////////////////////////////////////
    template<typename Mesh>
    MSHBoundaryFieldWriter(const std::string &mshPath, const Mesh &mesh,
                   bool linearSubsample = true,
                   MeshIO::MeshType meshType = MeshIO::MESH_GUESS,
                   bool binary = true)
    {
        // Manually construct MSHFieldWriter's members.
        this->m_outStream.open(mshPath);
        this->m_linearSubsample = linearSubsample;
        this->m_numVertices     = mesh.numBoundaryVertices();
        this->m_numNodes        = mesh.numBoundaryNodes();
        this->m_numElements     = mesh.numBoundaryElements();
        this->m_numNodes        = mesh.numBoundaryNodes();
        this->m_binary          = binary;

        if (!m_outStream.is_open()) {
            std::cout << "Failed to open output file '"
                      << mshPath << '\'' << std::endl;
        }
        else {
            std::vector<MeshIO::IOVertex>  outNodes;
            std::vector<MeshIO::IOElement> outElements;
            outElements.reserve(m_numElements);
            m_numOutputNodesPerElement.reserve(m_numElements);
            if (m_linearSubsample) {
                for (auto v : mesh.boundaryVertices())
                    outNodes.emplace_back(v.volumeVertex().node()->p);
                for (auto e : mesh.boundaryElements()) {
                    outElements.emplace_back(e.numVertices());
                    m_numOutputNodesPerElement.push_back(e.numVertices());
                    for (size_t c = 0; c < e.numVertices(); ++c)
                        outElements.back()[c] = e.vertex(c).index();
                }
            }
            else  {
                for (auto n : mesh.boundaryNodes())
                    outNodes.emplace_back(n.volumeNode()->p);
                for (auto e : mesh.boundaryElements()) {
                    outElements.emplace_back(e.numNodes());
                    m_numOutputNodesPerElement.push_back(e.numNodes());
                    for (size_t c = 0; c < e.numNodes(); ++c)
                        outElements.back()[c] = e.node(c).index();
                }
            }

            MeshIO::MeshIO_MSH io;
            io.setBinary(binary);
            io.save(m_outStream, outNodes, outElements, meshType);
        }
    }
};

#endif // MSHFIELDWRITER_HH
