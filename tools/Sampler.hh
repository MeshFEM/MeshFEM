#ifndef SAMPLER_HH
#define SAMPLER_HH

#include <EmbeddedElement.hh>
#include <MeshIO.hh>
#include <stdexcept>

template<size_t N>
struct EmbedSimplexImpl;

template<>
struct EmbedSimplexImpl<2> {
    template<class ESimplex, class Vertices, class NodeIndices>
    static void run(ESimplex &e,  const Vertices &verts, const NodeIndices &nidx) {
        e.embed(truncateFrom3D<VectorND<2>>(verts.at(nidx[0])),
                truncateFrom3D<VectorND<2>>(verts.at(nidx[1])),
                truncateFrom3D<VectorND<2>>(verts.at(nidx[2])));
    }
};

template<>
struct EmbedSimplexImpl<3> {
    template<class ESimplex, class Vertices, class NodeIndices>
    static void run(ESimplex &e,  const Vertices &verts, const NodeIndices &nidx) {
        e.embed(verts.at(nidx[0]),
                verts.at(nidx[1]),
                verts.at(nidx[2]),
                verts.at(nidx[3]));
    }
};

struct ElementSampler {
    struct Sample {
        size_t eidx;
        MeshIO::IOElement nidx;
        Eigen::VectorXd baryCoords;
    };

    template<size_t N>
    struct Sampler {
        using AESimplex = AffineEmbeddedSimplex<N, PointND<N>>;

        Sampler(const std::vector<MeshIO::IOVertex> &vertices, const std::vector<MeshIO::IOElement> &elements) : m_vertices(vertices), m_elements(elements) {
            size_t numElems = m_elements.size();
            m_embeddedSimplices.resize(numElems);
            for (size_t i = 0; i < numElems; ++i)
                EmbedSimplexImpl<N>::run(m_embeddedSimplices[i], m_vertices, m_elements[i]);
        }

        Sample operator()(const PointND<N> &p) const {
            typename AESimplex::BaryCoords l;
            for (size_t i = 0; i < m_embeddedSimplices.size(); ++i) {
                if (m_embeddedSimplices[i].contains(p, l)) {
                    Sample s;
                    s.eidx = i;
                    s.nidx = m_elements.at(i);
                    s.baryCoords = l;
                    return s;
                }
            }
            throw std::runtime_error("Sample point outside domain.");
        }

        Real volume(size_t i) const { return m_embeddedSimplices.at(i).volume(); }
    private:
        std::vector<AESimplex> m_embeddedSimplices;
        const std::vector<MeshIO::IOVertex>  &m_vertices;
        const std::vector<MeshIO::IOElement> &m_elements;
    };
};

#endif /* end of include guard: SAMPLER_HH */
