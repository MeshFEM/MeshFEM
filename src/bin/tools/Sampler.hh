#ifndef SAMPLER_HH
#define SAMPLER_HH

#include <MeshFEM/EmbeddedElement.hh>
#include <MeshFEM/MeshIO.hh>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <MeshFEM/DenseCollisionGrid.hh>
#include <MeshFEM/Future.hh>

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
        e.embed(VectorND<3>(verts.at(nidx[0])),
                VectorND<3>(verts.at(nidx[1])),
                VectorND<3>(verts.at(nidx[2])),
                VectorND<3>(verts.at(nidx[3])));
    }
};

struct ElementSampler {
    struct Sample {
        Sample(size_t ei, const MeshIO::IOElement&ni, const Eigen::VectorXd &bc)
            : eidx(ei), nidx(ni), baryCoords(bc) { }
        size_t eidx;
        MeshIO::IOElement nidx;
        Eigen::VectorXd baryCoords;
    };

    template<size_t N>
    struct Sampler {
        using Pt = PointND<N>;
        using AESimplex = AffineEmbeddedSimplex<N, Pt>;

        Sampler(const std::vector<MeshIO::IOVertex> &vertices, const std::vector<MeshIO::IOElement> &elements)
            : m_vertices(vertices), m_elements(elements) {
            size_t numElems = m_elements.size();
            m_embeddedSimplices.resize(numElems);
            for (size_t i = 0; i < numElems; ++i)
                EmbedSimplexImpl<N>::run(m_embeddedSimplices[i], m_vertices, m_elements[i]);
        }

        Sample operator()(const PointND<N> &p) const {
            typename AESimplex::BaryCoords l;

            // Run the collision-grid-accelerated query if we can
            // First look for points inside a simplex.
            // If none are found, allow a small tolerance in the query.
            const double eps = 1e-7;
            if (m_collisionGrid) {
                std::vector<size_t> candidates = m_collisionGrid->enclosingBoxes(p);
                for (size_t i : candidates) {
                    if (m_embeddedSimplices[i].contains(p, l))
                        return Sample(i, m_elements.at(i), l);
                }
                for (size_t i : candidates) {
                    if (m_embeddedSimplices[i].contains(p, l, eps))
                        return Sample(i, m_elements.at(i), l);
                }
            }
            else {
                for (size_t i = 0; i < m_embeddedSimplices.size(); ++i) {
                    if (m_embeddedSimplices[i].contains(p, l))
                        return Sample(i, m_elements.at(i), l);
                }
                for (size_t i = 0; i < m_embeddedSimplices.size(); ++i) {
                    if (m_embeddedSimplices[i].contains(p, l, eps))
                        return Sample(i, m_elements.at(i), l);
                }
            }
            std::stringstream ss;
            ss << "Sample point outside domain: " << p;
            throw std::runtime_error(ss.str());
        }

        // Accelerate future sampler queries by building a bbox grid.
        // Should be created if many (hundreds) of queries are to be run.
        // Since this method doesn't (shouldn't) affect the sampler's
        // user-facing behavior (aside from speeding things up), it is
        // considered const, and the collision grid pointer is marked mutable.
        void accelerate() const {
            if (m_collisionGrid) return;
            m_collisionGrid = Future::make_unique<DenseCollisionGrid<N>>(100, BBox<Pt>(m_vertices));
            size_t numElems = m_elements.size();
            for (size_t i = 0; i < numElems; ++i) {
                // Add element's bounding box to the collision grid.
                m_collisionGrid->addBox(BBox<Pt>(m_vertices, m_elements[i]), i);
            }
        }

        Real volume(size_t i) const { return m_embeddedSimplices.at(i).volume(); }
    private:
        std::vector<AESimplex> m_embeddedSimplices;
        const std::vector<MeshIO::IOVertex>  &m_vertices;
        const std::vector<MeshIO::IOElement> &m_elements;
        mutable std::unique_ptr<DenseCollisionGrid<N>> m_collisionGrid;
    };
};

#endif /* end of include guard: SAMPLER_HH */
