////////////////////////////////////////////////////////////////////////////////
// MeshConnectivity.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides abstract, dynamic access to connectivity information from
//      TetMesh/TriMesh.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/22/2016 22:46:13
////////////////////////////////////////////////////////////////////////////////
#ifndef MESHCONNECTIVITY_HH
#define MESHCONNECTIVITY_HH

struct MeshConnectivity {
    virtual ~MeshConnectivity() = default;

    // cells sharing faces with cell ei
    virtual size_t numElemNeighbors(size_t ei) const = 0;
    virtual size_t elemNeighbor(size_t ei, size_t j) const = 0;
};

template<class MeshDS>
struct MeshConnectivityImpl : public MeshConnectivity {
    MeshConnectivityImpl(const MeshDS &mesh) : m_mesh(mesh) { }

    size_t numElemNeighbors(size_t ei) const {
        assert(ei < m_mesh.numSimplices());
        return m_mesh.simplex(ei).numNeighbors();
    }

    size_t elemNeighbor(size_t ei, size_t j) const {
        assert(j < numElemNeighbors(ei));
        return m_mesh.simplex(ei).neighbor(j).index();
    }

private:
    const MeshDS &m_mesh;
};

#endif /* end of include guard: MESHCONNECTIVITY_HH */
