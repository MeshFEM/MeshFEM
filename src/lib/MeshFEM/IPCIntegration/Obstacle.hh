////////////////////////////////////////////////////////////////////////////////
// Obstacle.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//
//  Obstacle object used in multi-object collision
//
//  Author:  Haleh Mohammadian (halehOssadat), haleh.mohammadian@gmail.com
//  Created: 09/14/2023 16:11:15
*///////////////////////////////////////////////////////////////////////////////
#ifndef OBSTACLE_HH
#define OBSTACLE_HH

#include <Eigen/Dense>
#include <functional>


#include <MeshFEMCore/Types.hh>
#include <MeshFEM/Geometry.hh>

namespace MeshFEM {

struct Obstacle {
    using MXd = Eigen::MatrixXd;
    using MXi = Eigen::MatrixXi;
    using VXd = Eigen::VectorXd;
    using xFunction = std::function<MXd(double)>; // Function for getting obstacle vertex positions at a given time.

    using VMaxd = VecMaxN_T<double, 3>;

    Obstacle(const MXd &V = MXd(), const MXi &F = MXi(), const MXi &E = MXi(), const xFunction &xFUNC = xFunction())
        : m_x0(V), m_f(F), m_e(E), m_bbox(V), xFunc(xFUNC)
    {
        assert(F.cols() <= 3);
        m_x = m_x0;
        m_force = VMaxd::Zero(m_x.cols());
    }

    size_t numVertices() const { return m_x.rows(); }
    size_t numFaces() const { return m_f.rows(); }
    size_t numEdges() const { return m_e.rows(); }
    size_t dimension() const { return m_x.cols(); }

    const MXd &getVertices() const { return m_x; }
    const MXi &getFaces()    const { return m_f; }
    const MXi &getEdges()    const { return m_e; }
    const VMaxd &getForce()  const { return m_force; }

    BBox<VMaxd> getBBox() { return m_bbox; }

    void setVertices(const MXd &x){ m_x = x; }
    void setForce(const Eigen::Ref<const VXd> &f) {
        if (f.size() != m_x.size()) throw std::runtime_error("Unexpected force vector size.");
        m_force = Eigen::Map<const Eigen::MatrixXd>(f.data(), m_x.rows(), m_x.cols()).colwise().sum();
    }
    
    // Update the position of obstacle m_x as a function of time; m_x = xFunction(t).
    // This function is determined by the user
    void updatePositionForTime(double t) {
        m_x = xFunc(t);
    }

private:
    MXd m_x, m_x0; // Vertex positions and initial positions
    MXi m_f, m_e; // Faces and edges of triangular mesh
    BBox<VMaxd> m_bbox;
    const xFunction xFunc; // Obstacle position as a function of time for moving obstacle in time
    VMaxd m_force;
};

} // namespace MeshFEM

#endif
