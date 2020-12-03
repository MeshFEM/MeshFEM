////////////////////////////////////////////////////////////////////////////////
// Traction.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Force per unit area (3D) or length (2D) applied to the boundary of an
//  elastic object, represented as a piecewise constant vector field on the
//  boundary elements.
//  For now, force densites are assumed to be expressed in terms of the
//  *undeformed* area rather than the deformed area so that the total force
//  applied to a boundary element won't grow or shrink as the element deforms.
//  Also, the force directions are held fixed throughout the deformation
//  (rather than rotating to track the deforming boundary).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/01/2020 22:37:13
////////////////////////////////////////////////////////////////////////////////
#ifndef TRACTION_HH
#define TRACTION_HH

namespace Loads {

template<class Object>
struct Traction : public Load<Object::N, typename Object::Real> {
    static constexpr size_t N = Object::N;
    static constexpr size_t K   = Object::K;
    static constexpr size_t Deg = Object::Deg;

    using Real = typename Object::Real;
    using VXd  = typename Object::VXd;
    using MXNd = Eigen::Matrix<Real, Eigen::Dynamic, N>;

    Traction(std::weak_ptr<const Object> obj)
        : m_obj(obj) {
        m_boundaryTractions.setZero(getObj().mesh().numBoundaryElements(), N);
        m_updateCache();
        m_callbackID = getObj().registerRestConfigUpdateCallback([this]() { m_updateCache(); });
    }

    virtual Real energy() const override {
        return m_grad.dot(getObj().getVars());
    }

    // Derivative with respect to deformed configuration
    virtual VXd grad_x() const override {
        return m_grad;
    }

    // Derivative with respect to rest configuration (for shape optimization)
    virtual VXd grad_X() const override {
        // Do we really want the total force to decrease when the boundary shrinks??
        throw std::runtime_error("Unimplemented");
    }

    // Traction's potential is linear with respect to the deformed state
    virtual void hessian(SuiteSparseMatrix& /* H */, bool /* projectionMask */ = true) const override { }
    virtual SuiteSparseMatrix hessianSparsityPattern(Real /* val */ = 0.0) const override {
        const size_t nv = getObj().numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
        return SuiteSparseMatrix(Hsp);
    }

    void setBoundaryTractions(Eigen::Ref<const MXNd> val) { m_boundaryTractions = val; m_updateCache(); }
    const MXNd &getBoundaryTractions() const { return m_boundaryTractions; }

    virtual ~Traction() {
        if (auto o = m_obj.lock())
            o->deregisterRestConfigUpdateCallback(m_callbackID);
    }
private:
    std::weak_ptr<const Object> m_obj;
    int m_callbackID;
    MXNd m_boundaryTractions;
    VXd m_grad;

    void m_updateCache() {
        m_grad.setZero(getObj().numVars());
        const auto &m = getObj().mesh();
        typename Object::Mesh::ElementData::Phis phiIntegrals;
        auto integratedPhis = integratedShapeFunctions<Deg, K - 1>();
        for (const auto be : m.boundaryElements()) {
            for (const auto bn : be.nodes()) {
                m_grad.template segment<N>(N * bn.volumeNode().index()) -=
                    m_boundaryTractions.row(be.index()) * (integratedPhis[bn.localIndex()] * be->volume());
            }
        }
    }

    const Object &getObj() const {
        if (auto o = m_obj.lock()) return *o;
        throw std::runtime_error("Elastic object was destroyed");
    }
};

}

#endif /* end of include guard: TRACTION_HH */
