#include <cmath>
#include <cstdlib>
#include <string>
#include <sstream>

#include <Eigen/Dense>

template<typename Vector>
class CSGTree<Vector>::CSGNode
{
public:
    typedef typename Vector::Scalar Real;

    CSGNode(const std::string &name, CSGNode *parent = NULL)
        : m_name(name), m_hasName(true), m_parent(parent) { }

    CSGNode(CSGNode *parent = NULL) : m_hasName(false), m_parent(parent) { }

    CSGNode *parent() const { return m_parent; }
    void setParent(CSGNode *p) { m_parent = p; }
    std::string name() const { return m_hasName ? m_name : defaultName(); }
    virtual std::string defaultName() const { return "Default"; }

    virtual int indexOfChild(const CSGNode *) const { return -1; }
    virtual int numChildren() const { return 0; }
    virtual CSGTree<Vector>::CSGNode *child(size_t i) { return NULL; }
    virtual const CSGTree<Vector>::CSGNode *child(size_t i) const { return NULL; }

    virtual BBox_t boundingBox() const { return BBox_t(); }

    virtual void applyTranslation(const Vector &t) {
        for (int i = 0; i < numChildren(); ++i)
            child(i)->applyTranslation(t);
    }

        
    virtual bool isInside(const Vector &p) const = 0;
    virtual ~CSGNode() { }

protected:
    bool m_hasName;
    std::string m_name;
    CSGNode *m_parent;
};

template<typename Vector>
class CSGTree<Vector>::CSGPrimitive : public CSGTree<Vector>::CSGNode
{
public:
    typedef typename Vector::Scalar Real;

    CSGPrimitive(Vector center, Vector dimensions, Real rot = 0.0)
        : m_c(center), m_dim(dimensions), m_rot_inv(0) {
        setRotation(rot);
    }

    virtual bool isInside(const Vector &p) const = 0;
    virtual ~CSGPrimitive() { }

    Real getRotation() const { return -m_deg(m_rot_inv.angle()); }
    void setRotation(Real r) {
        m_rot_inv.angle() = -m_rad(r);
    }

    Vector toLocalCoords(const Vector &p) const {
        return m_rot_inv * (p - m_c);
    }

    BBox_t boundingBox() const {
        Eigen::Rotation2D<Real> rot = m_rot_inv.inverse();
        int n = m_dim.size();
        int nCorners = 1 << n;
        std::vector<Vector> corners(nCorners);
        for (int i = 0; i < nCorners; ++i) {
            corners[i] = Vector::Zero();
            for (int j = 0; j < n; ++j) {
                Real sign = ((i & (1 << j))) ? 1.0 : -1.0;
                corners[i][j] += sign * .5 * m_dim[j];
            }

            // Apply rotation and offset
            corners[i] = rot * corners[i] + m_c;
        }

        BBox_t b(corners[0], corners[0]);
        for (int i = 1; i < nCorners; ++i)
            b.unionBox(BBox_t(corners[i], corners[i]));

        return b;
    }

    void applyTranslation(const Vector &t) {
        m_c += t;
    }

protected:
    Vector m_c, m_dim;
private:
    Eigen::Rotation2D<Real> m_rot_inv;

    Real m_deg(Real angle) const { return 180.0 * (angle / M_PI); }
    Real m_rad(Real angle) const { return M_PI  * (angle / 180.0); }
};

template<typename Vector>
class CSGTree<Vector>::CSGBoolNode : public CSGTree<Vector>::CSGNode
{
public:
    CSGBoolNode(CSGOperation op, CSGTree<Vector>::CSGNode *left,
                CSGTree<Vector>::CSGNode *right)
        : m_op(op), m_left(left), m_right(right)
    {
        m_left->setParent(this);
        m_right->setParent(this);
    }

    std::string defaultName() const {
        switch(m_op) {
            case INTERSECT:
                return "Intersect";
            case UNION:
                return "Union";
            case SUBTRACT:
                return "Subtract";
            default:
                return "Invalid operation";
        }
    }

    bool isInside(const Vector &p) const {
        switch(m_op) {
            case INTERSECT:
                return m_left->isInside(p) && m_right->isInside(p);
            case UNION:
                return m_left->isInside(p) || m_right->isInside(p);
            case SUBTRACT:
                return (m_left->isInside(p) && !m_right->isInside(p));
            default:
                assert(false);
        }
        return false;
    }

    BBox_t boundingBox() const {
        BBox_t b = m_left->boundingBox();
        switch(m_op) {
            case INTERSECT:
                b.intersectBox(m_right->boundingBox());
                break;
            case UNION:
                b.unionBox(m_right->boundingBox());
                break;
            case SUBTRACT:
                // The entire left bounding box *could* be unaffected...
                break;
            default:
                assert(false);
        }

        return b;
    }

    int indexOfChild(const CSGNode *c) const {
        if (c == m_left)
            return 0;
        if (c == m_right)
            return 1;
        return -1;
    }

    int numChildren() const { return 2; }

    CSGTree<Vector>::CSGNode *child(size_t i) {
        if (i == 0)
            return m_left;
        if (i == 1)
            return m_right;
        assert(false);
    }

    const CSGTree<Vector>::CSGNode *child(size_t i) const {
        if (i == 0)
            return m_left;
        if (i == 1)
            return m_right;
        assert(false);
    }

    void swapChildren() {
        std::swap(m_left, m_right);
    }

    ~CSGBoolNode() {
        delete m_left;
        delete m_right;
    }

private:
    CSGOperation m_op;
    CSGTree<Vector>::CSGNode *m_left, *m_right;
};

template<typename Vector>
class CSGTree<Vector>::CSGRectangleNode : public CSGTree<Vector>::CSGPrimitive
{
    typedef typename Vector::Scalar Real;
public:
    CSGRectangleNode(const Vector &center, const Vector &dimensions, Real rot = 0)
        : CSGPrimitive(center, dimensions, rot)
    { }

    std::string defaultName() const {
        return "Rectangle";
    }

    bool isInside(const Vector &p) const {
        Vector l = this->toLocalCoords(p);
        return (l.cwiseAbs().array() <= (.5 * this->m_dim).array()).all();
    }

    ~CSGRectangleNode() { }
};


template<typename Vector>
class CSGTree<Vector>::CSGEllipseNode : public CSGTree<Vector>::CSGPrimitive
{
    typedef typename Vector::Scalar Real;
public:
    CSGEllipseNode(Vector center, const Vector &dimensions, Real rot = 0)
        : CSGPrimitive(center, dimensions, rot)
    {
        Real w = dimensions[0], h = dimensions[1];
        m_vertical = h > w;
        if (m_vertical) {
            m_a = h * .5;
            m_f = sqrt(m_a * m_a - .25 * w * w);
        }
        else {
            m_a = w * .5;
            m_f = sqrt(m_a * m_a - .25 * h * h);
        }
    }

    std::string defaultName() const {
        return "Ellipse";
    }

    bool isInside(const Vector &p) const {
        Vector l = this->toLocalCoords(p);
        Vector f(m_vertical ? 0 : m_f, m_vertical ? m_f : 0);
        return ((l - f).norm() + (l + f).norm()) <= 2 * m_a;
    }

    ~CSGEllipseNode() { }

private: 
    Real m_f, m_a;
    bool m_vertical;
};

template<typename Functor, typename CSGNode>
void dfsWorker(Functor &f, CSGNode *node)
{
    assert(node != NULL);
    f.preVisit(node);
    for (int i = 0; i < node->numChildren(); ++i) {
        dfsWorker(f, node->child(i));
    }
    f.postVisit(node);
}

template<typename Vector>
template<typename Functor>
Functor CSGTree<Vector>::dfs(Functor f, CSGNode *node)
{
    if (node == NULL) {
        for (int i = 0; i < numRoots(); ++i) {
            dfsWorker(f, root(i));
        }
    }
    else {
        dfsWorker(f, node);
    }
    return f;
}
