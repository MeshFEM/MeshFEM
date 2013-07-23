#include <cmath>
#include <cstdlib>
#include <string>
#include <sstream>
#include <algorithm>

#include <Eigen/Dense>

#include "Geometry.hh"

template<typename Vector>
class CSGTree<Vector>::CSGNode
{
public:
    typedef typename Vector::Scalar Real;
    typedef BoundaryPoint<Vector> _BoundaryPoint;

    CSGNode(const std::string &name, CSGNode *parent = NULL)
        : m_name(name), m_hasName(true), m_parent(parent) { }

    CSGNode(CSGNode *parent = NULL) : m_hasName(false), m_parent(parent) { }

    CSGNode *parent() const { return m_parent; }
    void setParent(CSGNode *p) { m_parent = p; }
    std::string name() const { return m_hasName ? m_name : defaultName(); }
    void setName(const std::string &name) { m_name = name; m_hasName = true; }
    virtual std::string defaultName() const { return "Default"; }

    virtual int indexOfChild(const CSGNode *) const { return -1; }
    virtual int numChildren() const { return 0; }
    virtual CSGTree<Vector>::CSGNode *child(size_t) { return NULL; }
    virtual const CSGTree<Vector>::CSGNode *child(size_t) const { return NULL; }

    virtual BBox_t boundingBox() const { return BBox_t(); }

    virtual void applyTranslation(const Vector &t) {
        for (int i = 0; i < numChildren(); ++i)
            child(i)->applyTranslation(t);
    }

    virtual std::vector<_BoundaryPoint> boundaryPoints(Real) const {
        return std::vector<_BoundaryPoint>();
    }

    virtual CSGNodeType nodeType() const = 0;
    virtual bool isInside(const Vector &p) const = 0;
    virtual Real signedDistance(const Vector &p) const = 0;
    virtual CSGNode *copy() const = 0;
    virtual bool operator==(const CSGNode &b) const = 0;

    virtual ~CSGNode() { }

protected:
    bool m_hasName;
    std::string m_name;
    CSGNode *m_parent;
};

// Lightweight union-type node used to collect CSG subtrees for rendering
// without altering ownership/parenthood of the subtrees.
template<typename Vector>
class CSGTree<Vector>::CSGGlueNode : public CSGTree<Vector>::CSGNode
{
public:
    CSGGlueNode(CSGTree<Vector>::CSGNode *left,
                CSGTree<Vector>::CSGNode *right)
        : m_left(left), m_right(right) { }
    bool isInside(const Vector &p) const {
        return m_left->isInside(p) || m_right->isInside(p);
    }
    Real signedDistance(const Vector &p) const {
        return std::min(m_left->signedDistance(p), m_right->signedDistance(p));
    }
    int indexOfChild(const CSGNode *c) const {
        if (c == m_left)  return 0;
        if (c == m_right) return 1;
        return -1;
    }

    int numChildren() const { return 2; }

    CSGTree<Vector>::CSGNode *child(size_t i) {
        if (i == 0) return m_left;
        if (i == 1) return m_right;
        assert(false);
    }

    const CSGTree<Vector>::CSGNode *child(size_t i) const {
        if (i == 0) return m_left;
        if (i == 1) return m_right;
        assert(false);
    }

    CSGNodeType nodeType() const { return CSG_NODE_UNION; }

    virtual CSGNode *copy() const { assert(false); }

    virtual bool operator==(const CSGNode &b) const {
        const CSGGlueNode *bCast = dynamic_cast<const CSGGlueNode *>(&b);
        return bCast && (*m_left == *bCast->m_left) &&
               (*m_right == *bCast->m_right);
    }

    ~CSGGlueNode() { }
private:
    CSGTree<Vector>::CSGNode *m_left, *m_right;
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

    virtual ~CSGPrimitive() { }

    const Vector &getCenter() const { return m_c; }
    const Vector &getDimensions() const { return m_dim; }

    void setCenter(const Vector &c) { m_c = c; }
    void setDimensions(const Vector &dim) { m_dim = dim; }

    Real getRotation() const { return -m_deg(m_rot_inv.angle()); }
    Real getRotationRad() const { return -m_rot_inv.angle(); }
    void setRotation(Real r) { m_rot_inv.angle() = -m_rad(r); }
    void setRotationRad(Real r) { m_rot_inv.angle() = -r; }

    Vector toLocalCoords(const Vector &p) const {
        return m_rot_inv * (p - m_c);
    }

    BBox_t boundingBox() const {
        Eigen::Rotation2D<Real> rot = m_rot_inv.inverse();
        int n = m_dim.size();
        int nCorners = 1 << n;
        std::vector<Vector> corners(nCorners);
        for (int i = 0; i < nCorners; ++i) {
            for (int j = 0; j < n; ++j) {
                Real sign = ((i & (1 << j))) ? 1.0 : -1.0;
                corners[i][j] = sign * .5 * m_dim[j];
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

    virtual bool operator==(const CSGNode &b) const {
        const CSGPrimitive *bCast = dynamic_cast<const CSGPrimitive *>(&b);
        return bCast && (m_c == bCast->m_c) && (m_dim == bCast->m_dim) &&
               (std::abs(this->getRotation() - bCast->getRotation() < 1e-6));
    }

protected:
    Vector m_c, m_dim;
    Eigen::Rotation2D<Real> m_rot_inv;

private:
    Real m_deg(Real angle) const { return 180.0 * (angle / M_PI); }
    Real m_rad(Real angle) const { return M_PI  * (angle / 180.0); }
};

template<typename Vector>
class CSGTree<Vector>::CSGBoolNode : public CSGTree<Vector>::CSGNode
{
    typedef BoundaryPoint<Vector> _BoundaryPoint;
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

    CSGNodeType nodeType() const {
        switch(m_op) {
            case INTERSECT:
                return CSG_NODE_INTERSECT;
            case UNION:
                return CSG_NODE_UNION;
            case SUBTRACT:
                return CSG_NODE_SUBTRACT;
            default:
                assert(false);
        }
    }

    bool isInside(const Vector &p) const {
        switch (m_op) {
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

    Real signedDistance(const Vector &p) const {
        switch (m_op) {
            case INTERSECT:
                return std::max(m_left->signedDistance(p),
                                m_right->signedDistance(p));
            case UNION:
                return std::min(m_left->signedDistance(p),
                                m_right->signedDistance(p));
            case SUBTRACT:
                // Intersect left with complement of right
                return std::max(m_left->signedDistance(p),
                                -m_right->signedDistance(p));
            default:
                assert(false);
        }
        return 0.0;
    }

    std::vector<_BoundaryPoint> boundaryPoints(Real pointSpacing) const {
        std::vector<_BoundaryPoint> leftPts, rightPts, bndPts;
        leftPts = m_left->boundaryPoints(pointSpacing);
        rightPts = m_right->boundaryPoints(pointSpacing);

        // Perturb boundary points slightly in the normal direction to handle
        // the case where to childrens' boundaries coincide
        std::vector<Vector> perturbLeftPts, perturbRightPts;
        perturbLeftPts.reserve(leftPts.size());
        perturbRightPts.reserve(rightPts.size());
        for (size_t i = 0; i < leftPts.size(); ++i)
            perturbLeftPts[i] = leftPts[i].p + 1e-12 * leftPts[i].n;
        for (size_t i = 0; i < rightPts.size(); ++i)
            perturbRightPts[i] = rightPts[i].p + 1e-12 * rightPts[i].n;

        // New boundary points comprise
        switch(m_op) {
            case INTERSECT: // (left & right)
                // each child boundary point if it is in the other child
                for (size_t i = 0; i < leftPts.size(); ++i) {
                    if (m_right->isInside(perturbLeftPts[i]))
                        bndPts.push_back(leftPts[i]);
                }
                for (size_t i = 0; i < rightPts.size(); ++i) {
                    if (m_left->isInside(perturbRightPts[i]))
                        bndPts.push_back(rightPts[i]);
                }
                break;
            case UNION: // (left | right)
                // each child boundary point if it isn't in the other child
                for (size_t i = 0; i < leftPts.size(); ++i) {
                    if (!(m_right->isInside(perturbLeftPts[i])))
                        bndPts.push_back(leftPts[i]);
                }
                for (size_t i = 0; i < rightPts.size(); ++i) {
                    if (!(m_left->isInside(perturbRightPts[i])))
                        bndPts.push_back(rightPts[i]);
                }
                break;
            case SUBTRACT: // (left - right)
                // left child boundary point if it isn't in right child
                for (size_t i = 0; i < leftPts.size(); ++i) {
                    if (!(m_right->isInside(perturbLeftPts[i])))
                        bndPts.push_back(leftPts[i]);
                }
                // right child boundary point if it is in left child
                // (with reversed normal)
                for (size_t i = 0; i < rightPts.size(); ++i) {
                    if (m_left->isInside(perturbRightPts[i])) {
                        bndPts.push_back(_BoundaryPoint(rightPts[i].p,
                                    -rightPts[i].n, rightPts[i].a));
                    }
                }

                break;
            default:
                assert(false);
        }
        return bndPts;
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
        if (c == m_left)  return 0;
        if (c == m_right) return 1;
        return -1;
    }

    int numChildren() const { return 2; }

    CSGTree<Vector>::CSGNode *child(size_t i) {
        if (i == 0) return m_left;
        if (i == 1) return m_right;
        assert(false);
    }

    const CSGTree<Vector>::CSGNode *child(size_t i) const {
        if (i == 0) return m_left;
        if (i == 1) return m_right;
        assert(false);
    }

    void swapChildren() {
        std::swap(m_left, m_right);
    }


    virtual CSGNode *copy() const {
        return new CSGBoolNode(m_op, m_left->copy(), m_right->copy());
    }

    virtual bool operator==(const CSGNode &b) const {
        const CSGBoolNode *bCast = dynamic_cast<const CSGBoolNode *>(&b);
        return bCast && (m_op == bCast->m_op) && (*m_left == *bCast->m_left) &&
               (*m_right == *bCast->m_right);
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
    typedef BoundaryPoint<Vector> _BoundaryPoint;
public:
    CSGRectangleNode(const Vector &center, const Vector &dimensions, Real rot = 0)
        : CSGPrimitive(center, dimensions, rot)
    { }

    std::string defaultName() const {
        return "Rectangle";
    }

    CSGNodeType nodeType() const {
        return CSG_NODE_RECT;
    }

    bool isInside(const Vector &p) const {
        Vector l = this->toLocalCoords(p);
        return (l.cwiseAbs().array() <= (.5 * this->m_dim).array()).all();
    }

    Real signedDistance(const Vector &p) const {
        Vector d = this->toLocalCoords(p).cwiseAbs() - .5 * this->m_dim;
        bool inside = (d.array() < Vector::Zero().array()).all();
        if (!inside)
            return d.cwiseMax(Vector::Zero()).norm();
        return d.maxCoeff(); // Interior distance is to closest edge
    }

    // Corners are always chosen as boundary points.
    std::vector<_BoundaryPoint> boundaryPoints(Real pointSpacing) const {
        std::vector<_BoundaryPoint> bndPts;
        Real width = this->m_dim[0];
        Real height = this->m_dim[1];
        Real perimeter = 2.0 * (width + height);
        size_t N = ceil(perimeter / pointSpacing);
        // Always at least choose the corners as boundary points.
        size_t nCorners = 4;
        N = std::max(N, nCorners);

        Vector halfDim = .5 * this->m_dim;
        // Corner numbering:
        // 2 3
        // 0 1
        for (size_t i = 0; i < nCorners; ++i) {
            Vector p, n;
            for (int j = 0; j < 2; ++j) {
                Real sign = ((i & (1 << j))) ? 1.0 : -1.0;
                p[j] = sign * halfDim[j];
                n[j] = sign;
            }

            n /= n.norm();
            // Note: areas are assigned to the corners below
            bndPts.push_back(_BoundaryPoint(p, n, 0.0));
        }

        N -= nCorners;

        int widthPoints = (N * width) / (width + height);
        int heightPoints = N - widthPoints;

        // Fit leftPoints + 1 segments on the left edge.
        int leftPoints = .5 * heightPoints;
        Vector p, n(-1, 0);
        Real segmentLength = height / (leftPoints + 1);
        p[0] = -.5 * width;
        for (int i = 0; i < leftPoints; ++i) {
            p[1] = (i + 1) * segmentLength - halfDim[1];
            bndPts.push_back(_BoundaryPoint(p, n, segmentLength));
        }
        // Left segments contribute to the two left corner areas
        bndPts[0].a += .5 * segmentLength;
        bndPts[2].a += .5 * segmentLength;

        // Fit rightPoints + 1 segments on the right edge
        int rightPoints = heightPoints - leftPoints;
        n = Vector(1, 0);
        segmentLength = height / (rightPoints + 1);
        p[0] = .5 * width;
        for (int i = 0; i < rightPoints; ++i) {
            p[1] = (i + 1) * segmentLength - halfDim[1];
            bndPts.push_back(_BoundaryPoint(p, n, segmentLength));
        }
        // Right segments contribute to the two right corner areas
        bndPts[1].a += .5 * segmentLength;
        bndPts[3].a += .5 * segmentLength;

        // Fit topPoints + 1 segments on the top edge
        int topPoints = .5 * widthPoints;
        n = Vector(0, 1);
        segmentLength = width / (topPoints + 1);
        p[1] = .5 * height;
        for (int i = 0; i < topPoints; ++i) {
            p[0] = (i + 1) * segmentLength - halfDim[0];
            bndPts.push_back(_BoundaryPoint(p, n, segmentLength));
        }
        // Top segments contribute to the two top corner areas
        bndPts[2].a += .5 * segmentLength;
        bndPts[3].a += .5 * segmentLength;

        // Fit bottomPoints + 1 segments on the bottom edge
        n = Vector(0, -1);
        int bottomPoints = widthPoints - topPoints;
        segmentLength = width / (bottomPoints + 1);
        p[1] = -.5 * height;
        for (int i = 0; i < bottomPoints; ++i) {
            p[0] = (i + 1) * segmentLength - halfDim[0];
            bndPts.push_back(_BoundaryPoint(p, n, segmentLength));
        }
        // Bottom segments contribute to the two bottom corner areas
        bndPts[0].a += .5 * segmentLength;
        bndPts[1].a += .5 * segmentLength;
        
        // Transorm all boundary points
        Eigen::Rotation2D<Real> rot = this->m_rot_inv.inverse();
        for (size_t i = 0; i < bndPts.size(); ++i) {
            _BoundaryPoint &bp = bndPts[i];
            bp.p = rot * bp.p + this->m_c; 
            bp.n = rot * bp.n;
        }

        // Verify the point areas sum to the perimeter
        Real areaSum = 0.0;
        for (size_t i = 0; i < bndPts.size(); ++i)
            areaSum += bndPts[i].a;
        assert(std::abs(areaSum - perimeter) < 1e-7);

        return bndPts;
    }

    virtual CSGNode *copy() const {
        return new CSGRectangleNode(this->getCenter(), this->getDimensions(),
                                    this->getRotation());
    }

    ~CSGRectangleNode() { }
};


template<typename Vector>
class CSGTree<Vector>::CSGEllipseNode : public CSGTree<Vector>::CSGPrimitive
{
    typedef typename Vector::Scalar Real;
    typedef BoundaryPoint<Vector> _BoundaryPoint;
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

    Vector getFocus() const {
        Vector f(m_vertical ? 0 : m_f, m_vertical ? m_f : 0);
        Eigen::Rotation2D<Real> rot = this->m_rot_inv.inverse();
        return rot * f;
    }

    Real getMajorRadius() const {
        return m_a;
    }

    Real getMinorRadius() const {
        return sqrt(m_a * m_a - m_f * m_f);
    }

    CSGNodeType nodeType() const {
        return CSG_NODE_ELLIPSE;
    }

    bool isInside(const Vector &p) const {
        Vector l = this->toLocalCoords(p);
        Vector f(m_vertical ? 0 : m_f, m_vertical ? m_f : 0);
        return ((l - f).norm() + (l + f).norm()) <= 2 * m_a;
    }

    Real signedDistance(const Vector &p) const {
        // This is an approximation!
        Vector l = this->toLocalCoords(p);
        Vector f(m_vertical ? 0 : m_f, m_vertical ? m_f : 0);
        return ((l - f).norm() + (l + f).norm()) - 2 * m_a;
    }

    // Boundary points are evely spread around ellipse (by arc length)
    // Each ellipse gets area of (arc length) / N.
    std::vector<_BoundaryPoint> boundaryPoints(Real pointSpacing) const {
        std::vector<Real> parameterValues;
        Real a = getMajorRadius();
        Real b = getMinorRadius();
        Real pointAreas;
        ellipseParameterPoints(pointSpacing, a, b, parameterValues,
                               pointAreas);

        size_t N = parameterValues.size();
        std::vector<_BoundaryPoint> bndPts;
        bndPts.reserve(N);

        Eigen::Rotation2D<Real> rot = this->m_rot_inv.inverse();

        for (size_t i = 0; i < N; ++i) {
            Real t = parameterValues[i];
            // Parametrization always assumes major axis is in the x direction.
            Vector p(a * sin(t), b * cos(t));
            // Clockwise tangent vector (parametrization traces clockwise):
            //      t = (a * cos(t), -b * sin(t))
            // Normal is tangent rotated counter-clockwise by 90 degrees:
            //      n = (b * sin(t), a * cos(t))
            Vector n(b * sin(t), a * cos(t));
            n /= n.norm();

            // Vertical elipses have to be rotated by 90 degrees
            if (m_vertical) {
                p = Vector(-p[1], p[0]);
                n = Vector(-n[1], n[0]);
            }

            p = rot * p + this->m_c;
            n = rot * n;

            bndPts.push_back(_BoundaryPoint(p, n, pointAreas));
        }

        return bndPts;
    }

    virtual CSGNode *copy() const {
        return new CSGEllipseNode(this->getCenter(), this->getDimensions(),
                                  this->getRotation());
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
        for (size_t i = 0; i < numRoots(); ++i) {
            dfsWorker(f, root(i));
        }
    }
    else {
        dfsWorker(f, node);
    }
    return f;
}
