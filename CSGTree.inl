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
        m_rot_inv.setDegrees(-rot);
    }

    virtual ~CSGPrimitive() { }

    const Vector &getCenter() const { return m_c; }
    const Vector &getDimensions() const { return m_dim; }

    void setCenter(const Vector &c) { m_c = c; }
    void setDimensions(const Vector &dim) { m_dim = dim; }

    Real getRotation() const { return -m_rot_inv.deg(); }
    Real getRotationRad() const { return -m_rot_inv.rad(); }
    void setRotation(Real r) { m_rot_inv.setDegrees(-r); }
    void setRotationRad(Real r) { m_rot_inv.setAngle(-r); }

    Vector toLocalCoords(const Vector &p) const {
        return m_rot_inv(p - m_c);
    }

    BBox_t boundingBox() const {
        int n = m_dim.size();
        int nCorners = 1 << n;
        std::vector<Vector> corners(nCorners);
        for (int i = 0; i < nCorners; ++i) {
            for (int j = 0; j < n; ++j) {
                Real sign = ((i & (1 << j))) ? 1.0 : -1.0;
                corners[i][j] = sign * .5 * m_dim[j];
            }

            // Apply rotation and offset
            corners[i] = m_rot_inv.inverse(corners[i]) + m_c;
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
    FastRotation2D<Real, Vector> m_rot_inv;
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
    using CSGTree<Vector>::CSGPrimitive::m_rot_inv;
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
        return (fabs(l[0]) <= .5 * this->m_dim[0]) && 
               (fabs(l[1]) <= .5 * this->m_dim[1]);
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
        for (size_t i = 0; i < bndPts.size(); ++i) {
            _BoundaryPoint &bp = bndPts[i];
            bp.p = m_rot_inv.inverse(bp.p) + this->m_c; 
            bp.n = m_rot_inv.inverse(bp.n);
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
    using CSGTree<Vector>::CSGPrimitive::m_rot_inv;
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
        return m_rot_inv.inverse(f);
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

            p = m_rot_inv.inverse(p) + this->m_c;
            n = m_rot_inv.inverse(n);

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

// Note: for pie slices, dimensions are actually (radius, angle)
// Angle must be in radians, [0, 2 * pi]
template<typename Vector>
class CSGTree<Vector>::CSGPieSliceNode : public CSGTree<Vector>::CSGPrimitive
{
    using CSGTree<Vector>::CSGPrimitive::m_c;
public:
    CSGPieSliceNode(Vector center, const Vector &dimensions, Real rot = 0)
        : CSGPrimitive(center, dimensions, rot)
    { }

    Real getRadius() const { return this->m_dim[0]; }
    Real getAngle()  const { return this->m_dim[1]; }

    std::string defaultName() const { return "Pie Slice"; }

    CSGNodeType nodeType() const { return CSG_NODE_PIE_SLICE; }

    bool isInside(const Vector &test) const {
        Vector p = test - m_c;
        Real ptheta = atan2(p[1], p[0]);
        Real diff = fmod(ptheta - this->getRotationRad(), 2 * M_PI);
        if (diff < 0) diff += 2 * M_PI;
        return ((diff < getAngle()) && (p.norm() < getRadius()));
    }

    Real signedDistance(const Vector &test) const {
        Vector p = test - m_c;
        Real ptheta = atan2(p[1], p[0]);
        Real diff = fmod(ptheta - this->getRotationRad(), 2 * M_PI);
        if (diff < 0) diff += 2 * M_PI;
        Real wangle = getAngle();
        Real plen = p.norm();
        Real r = getRadius();

        // Determine the angle to the closest border
        Real angleDist = std::min(fabs(diff - wangle), diff);
        angleDist = std::min(angleDist, (Real) (2 * M_PI - diff));

        // Compute unsigned border distance by decomposing into
        // distances perpendicular and parallel to the wedge border.
        Real borderDist = Vector(std::max(plen - r, (Real) 0.0),
                                 plen * sin(angleDist)).norm();
        if (diff < wangle) {
            // Inside the wedge angle, the distance is the max of
            // the radial and (-border dist)
            // (an intersection operation)
            return std::max(plen - r, -borderDist);
        }
        else {
            // Outside the wedge angle, the distance is the border
            // dist.
            return borderDist;
        }
    }

    // Boundary points are evenly spread around perimeter
    // r * ( a + 2), where a is the wedge angle in radians
    std::vector<_BoundaryPoint> boundaryPoints(Real pointSpacing) const {
        std::vector<_BoundaryPoint> bndPts;
        Real a = fmod(getAngle(), 2 * M_PI);
        Real r = getRadius();
        if (a < 0) a += 2 * M_PI;
        Real arclen = r * a;
        Real perimeter = arclen + 2 * r;
        size_t N = ceil(perimeter / pointSpacing);

        size_t Narc = ceil(N * (arclen / perimeter));
        assert(Narc > 1);

        // the "minangle" border will include the origin point, so prefer to
        // give it more points.
        size_t Nminborder = ceil((N - Narc) / 2.0);
        size_t Nmaxborder = N - Narc - Nminborder;

        // Compute the segment lengths (for points interior to the arc, the
        // minangle border, and the maxangle border respectively)
        Real arcSegmentLen = arclen / (Narc - 1);
        Real minBorderSegmentLen = r / (Nminborder);
        Real maxBorderSegmentLen = r / (Nmaxborder + 1);

        // Do the Narc - 2 interal points for the arc
        Real angleDelta = a / (Narc - 1);
        Real currAngle = this->getRotationRad();
        for (size_t i = 1; i < Narc - 1; ++i) {
            currAngle += angleDelta;
            Vector n(cos(currAngle), sin(currAngle));
            bndPts.push_back(_BoundaryPoint(r * n + m_c, n, arcSegmentLen));
        }

        Real minAngle = this->getRotationRad();
        minAngle = fmod(minAngle, 2 * M_PI);
        if (minAngle < 0) minAngle += 2 * M_PI;
        Real maxAngle = minAngle + a;

        // Do the arc endpoints. Note, the arc always meets the end lines at 90
        // degrees, so the "averaged normal" is a 45 degree rotation of the
        // arc's normal.
        Real c45 = cos(M_PI / 4), s45 = sin(M_PI / 4);
        Vector n(cos(minAngle), sin(minAngle));
        Vector p = r * n + m_c;
        // Rotate minangle corner's normal clockwise by 45 degrees
        n = Vector(n[0] * c45 + n[1] * s45, -n[0] * s45 + n[1] * c45);
        bndPts.push_back(_BoundaryPoint(p, n,
                    .5 * (arcSegmentLen + minBorderSegmentLen)));
        n = Vector(cos(maxAngle), sin(maxAngle));
        p = r * n + m_c;
        // Rotate maxangle corner's normal ccw by 45 degrees
        n = Vector(n[0] * c45 - n[1] * s45, n[0] * s45 + n[1] * c45);
        bndPts.push_back(_BoundaryPoint(p, n,
                    .5 * (arcSegmentLen + maxBorderSegmentLen)));

        // Do the Nminborder - 1 interior points
        // Normal vector is a downward unit vector rotated by minAngle.
        n = Vector(sin(minAngle), -cos(minAngle));
        for (size_t i = 1; i < Nminborder; ++i) {
            p = (i * minBorderSegmentLen) * Vector(-n[1], n[0]) + m_c;
            bndPts.push_back(_BoundaryPoint(p, n, minBorderSegmentLen));
        }

        // Do the Nmaxborder interior points
        // Normal vector is an upward unit vector rotated by maxAngle
        n = Vector(-sin(maxAngle), cos(maxAngle));
        for (size_t i = 1; i <= Nmaxborder; ++i) {
            p = (i * maxBorderSegmentLen) * Vector(n[1], -n[0]) + m_c;
            bndPts.push_back(_BoundaryPoint(p, n, maxBorderSegmentLen));
        }
        
        // Do the reentrant corner point.
        // Normal vector is midway between min and max edges.
        Real theta = .5 * (2 * M_PI - a) + maxAngle;
        bndPts.push_back(_BoundaryPoint(m_c, Vector(cos(theta), sin(theta)),
                    .5 * (minBorderSegmentLen + maxBorderSegmentLen)));

        return bndPts;
    }

    BBox_t boundingBox() const {
        Real minAngle = this->getRotationRad();
        minAngle = fmod(minAngle, 2 * M_PI);
        if (minAngle < 0) minAngle += 2 * M_PI;
        Real maxAngle = minAngle + getAngle();

        Vector minV(cos(minAngle), sin(minAngle)), maxV;
        maxV = minV;

        Real currAngle = minAngle;
        currAngle = (M_PI / 2) * ceil(currAngle / (M_PI / 2));

        while (currAngle < maxAngle) {
            Vector cs(cos(currAngle), sin(currAngle));
            minV = minV.cwiseMin(cs);
            maxV = maxV.cwiseMax(cs);
            currAngle += M_PI / 2;
        }

        Vector cs(cos(maxAngle), sin(maxAngle));
        minV = getRadius() * minV.cwiseMin(cs) + this->m_c;
        maxV = getRadius() * maxV.cwiseMax(cs) + this->m_c;

        std::vector<Vector> corners(4);
        corners[0] = Vector(minV[0], minV[1]);
        corners[1] = Vector(maxV[0], minV[1]);
        corners[2] = Vector(maxV[0], maxV[1]);
        corners[3] = Vector(minV[0], maxV[1]);

        BBox_t b(this->m_c, this->m_c);
        for (int i = 0; i < 4; ++i)
            b.unionBox(BBox_t(corners[i], corners[i]));

        return b;
    }

    virtual CSGNode *copy() const {
        return new CSGPieSliceNode(this->getCenter(), this->getDimensions(),
                                  this->getRotation());
    }

    ~CSGPieSliceNode() { }
};

template<typename Real>
Real fast_fmod(Real x, Real mod)
{
    return x - ((int) (x / mod)) * mod;
}

// Note: for laminates, dimensions are actually (epsilon, theta), where epsilon
// is the spacing between slice centers and theta is the thickness of each slice
// (in [0, 1]).
template<typename Vector>
class CSGTree<Vector>::CSGLaminateNode : public CSGTree<Vector>::CSGPrimitive
{
    using CSGTree<Vector>::CSGPrimitive::m_c;
public:
    CSGLaminateNode(Vector center, const Vector &dimensions, Real rot = 0)
        : CSGPrimitive(center, dimensions, rot)
    { }

    Real getEpsilon()  const { return this->m_dim[0]; }
    Real getTheta() const { return this->m_dim[1]; }

    std::string defaultName() const { return "Laminate"; }

    CSGNodeType nodeType() const { return CSG_NODE_LAMINATE; }

    // 
    bool isInside(const Vector &p) const {
        Vector l = this->toLocalCoords(p);
        Real epsilon = getEpsilon();
        Real xEpsilon = fast_fmod(std::abs(l[0]), epsilon);
        return (xEpsilon < epsilon * getTheta() / 2.0) ||
               (xEpsilon > epsilon * (1 -  getTheta() / 2.0));
    }

    Real signedDistance(const Vector &p) const {
        Vector l = this->toLocalCoords(p);
        Real epsilon = getEpsilon();
        Real xEpsilon = fast_fmod(std::abs(l[0]), epsilon);
        return std::min(xEpsilon - epsilon * getTheta() / 2.0,
                        epsilon * (1 -  getTheta() / 2.0) - xEpsilon);
    }

    // Laminates are technically infinite, so it is impossible to generate a
    // finite number of boundary points... better use marching squares!
    std::vector<_BoundaryPoint> boundaryPoints(Real pointSpacing) const {
        std::vector<_BoundaryPoint> bndPts;
        return bndPts;
    }

    // Return a large bounding box so that when we forget to intersect with
    // something we don't break.
    BBox_t boundingBox() const {
        return BBox_t(Vector(-1000, -1000), Vector(1000, 1000));
    }

    virtual CSGNode *copy() const {
        return new CSGLaminateNode(this->getCenter(), this->getDimensions(),
                                   this->getRotation());
    }

    ~CSGLaminateNode() { }
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
