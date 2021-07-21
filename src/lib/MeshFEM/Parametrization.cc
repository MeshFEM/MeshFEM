#include "Parametrization.hh"
#include "Laplacian.hh"
#include "MassMatrix.hh"
#include "Eigensolver.hh"

namespace Parametrization {

struct SPSDSystemSolver : public SPSDSystem<Real> {
    using Base = SPSDSystem<Real>;
    using Base::Base;
};

// Globally scale the parametrization to minimize the squared difference in areas.
// The optimal *area* scaling factor is:
//      min_s 1/2 ||s paramArea - origArea||^2
//          ==>     (s paramArea - origArea) . paramArea = 0
//          ==>     s = (origArea . paramArea) / ||paramArea||^2
// which corresponds to scaling the parametrization by sqrt(s).
// Also translate it to put the (approximate) center of mass at the origin.
UVMap rescale(const Mesh &mesh, Eigen::Ref<const UVMap> uv) {
    // Compute per-triangle areas before and after parametrization
    Eigen::VectorXd origArea(mesh.numTris()), paramArea(mesh.numTris());
    for (const auto t : mesh.elements()) {
        origArea[t.index()] = t->volume();
        std::array<Point2D, 3> poly;
        for (auto v : t.vertices())
            poly[v.localIndex()] = uv.row(v.index()).transpose();
        paramArea[t.index()] = area(poly);
    }

    // Eigen::VectorXd lengthScales = (origArea.array() / paramArea.array()).sqrt().matrix();
    // std::cout << lengthScales.maxCoeff() << ", " << lengthScales.minCoeff() << std::endl;

    // Note: the mass-matrix version would need to use the *flattened* mesh!
    // It's probably not worth the expense of constructing these flat mesh quantities.
    // auto M = MassMatrix::construct(mesh);
    // UVMap result(uv);
    // result.col(0) = uv.col(0).array() - M.apply(uv.col(0).eval()).sum() / mesh.volume();
    // result.col(1) = uv.col(1).array() - M.apply(uv.col(1).eval()).sum() / mesh.volume();

    return (uv.rowwise() - uv.colwise().mean()) * std::sqrt(origArea.dot(paramArea) / paramArea.squaredNorm());
}

////////////////////////////////////////////////////////////////////////////////
// Parametrization Algorithms
////////////////////////////////////////////////////////////////////////////////
// Compute a least-squares conformal parametrization with the global scale factor
// chosen to minimize the L2 norm of the pointwise area distortion.
UVMap lscm(const Mesh &mesh, const UVMap &initParam) {
    const size_t nv = mesh.numVertices();
    UVMap uv(nv, 2);

    SPSDSystemSolver Ksys(assembleLSCMMatrix(mesh));

    // Pin down the null-space (translation, rotation) and avoid the trivial
    // zero solution by fixing two vertices' UVs: boundary vertex 0 and the vertex
    // furthest from it.
    {
        Point3D p0 = mesh.boundaryNode(0).volumeNode()->p;
        size_t idx0 = mesh.boundaryNode(0).volumeNode().index();
        Real furthestDist = 0;
        size_t furthestIdx = 0;

        for (auto bn : mesh.boundaryNodes()) {
            auto n = bn.volumeNode();
            Real dist = (n->p - p0).norm();
            if (dist > furthestDist) {
                furthestDist = dist;
                furthestIdx = n.index();
            }
        }

        std::vector<size_t>    fixedVars = {idx0, furthestIdx, nv + idx0, nv + furthestIdx};
        std::vector<Real> fixedVarValues = {0.0, furthestDist, 0.0, 0.0};

        if (size_t(initParam.rows()) == nv) {
            fixedVarValues[0] = initParam(       idx0, 0);
            fixedVarValues[1] = initParam(furthestIdx, 0);
            fixedVarValues[2] = initParam(       idx0, 1);
            fixedVarValues[3] = initParam(furthestIdx, 1);
        }

        Ksys.fixVariables(fixedVars, fixedVarValues);
    }

    Eigen::VectorXd soln;
    Ksys.solve(Eigen::VectorXd::Zero(2 * nv), soln);
    Eigen::Map<Eigen::VectorXd>(uv.data(), 2 * nv) = soln;

    return rescale(mesh, uv);
}

NDMap harmonic(const Mesh &mesh, NDMap &boundaryData) {
    const size_t nbn = mesh.numBoundaryNodes(),
                 nn  = mesh.numNodes();
    if (size_t(boundaryData.rows()) != nbn) throw std::runtime_error("Invalid boundary data size");
    size_t numComponents = boundaryData.cols();

    NDMap result(nn, numComponents);

    auto L = Laplacian::construct(mesh);
    L.sumRepeated();
    L.needs_sum_repeated = false;
    SPSDSystemSolver Lsys(L);

    // Avoid resetting the SPSDSystemSolver and fixing variables anew for each component solve
    // by always fixing the boundary variables to "0" and directly computing the "load"
    // contributed by these constraints
    std::vector<size_t> fixedVars(nbn);
    for (auto bn : mesh.boundaryNodes())
        fixedVars[bn.index()] = bn.volumeNode().index();
    Lsys.fixVariables(fixedVars, std::vector<double>(nbn, 0.0));
    std::vector<double> negDirichletValues(nn, 0.0);
    std::vector<double> soln;

    for (size_t c = 0; c < numComponents; ++c) {
        for (auto bn : mesh.boundaryNodes())
            negDirichletValues[bn.volumeNode().index()] = -boundaryData(bn.index(), c);
        auto rhs = L.apply(negDirichletValues);
        Lsys.solve(rhs, soln);

        for (auto n : mesh.nodes()) {
            auto bn = n.boundaryNode();
            result(n.index(), c) = bn ? boundaryData(bn.index(), c) : soln[n.index()];
        }
    }

    return result;
}

MESHFEM_EXPORT
UVMap scp(const Mesh &mesh, SCPInnerProduct iprod, Real eps) {
    const size_t nv = mesh.numVertices();

    SuiteSparseMatrix L_C(assembleLSCMMatrix(mesh));
    MatvecCallback applyMetric;

    if (iprod == SCPInnerProduct::I_B) {
        applyMetric = [&](Eigen::Ref<const VXd> x_in) {
            if (size_t(x_in.size()) != 2 * nv) throw std::runtime_error("unexpected size");
            VXd result(VXd::Zero(2 * nv));
            const size_t nbv = mesh.numBoundaryVertices();
            Real uAvg = 0.0, vAvg = 0.0;
            for (const auto &b : mesh.boundaryVertices()) {
                const size_t vi = b.volumeVertex().index();
                result[     vi] = x_in[     vi];
                result[nv + vi] = x_in[nv + vi];
                uAvg += x_in[     vi];
                vAvg += x_in[nv + vi];
            }
            return result;
        };
    }
    else if (iprod == SCPInnerProduct::Mass) {
        SuiteSparseMatrix Mscalar = MassMatrix::construct(mesh);
        applyMetric = [=](Eigen::Ref<const VXd> x_in) {
            if (size_t(x_in.size()) != 2 * nv) throw std::runtime_error("unexpected size");
            VXd result(2 * nv);
            Mscalar.applyRaw(x_in.head(nv).data(), result.head(nv).data());
            Mscalar.applyRaw(x_in.tail(nv).data(), result.tail(nv).data());
            return result;
        };
    }
    else if (iprod == SCPInnerProduct::BMass) {
        SuiteSparseMatrix BMscalar = MassMatrix::construct(mesh.boundary());
        {
            // Convert from per-boundary-vertex to per-vertex matrix.
            TripletMatrix<> BMVertexScalar(nv, nv);
            BMVertexScalar.reserve(BMscalar.nnz());
            for (const auto &t : BMscalar) {
                BMVertexScalar.addNZ(mesh.boundaryVertex(t.i).volumeVertex().index(),
                                     mesh.boundaryVertex(t.j).volumeVertex().index(),
                                     t.v);
            }
            BMscalar = SuiteSparseMatrix(BMVertexScalar);
            BMscalar.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        }
        applyMetric = [=](Eigen::Ref<const VXd> x_in) {
            if (size_t(x_in.size()) != 2 * nv) throw std::runtime_error("unexpected size");
            VXd result(2 * nv);
            BMscalar.applyRaw(x_in.head(nv).data(), result.head(nv).data());
            BMscalar.applyRaw(x_in.tail(nv).data(), result.tail(nv).data());
            return result;
        };
    }
    else throw std::runtime_error("Unknown SCPInnerProduct");

    // Basis for the nullspace.
    Eigen::MatrixX2d Z(2 * nv, 2);
    Z.setZero();
    Z.topLeftCorner(nv, 1).setOnes();
    Z.bottomRightCorner(nv, 1).setOnes();

    Eigen::MatrixXd evec;
#if 0
    // Modify the quadratic form "x^T M x" so that it evaluates to zero on the
    // nullspace of L_C (spanned by the columns of "Z") but keeps its original
    // value on all vectors M-orthogonal to Z's columns (i.e., on all non-zero
    // generalized eigenvectors). This makes the columns of Z no longer the
    // eigenvectors with largest eigenvalue.
    Eigen::MatrixX2d MZ(2 * nv, 2);
    MZ.col(0) = applyMetric(Z.col(0));
    MZ.col(1) = applyMetric(Z.col(1));
    MZ.col(0) /= std::sqrt(Z.col(0).dot(MZ.col(0)));
    MZ.col(1) /= std::sqrt(Z.col(1).dot(MZ.col(1)));

    MatvecCallback B = [&](Eigen::Ref<const VXd> x) {
        return applyMetric(x) - MZ * (MZ.transpose() * x);
    };

    Real mu;
    L_C.addScaledIdentity(eps);
    std::tie(mu, evec) = nthLargestEigenvalueAndEigenvectorGen(B, L_C, 0, 1e-12);

    std::cout.precision(19);
    std::cout << 1.0 / mu << std::endl;
#else
    VXd lambdas;
    std::tie(lambdas, evec) = smallestNonzeroGenEigenpairsPSDKnownKernel(L_C, applyMetric, Z, /* k = */ 1, eps, 1e-12);
    // std::cout.precision(19);
    // std::cout << lambdas << std::endl;
#endif
    return rescale(mesh, Eigen::Map<UVMap>(evec.data(), nv, 2));
}

////////////////////////////////////////////////////////////////////////////////
// Matrix assembly
////////////////////////////////////////////////////////////////////////////////
// Assemble (upper triangle of) LSCM matrix K =
// [L   A] = [L   A]
// [A^T L]   [-A  L]
// where L_ij = int grad phi_i . grad phi_j dA          is the P1 FEM Laplacian and
//       A_ij = int n . (grad phi_j x grad phi_i) dA    is the skew symmetric "parametric area calculator"
//            = int s_ij (1 / 2A) dA = sum_T s_ij|_T / 2
//       s_ij|_T = 1 if local(i) == local(j) + 1, -1 if local(i) == local(j) - 1, 0 otherwise   (this is as evaluated on a particular triangle T)
// This is the quadratic form for [u v] giving the LSCM energy.
// Note: the interior edge contributions to the "area calculator" matrix cancel out, and it can be written as an integral over the boundary.
// However, if we want to support varying triangle weights as recommended in Spectral Conformal Parametrization,
// we need to compute the per-triangle contribution. (This seems to actually be a bad idea though--probably they just need to incorporate a mass matrix in their generalized eigenvalue problem.)
TripletMatrix<> assembleLSCMMatrix(const Mesh &mesh) {
    const size_t nv = mesh.numVertices();
    TripletMatrix<> K(2 * nv, 2 * nv);
    K.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    for (auto tri : mesh.elements()) {
        const auto &gradLambda = tri->gradBarycentric();
        for (auto ni : tri.nodes()) {
            for (auto nj : tri.nodes()) {
                if (ni.index() > nj.index()) continue; // lower triangle
                // Symmetric Laplacian blocks
                const Real val = gradLambda.col(ni.localIndex()).dot(gradLambda.col(nj.localIndex())) * tri->volume();
                K.addNZ(     ni.index(),      nj.index(), val); // (u, u) block
                K.addNZ(nv + ni.index(), nv + nj.index(), val); // (v, v) block

                // Skew symmetric A block (u, v)
                if (ni.localIndex() == nj.localIndex()) continue;
                int s = (ni.localIndex() == (nj.localIndex() + 1) % 3) ? 1.0 : -1.0;
                K.addNZ(ni.index(), nv + nj.index(),  0.5 * s);
                K.addNZ(nj.index(), nv + ni.index(), -0.5 * s);
            }
        }
    }
    return K;
}

TripletMatrix<> assembleBMatrixSCP(const Mesh &mesh) {
    const size_t nv = mesh.numVertices();
    TripletMatrix<> B(2 * nv, 2 * nv);
    B.reserve(2 * mesh.numBoundaryVertices());
    B.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    for (const auto &bv : mesh.boundaryVertices()) {
        const size_t i = bv.volumeVertex().index();
        B.addNZ(     i,      i, 1.0);
        B.addNZ(nv + i, nv + i, 1.0);
    }

    return B;
}

TripletMatrix<> assembleMassMatrix(const Mesh &mesh) {
    TripletMatrix<> Mscalar = MassMatrix::construct(mesh);
    Mscalar.sumRepeated();

    TripletMatrix<> M(Mscalar.n * 2, Mscalar.m * 2);
    M.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;
    M.reserve(2 * Mscalar.nnz());

    const size_t nv = mesh.numVertices();
    for (const auto &t : Mscalar) {
        M.addNZ(t.i, t.j, t.v);
        M.addNZ(nv + t.i, nv + t.j, t.v);
    }

    return M;
}

}
