#include "BorderedSparseHessian.hh"

namespace MeshFEM {

BorderedSparseFactorization::BorderedSparseFactorization(const BorderedSparseHessian &H, const std::vector<size_t> &fixedVars,
                                CholeskyProvider factorizer) {
    m_sparseDenseStructure = H.varStructure().sparseDenseStructure();
    m_lowRankRank = H.low_rank_rank();
    m_setFixedVars(fixedVars);
    m_solver = make_cholesky_factorizer(factorizer);
    m_solver->factorize(*(H.H_ss), sparseFixedVars());
    m_updateDenseFactorization(H);
}

// Compute the "dense part" of the factorization of:
//  [H_ss B]
//  [B^T  D]
// where B = [H_sd V_s]
// and   D = [H_dd     V_d]
//           [V_d^T   -I_r]
// The Cholesky sparse factorization of (a potentially modified) `H_ss` has
// already been computed in `solver`.
bool BorderedSparseFactorization::m_updateDenseFactorization(const BorderedSparseHessian &H) {
    H.validate();

    size_t nsv = H.varStructure().numSparseVars();
    size_t ndv = H.varStructure().numDenseVars();
    size_t r   = H.low_rank_rank();
    size_t numDenseCols = ndv + r;
    if (numDenseCols == 0) return false; // Nothing to do

    std::vector<size_t> dfv = denseFixedVars();
    if (dfv.size() > 0) throw std::runtime_error("Dense fixed variables are not supported yet.");
    // TODO: implement dense fixed variables
    // This means doing row/column removal on D and removing columns of `H_sd`
    // in `B`. Note that we need not explicitly handle sparse fixed variables
    // (i.e., remove rows of `H_sd` and `V_s`) since they are already handled by
    // `solver().solveMultiRHS`.

    Eigen::MatrixXd D(numDenseCols, numDenseCols);
    B.resize(nsv, ndv + r);

    // Note: the following block initialization throws assertions when
    // some of the blocks are 0x0 (even in the empty case they must
    // have the correct row or column dimension).
    // B << H.H_sd, H.V_s;
    // D << H.H_dd, H.V_d,
    //      H.V_d.transpose(), -Eigen::MatrixXd::Identity(r, r);

    if (ndv > 0) {
        B.leftCols(ndv) = H.H_sd;
        D.topLeftCorner(ndv, ndv) = H.H_dd;
    }
    if (r > 0) {
        B.rightCols(r) = H.V_s;
        if (ndv > 0) {
            D.topRightCorner(ndv, r) = H.V_d;
            D.bottomLeftCorner(r, ndv) = H.V_d.transpose();
        }
        D.bottomRightCorner(r, r) = -Eigen::MatrixXd::Identity(r, r);
    }

    solver().solveMultiRHS(B, H_ss_inv_B);

    S = D - B.transpose() * H_ss_inv_B;

    // Only do Hessian projection for the "H_dd" part of the Schur complement.
    // Note that the lower-right block implementing the SMW update
    // is always negative definite and should not be projected.
    indefinite = false;
    if (ndv > 0) {
        using MXd = Eigen::MatrixXd;
        MXd S_dd = S.topLeftCorner(ndv, ndv);
        Eigen::SelfAdjointEigenSolver<MXd> eigs(S_dd);

        // TODO: offer customization of the projection
        // (currently we're doing abs-based).
        indefinite = (eigs.eigenvalues().array() < 0).any();
        auto S_lambda = eigs.eigenvalues().cwiseAbs().cwiseMax(1e-10).eval();
        S.topLeftCorner(ndv, ndv) = eigs.eigenvectors() * S_lambda.asDiagonal() * eigs.eigenvectors().transpose();
    }

    return indefinite;
}

void BorderedSparseHessian::validate() const {
    const size_t nsv = varStructure().numSparseVars();
    const size_t ndv = varStructure().numDenseVars();

    if (nsv > 0) {
        if (!H_ss) throw std::runtime_error("H_ss is null");

        if (nsv != H_ss->numScalarCols()) throw std::runtime_error("H_ss has the wrong number of columns");
        if (nsv != H_ss->numScalarRows()) throw std::runtime_error("H_ss is not square");
    }

    if ((ndv > 0) && (nsv > 0)) {
        if (nsv != size_t(H_sd.rows())) throw std::runtime_error("H_sd has the wrong number of rows");
        if (ndv != size_t(H_sd.cols())) throw std::runtime_error("H_sd has the wrong number of columns");
        if ((ndv != size_t(H_dd.rows())) || (ndv != size_t(H_dd.cols()))) throw std::runtime_error("H_dd is the wrong shape");
    }

    if ((V_s.size() != 0) || (V_d.size() != 0)) {
        if (nsv != size_t(V_s.rows())) throw std::runtime_error("V_s has the wrong number of columns");
        if (ndv != size_t(V_d.rows())) throw std::runtime_error("V_d has the wrong number of columns");
    }

    if ((C_s.size() != 0) || (C_d.size() != 0)) {
        if (nsv != size_t(C_s.rows())) throw std::runtime_error("C_s has the wrong number of columns");
        if (ndv != size_t(C_d.rows())) throw std::runtime_error("C_d has the wrong number of columns");
    }
}

void BorderedSparseHessian::addNZ(size_t i, size_t j, const Real val) {
    assert(i <= j); // Only support `UPPER_TRIANGLE` symmetry mode...
    const auto &vs = varStructure();
    bool isSparse_i = vs.isSparseVar(i), // Note that sparse variables could
         isSparse_j = vs.isSparseVar(j); // be collected at the beginning or
                                         // end of the variable list!
    if (isSparse_i && isSparse_j) H_ss->addNZScalar(i - vs.sparseVarOffset(), j - vs.sparseVarOffset(), val);
    else if (isSparse_i) H_sd(i - vs.sparseVarOffset(), j - vs.denseVarOffset()) += val;
    else if (isSparse_j) H_sd(j - vs.sparseVarOffset(), i - vs.denseVarOffset()) += val;
    else H_dd(i - vs.denseVarOffset(), j - vs.denseVarOffset()) += val;
}

void BorderedSparseHessian::applyRaw(const double *x_ptr, double *result_ptr) const {
    const auto &vs = varStructure();

    validate();

    Eigen::Map<const Eigen::VectorXd> x(x_ptr, vs.numVars());
    Eigen::Map<Eigen::VectorXd> result(result_ptr, vs.numVars());
    H_ss->applyRawParallel(vs.sparseVars(x).data(), vs.sparseVars(result).data());

    const size_t ndv = vs.numDenseVars();
    const size_t nsv = vs.numSparseVars();

    // Padding terms
    if (ndv > 0) {
        if (nsv > 0) {
            vs.sparseVars(result) += H_sd * vs.denseVars(x);
            vs. denseVars(result)  = H_sd.transpose() * vs.sparseVars(x) + H_dd * vs.denseVars(x); // initializes!
        }
        else {
            vs. denseVars(result) = H_dd * vs.denseVars(x);
        }
    }

    // Low-rank term V V^T x
    const size_t r = low_rank_rank();
    if (r > 0) {
        Eigen::VectorXd Vt_x;
        Vt_x.setZero(r);

        if (ndv > 0) Vt_x += V_d.transpose() * vs.denseVars(x);
        if (nsv > 0) Vt_x += V_s.transpose() * vs.sparseVars(x);

        if (ndv > 0) vs.denseVars(result)  += V_d * Vt_x;
        if (nsv > 0) vs.sparseVars(result) += V_s * Vt_x;
    }
}

// Solve the system:
//    [H_ss B][x] = [b_s]    := [b]
//    [B^T  D][y] = [b_d; 0]    [c]
// Using the Schur complement formulas:
//    S := D - B^T H_ss^{-1} B
//    y = S^{-1} (c - B^T H_ss^{-1} b)
//    x = H_ss^{-1} (b - B y),
void BorderedSparseFactorization::solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const {
    if (!exists()) throw std::runtime_error("Factorization doesn't exist.");
    size_t nsv = m_sparseDenseStructure.numSparseVars;
    size_t ndv = m_sparseDenseStructure.numDenseVars;

    if (B.cols() == 0) {
        // Usual case: no dense variables or low-rank terms;
        // implement thin wrapper around the sparse solver.
        assert((ndv == 0) && (m_lowRankRank == 0));
        solver().solve(b, x);
        return;
    }

    Eigen::VectorXd b_s = m_sparseDenseStructure.sparseVars(b);
    Eigen::VectorXd c(ndv + m_lowRankRank);
    c << m_sparseDenseStructure.denseVars(b), Eigen::VectorXd::Zero(m_lowRankRank);

    // Solve the dense Schur complement system using a QR factorization; we
    // don't use Cholesky since the matrix will be indefinite or negative
    // definite in the case of a low-rank update.
    Eigen::HouseholderQR<Eigen::MatrixXd> dense_solver(S);

    Eigen::VectorXd y = dense_solver.solve(c - H_ss_inv_B.transpose() * b_s);
    Eigen::VectorXd x_s = solver().solve((b_s - B * y).eval());

    m_sparseDenseStructure.sparseVars(x) = x_s;
    m_sparseDenseStructure.denseVars(x)  = y.head(ndv);
}

} // namespace MeshFEM
