////////////////////////////////////////////////////////////////////////////////
// CommonNeoHookean.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  The variant of the neo-Hookean material model used in PolyFEM:
//      psi(F) = mu/2 (I1 - d - 2 ln(J)) + lambda/2 (ln(J))^2
//  It's also the primary version analyzed in [Smith et al. 2018: Stable Neo-Hookean].
//  Note that while this model is common in graphics, it does *not* properly
//  model 2D materials since it does not solve for the relaxation in the
//  transverse direction.
//  It corresponds to the UJOption = 2 setting discussed here:
//     https://osupdocs.forestry.oregonstate.edu/index.php/Neo-Hookean_Material
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/18/2025 12:18:22
*///////////////////////////////////////////////////////////////////////////////
#ifndef COMMONNEOHOOKEAN_HH
#define COMMONNEOHOOKEAN_HH

#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>

#include <catamari/dense_basic_linear_algebra.hpp>

template<typename _Real, size_t _Dim>
struct CommonNeoHookeanEnergy : public Concepts::NeoHookeanEnergy {
    static constexpr size_t Dimension = _Dim;
    static constexpr size_t N         = _Dim;
    static constexpr EDensityType EDType = EDensityType::FBased;
    using Real = _Real;
    using Matrix = Eigen::Matrix<Real, N, N>;

    static constexpr const char *name() { return "CommonNeoHookean"; }

    CommonNeoHookeanEnergy(const CommonNeoHookeanEnergy &/* other */) = default;
    CommonNeoHookeanEnergy() { setDeformationGradient(Matrix::Identity()); }
    CommonNeoHookeanEnergy &operator=(const CommonNeoHookeanEnergy &/* other */) = default;

    CommonNeoHookeanEnergy(const CommonNeoHookeanEnergy &other, UninitializedDeformationTag &&)
        : m_lambda(other.m_lambda), m_mu(other.m_mu) { }

    CommonNeoHookeanEnergy(Real lambda = 0, Real mu = 0.5)
        : m_lambda(lambda), m_mu(mu) { setDeformationGradient(Matrix::Identity()); }

    void setDeformationGradient(const Matrix& deformation_gradient, const EvalLevel elevel = EvalLevel::Full) {
        m_F = deformation_gradient;
        m_detF = deformation_gradient.determinant();
        m_logDetF = std::log(m_detF);

        if (elevel < EvalLevel::Gradient) return;
        m_Finv = m_F.inverse();
        m_Finv_T = m_Finv.transpose();

        if (elevel < EvalLevel::Hessian) return;

        using VMap = Eigen::Map<const VecN_T<Real, N * N>>;
        using MMap = Eigen::Map<Matrix>;
        Hessian &H = m_d2energy;
        if (elevel == EvalLevel::HessianWithDisabledProjection) {
            // Unprojected Hessian
            H = m_lambda * VMap(m_Finv_T.data()) * VMap(m_Finv_T.data()).transpose();

            Real coeff = m_mu - m_lambda * m_logDetF;
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < N; ++i)
                    MMap(H.col(i + j * N).data()) += coeff * m_Finv_T.col(j) * m_Finv.col(i).transpose();

            H.diagonal().array() += m_mu;
        }
        else {
            // Analytical Hessian projection
            // This SVD is taking 40% of the time!!!
            Eigen::JacobiSVD<Matrix, Eigen::NoQRPreconditioner> svd;
            svd.compute(m_F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Matrix &U = svd.matrixU();
            const Matrix &V = svd.matrixV();
            const VecN_T<Real, N> &sigma = svd.singularValues();
            // const Matrix &U = m_F;
            // const Matrix &V = m_Finv_T;
            // const VecN_T<Real, N> sigma = m_F.diagonal();

            auto sigma_inv = (1.0 / sigma.array()).matrix().eval();

            Real c = m_lambda * m_logDetF - m_mu;

            Matrix A;
            A.setConstant(m_lambda);
            A.diagonal().array() -= c;
            A = A.array().rowwise() * sigma_inv.transpose().array();
            A = A.array().colwise() * sigma_inv            .array();
            Eigen::SelfAdjointEigenSolver<Matrix> eig;
            eig.computeDirect(A); // Use faster but less accurate closed-form solver
            const auto &Q_A = eig.eigenvectors();
            const VecN_T<Real, N> &eigenvalues_A = eig.eigenvalues();
            // const auto Q_A = m_F;
            // const auto eigenvalues_A = m_F.diagonal();

            Eigen::Matrix<Real, N * N, N> flattenedScalingBasis;
            for (size_t j = 0; j < N; ++j)
                MMap(flattenedScalingBasis.col(j).data()) = U.col(j) * V.col(j).transpose();

            Eigen::Matrix<Real, N * N, N> flattenedEigenmatrices = flattenedScalingBasis * Q_A;
#if 0
            H = flattenedEigenmatrices * (eig.eigenvalues().array() + m_mu).matrix().cwiseMax(0.0).asDiagonal() * flattenedEigenmatrices.transpose();

            for (size_t j = 0; j < N; ++j) {
                for (size_t i = j + 1; i < N; ++i) {
                    const Real inv_sigma_product = sigma_inv[i] * sigma_inv[j];
                    const Real lambda_T = m_mu + c * inv_sigma_product;
                    const Real lambda_L = m_mu - c * inv_sigma_product;

                    if (lambda_L < 0 && lambda_T < 0) continue;

                    Matrix ui_o_vj = U.col(i) * V.col(j).transpose();
                    Matrix uj_o_vi = U.col(j) * V.col(i).transpose();

                    if (lambda_T > 0) {
                        Matrix T = ui_o_vj - uj_o_vi; // "Twist" eigenmatrix (unnormalized)
                        H += VMap(T.data()) * VMap(T.data()).transpose() * (lambda_T * 0.5);
                    }

                    if (lambda_L > 0) {
                        Matrix L = ui_o_vj + uj_o_vi; // "Flip"  eigenmatrix (unnormalized)
                        H += VMap(L.data()) * VMap(L.data()).transpose() * (lambda_L * 0.5);
                    }
                }
            }
#else
            Hessian nonnullEigenmatrices;
            VecN_T<Real, N * N> nonnullEigenvalues;
            size_t num_nonnull = 0;

            auto insert_eigenpair = [&](Real eigenvalue, const auto &flat_eigenmatrix) {
                nonnullEigenvalues[num_nonnull] = eigenvalue;
                nonnullEigenmatrices.col(num_nonnull++) = flat_eigenmatrix;
            };

            for (size_t j = 0; j < N; ++j) {
                Real lambda_scale_j = eigenvalues_A[j] + m_mu;
                if (lambda_scale_j > 0) insert_eigenpair(lambda_scale_j, flattenedEigenmatrices.col(j));

                for (size_t i = j + 1; i < N; ++i) {
                    const Real inv_sigma_product = sigma_inv[i] * sigma_inv[j];
                    const Real lambda_T = m_mu + c * inv_sigma_product;
                    const Real lambda_L = m_mu - c * inv_sigma_product;

                    if (lambda_L < 0 && lambda_T < 0) continue;

                    Matrix ui_o_vj = U.col(i) * V.col(j).transpose();
                    Matrix uj_o_vi = U.col(j) * V.col(i).transpose();

                    if (lambda_T > 0) {
                        Matrix T = ui_o_vj - uj_o_vi; // "Twist" eigenmatrix (unnormalized)
                        insert_eigenpair(lambda_T * 0.5, VMap(T.data()));
                    }

                    if (lambda_L > 0) {
                        Matrix L = ui_o_vj + uj_o_vi; // "Flip" eigenmatrix (unnormalized)
                        insert_eigenpair(lambda_L * 0.5, VMap(L.data()));
                    }
                }
            }

            // One big syrk...
            // VecN_T<Real, N * N> sqrt_eigenvalues;
            // sqrt_eigenvalues.head(num_nonnull) = nonnullEigenvalues.head(num_nonnull).cwiseSqrt();
            // nonnullEigenmatrices.leftCols(num_nonnull) = nonnullEigenmatrices.leftCols(num_nonnull).array().rowwise() * sqrt_eigenvalues.head(num_nonnull).transpose().array();
#if 1
            // H = nonnullEigenmatrices.leftCols(num_nonnull) * nonnullEigenmatrices.leftCols(num_nonnull).transpose();
            if (num_nonnull == 0) H.setZero();
            else H = nonnullEigenmatrices.leftCols(num_nonnull) * nonnullEigenvalues.head(num_nonnull).asDiagonal() * nonnullEigenmatrices.leftCols(num_nonnull).transpose();
#else
            catamari::BlasMatrixView<Real> H_view;
            catamari::ConstBlasMatrixView<Real> Q_view;
            H_view.data = H.data();
            H_view.height = H.rows();
            H_view.width = H.cols();
            H_view.leading_dim = H.rows();

            Q_view.data = nonnullEigenmatrices.data();
            Q_view.height = nonnullEigenmatrices.rows();
            Q_view.width = num_nonnull;
            Q_view.leading_dim = nonnullEigenmatrices.rows();

            LowerNormalHermitianOuterProduct(/* alpha = */ Real{1}, Q_view, /* beta = */ Real{0}, &H_view);
            H.template triangularView<Eigen::StrictlyUpper>() = H.transpose();
            // MatrixMultiplyNormalTranspose(/* alpha = */ Real{1}, Q_view, Q_view, /* beta = */ Real{0}, &H_view);
#endif

#endif

#if 0
            {
                // Validate against brute-force projection.
                Hessian H_unproj = m_lambda * VMap(m_Finv_T.data()) * VMap(m_Finv_T.data()).transpose();

                Real coeff = m_mu - m_lambda * m_logDetF;
                for (size_t j = 0; j < N; ++j)
                    for (size_t i = 0; i < N; ++i)
                        MMap(H_unproj.col(i + j * N).data()) += coeff * m_Finv_T.col(j) * m_Finv.col(i).transpose();

                H_unproj.diagonal().array() += m_mu;

                Eigen::SelfAdjointEigenSolver<Hessian> Hes(H_unproj);
                Hessian H_brute_proj = Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(0.0).asDiagonal() * Hes.eigenvectors().transpose();

                std::cout << "Brute-force projection vs analytical projection relative error: " << (H_brute_proj - H).norm() / H_brute_proj.norm() << std::endl;
                std::cout << "Relative change from projection: " << (H_brute_proj - H_unproj).norm() / H_unproj.norm() << std::endl;
            }
#endif
        }
    }

    const Matrix &getDeformationGradient() const { return m_F; }

    Real energy() const {
        // Standard behavior: return inf for inverted elements
        if (m_detF < 0) return std::numeric_limits<Real>::infinity();
        return m_mu / 2.0 * (m_F.squaredNorm() - Dimension - 2.0 * m_logDetF) + m_lambda / 2.0 * m_logDetF * m_logDetF;
    }

    Matrix denergy() const { return m_mu * m_F + (m_lambda * m_logDetF - m_mu) * m_Finv_T; }
    Real denergy(const Matrix& dF) const { return doubleContract(dF, denergy()); }

    template<typename Mat_>
    Matrix delta_denergy(const Mat_ &dF) const {
        return applyFlattened4thOrderTensor(d2energy(), dF);
        // Matrix dF_mat = dF.matrix();
        // return m_mu * dF_mat
        //     - (m_lambda * m_logDetF - m_mu) * m_Finv_T * dF_mat.transpose() * m_Finv_T
        //     + m_lambda * doubleContract(m_Finv_T, dF) * m_Finv_T;
        // return m_mu * (dF_mat + m_Finv_T * dF_mat.transpose() * m_Finv_T)
        //          - m_lambda * m_logDetF * m_Finv_T * dF_mat.transpose() * m_Finv_T
        //          + m_lambda * doubleContract(m_Finv_T, dF) * m_Finv_T;
    }

    Real d2energy(const Matrix &dF_a, const Matrix &dF_b) const {
        return doubleContract(dF_a, delta_denergy(dF_b));
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_denergy(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const {
        throw std::runtime_error("Unimplemented.");
    }

    using Hessian = Eigen::Matrix<Real, N * N, N * N>;
    const Hessian &d2energy() const { return m_d2energy; }

    Matrix PK2Stress() const { return m_Finv_T.transpose() * denergy(); }

private:
    Real m_lambda = 0.0; // Lame's first parameter
    Real m_mu = 0.0;     // Shear modulus

    // Cached deformation quantities.
    Matrix m_F, m_Finv, m_Finv_T;
    Real m_detF, m_logDetF;
    Hessian m_d2energy;
};

#endif // COMMONNEOHOOKEAN_HH
