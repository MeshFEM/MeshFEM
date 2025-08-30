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

#include <MeshFEM/Utilities/fast_2x2_decompositions.hh>
#include <MeshFEM/Utilities/fast_3x3_decompositions.hh>
#include <MeshFEM/Utilities/DensePSDDetect.hh>

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
        // Evaluate the exact (unprojected) Hessian
        H = m_lambda * VMap(m_Finv_T.data()) * VMap(m_Finv_T.data()).transpose();

        Real coeff = m_mu - m_lambda * m_logDetF;
        for (size_t j = 0; j < N; ++j)
            for (size_t i = 0; i < N; ++i)
                MMap(H.col(i + j * N).data()) += coeff * m_Finv_T.col(j) * m_Finv.col(i).transpose();

        H.diagonal().array() += m_mu;

        if (elevel == EvalLevel::HessianWithDisabledProjection) return;

        // Project out the negative eigencomponents, subtracting them from the
        // exact Hessian.
#if 0
        // This SVD is taking 40% of the time!!!
        Eigen::JacobiSVD<Matrix, Eigen::NoQRPreconditioner> svd;
        svd.compute(m_F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Matrix &U = svd.matrixU();
        const Matrix &V = svd.matrixV();
        const VecN_T<Real, N> &sigma = svd.singularValues();
#else
        Matrix U, V;
        VecN_T<Real, N> sigma;
        fast_decompositions::svd</* FullyRobust = */ false>(m_F, U, sigma, V);
#endif
        auto sigma_inv = (1.0 / sigma.array()).matrix().eval();

        Real c = m_lambda * m_logDetF - m_mu;

        // Unfortunately the "scaling" mode block is not diagonal for this
        // energy, potentially requiring an `N x N` eigendecomposition.
        Matrix A;
        A.setConstant(m_lambda);
        A.diagonal().array() -= c;
        A = A.array().rowwise() * sigma_inv.transpose().array();
        A = A.array().colwise() * sigma_inv            .array();
        A.diagonal().array() += m_mu;

        // ... however it quite often is positive definite, in which case we
        // can skip the diagonalization.
        bool coupledBlockNonPosdef = !isPSDSylvester(A);

        Eigen::SelfAdjointEigenSolver<Matrix> eig;
        Eigen::Matrix<Real, N * N, N> flattenedScalingEigenmatrices;
        if (coupledBlockNonPosdef) {
            eig.computeDirect(A); // Use faster but less accurate closed-form solver

            Eigen::Matrix<Real, N * N, N> flattenedScalingBasis;
            for (size_t j = 0; j < N; ++j)
                MMap(flattenedScalingBasis.col(j).data()) = U.col(j) * V.col(j).transpose();
            flattenedScalingEigenmatrices = flattenedScalingBasis * eig.eigenvectors();
        }

        Hessian eigenmatricesToProject;
        VecN_T<Real, N * N> projectionShifts;
        size_t num_components_to_project = 0;

        for (size_t j = 0; j < N; ++j) {
            if (coupledBlockNonPosdef && (eig.eigenvalues()[j] < 0)) {
                eigenmatricesToProject.col(num_components_to_project) = flattenedScalingEigenmatrices.col(j);
                projectionShifts[num_components_to_project++] = -eig.eigenvalues()[j];
            }
            for (size_t i = j + 1; i < N; ++i) {
                const Real inv_sigma_product = sigma_inv[i] * sigma_inv[j];
                const Real lambda_T = m_mu + c * inv_sigma_product;
                const Real lambda_L = m_mu - c * inv_sigma_product;

                if (lambda_L >= 0 && lambda_T >= 0) continue;

                Matrix ui_o_vj = U.col(i) * V.col(j).transpose();
                Matrix uj_o_vi = U.col(j) * V.col(i).transpose();

                if (lambda_T < 0) {
                    Matrix T = ui_o_vj - uj_o_vi; // "Twist" eigenmatrix (unnormalized)
                    MMap(eigenmatricesToProject.col(num_components_to_project).data()) = ui_o_vj - uj_o_vi; // "Twist" eigenmatrix (unnormalized)
                    projectionShifts[num_components_to_project++] = -lambda_T * 0.5;
                }

                if (lambda_L < 0) {
                    Matrix L = ui_o_vj + uj_o_vi; // "Flip"  eigenmatrix (unnormalized)
                    MMap(eigenmatricesToProject.col(num_components_to_project).data()) = ui_o_vj + uj_o_vi; // "Flip"  eigenmatrix (unnormalized)
                    projectionShifts[num_components_to_project++] = -lambda_L * 0.5;
                }
            }
        }
        if (num_components_to_project > 0)
            H += eigenmatricesToProject.leftCols(num_components_to_project) * projectionShifts.head(num_components_to_project).asDiagonal() * eigenmatricesToProject.leftCols(num_components_to_project).transpose();

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

            std::cout << "Brute-force projection vs analytical projection relative error: " << (H_brute_proj - H).norm() / H_unproj.norm() << std::endl;
            std::cout << "Relative change from projection: " << (H_brute_proj - H_unproj).norm() / H_unproj.norm() << std::endl;
        }
#endif
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
