////////////////////////////////////////////////////////////////////////////////
// TensionFieldTheory.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements a field theory relaxed energy density for a fully generic,
//  potentially anisotropic, 2D "C-based" energy density. Here "C-based" means
//  the energy density is expressed in terms of the Cauchy-Green deformation
//  tensor.
//  Our implementation is based on the applied math paper
//  [Pipkin1994:"Relaxed energy densities for large deformations of membranes"]
//  whose optimality criterion for the wrinkling strain we use to obtain a
//  slightly less expensive optimization formulation.
//
//  We implement the second derivatives needed for a Newton-based equilibrium
//  solver; these are nontrivial since they must account for the dependence of
//  the wrinkling strain on C.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/28/2020 19:01:21
////////////////////////////////////////////////////////////////////////////////
#ifndef TENSIONFIELDTHEORY_HH
#define TENSIONFIELDTHEORY_HH
#include <Eigen/Dense>
#include <MeshFEM/Types.hh>
#include <stdexcept>
#include <iostream>
#include <MeshFEM/newton_optimizer/dense_newton.hh>

template<class _Psi>
struct IsotropicWrinkleStrainProblem {
    using Psi      = _Psi;
    using Real     = typename Psi::Real;
    using VarType  = Eigen::Matrix<Real, 1, 1>;
    using HessType = Eigen::Matrix<Real, 1, 1>;
    using M2d      = Mat2_T<Real>;

    IsotropicWrinkleStrainProblem(Psi &psi, const M2d &C, const Vec2_T<Real> &n)
        : m_psi(psi), m_C(C), m_nn(n * n.transpose()) { }

    void setC(const M2d &C) { m_C = C; }
    const M2d &getC() const { return m_C; }

    size_t numVars() const { return 1; }
    void setVars(const VarType &vars) {
        m_a = vars;
        m_psi.setC(m_C + m_a[0] * m_nn);
    }
    const VarType &getVars() const { return m_a; }

    Real energy()      const { return m_psi.energy(); }
    VarType gradient() const { return  VarType(0.5 * doubleContract(m_nn, m_psi.PK2Stress())          ); }
    HessType hessian() const { return HessType(0.5 * doubleContract(m_psi.delta_PK2Stress(m_nn), m_nn)); }

    void solve() { dense_newton(*this, /* maxIter = */ 100, /*gradTol = */1e-14, /* verbose = */ false); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    VarType m_a = VarType::Zero();
    Psi &m_psi;
    M2d m_C, m_nn;
};

template<class _Psi>
struct AnisotropicWrinkleStrainProblem {
    using Psi      = _Psi;
    using Real     = typename Psi::Real;
    using VarType  = Vec2_T<Real>;
    using HessType = Mat2_T<Real>;
    using M2d      = Mat2_T<Real>;

    AnisotropicWrinkleStrainProblem(Psi &psi, const M2d &C, const VarType &n)
        : m_psi(psi), m_C(C), m_ntilde(n) { }

    void setC(const M2d &C) { m_C = C; }
    const M2d &getC() const { return m_C; }

    size_t numVars() const { return 2; }
    void setVars(const VarType &vars) {
        m_ntilde = vars;
        m_psi.setC(m_C + m_ntilde * m_ntilde.transpose());
    }
    const VarType &getVars() const { return m_ntilde; }
    Real energy() const { return m_psi.energy(); }

    // S : (0.5 * (n delta_n^T + delta_n n^T))
    //  = S : n delta_n^T = (S n) . delta_n
    VarType gradient() const { return m_psi.PK2Stress() * m_ntilde; }

    //   psi(n * n^T)
    //  dpsi = n^T psi'(n * n^T) . dn
    // d2psi = dn_a^T psi'(n * n^T) . dn_b + n^T (psi'' : (n * dn_a^T)) . dn_b
    //       = psi' : (dn_a dn_b^T) + ...
    HessType hessian() const {
        HessType h = m_psi.PK2Stress(); // psi' : (dn_a dn_b^T)
        M2d dnn(M2d::Zero());
        dnn.col(0) = m_ntilde;
        h.row(0) += m_ntilde.transpose() * m_psi.delta_PK2Stress(symmetrized_x2(dnn));
        dnn.col(0).setZero();
        dnn.col(1) = m_ntilde;
        h.row(1) += m_ntilde.transpose() * m_psi.delta_PK2Stress(symmetrized_x2(dnn));
        if (h.array().isNaN().any()) throw std::runtime_error("NaN Hessian");
        if (std::abs(h(0, 1) - h(1, 0)) > 1e-10 * std::abs(h(1, 0)) + 1e-10)
            throw std::runtime_error("Asymmetric Hessian");
        return h;
    }

    void solve() { dense_newton(*this, /* maxIter = */ 100, /*gradTol = */1e-14, /* verbose = */ false); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Psi &m_psi;
    M2d m_C;
    VarType m_ntilde = VarType::Zero();
};

// Define a relaxed 2D C-based energy based on a given 2D C-based energy `Psi_C`
template<class Psi_C>
struct RelaxedEnergyDensity {
    static_assert(Psi_C::EDType == EDensityType::CBased,
                  "Tension field theory only works on C-based energy densities");
    static constexpr size_t N = Psi_C::N;
    static constexpr size_t Dimension = N;
    static constexpr EDensityType EDType = EDensityType::CBased;

    using Matrix = typename Psi_C::Matrix;
    using Real   = typename Psi_C::Real;
    using ES     = Eigen::SelfAdjointEigenSolver<Matrix>;
    using V2d    = Vec2_T<Real>;
    static_assert(N == 2, "Tension field theory relaxation only defined for 2D energies");

    static std::string name() {
        return std::string("Relaxed") + Psi_C::name();
    }

    // Forward all constructor arguments to m_psi
    template<class... Args>
    RelaxedEnergyDensity(Args&&... args)
         : m_psi(std::forward<Args>(args)...),
           m_anisoProb(m_psi, Matrix::Identity(), V2d::Zero()) { }

    // We need a custom copy constructor since m_anisoProb contains a
    // reference to this->m_psi
    RelaxedEnergyDensity(const RelaxedEnergyDensity &b)
        : m_psi(b.m_psi),
          m_anisoProb(m_psi, Matrix::Identity(), V2d::Zero()) {
        setC(b.m_C);
    }

    // Note: UninitializedDeformationTag argument must be a rvalue reference so it exactly
    // matches the type passed by the constructor call
    // RelaxedEnergyDensity(b, UninitializedDeformationTag()); otherwise the
    // perfect forwarding constructor above will be preferred for this call,
    // incorrectly forwarding b to Psi's constructor.
    RelaxedEnergyDensity(const RelaxedEnergyDensity &b, UninitializedDeformationTag &&)
        : m_psi(b.m_psi, UninitializedDeformationTag()),
          m_anisoProb(m_psi, Matrix::Identity(), V2d::Zero()) {
        setC(b.m_C);
    }

    void setC(const Matrix &C) {
        m_C = C;
        if (!relaxationEnabled()) {
            m_psi.setC(C);
            return;
        }

        // Note: Eigen guarantees eigenvalues are sorted in ascending order.
        ES C_eigs(C);
        // std::cout << "C eigenvalues: " << C_eigs.eigenvalues().transpose() << std::endl;
        // Detect full compression
        if (C_eigs.eigenvalues()[1] < 1) {
            m_tensionState = 0;
            m_wrinkleStrain = -C;
            return;
        }

        m_psi.setC(C);
        ES S_eigs(m_psi.PK2Stress());
        // Detect full tension
        // std::cout << "S eigenvalues: " << S_eigs.eigenvalues().transpose() << std::endl;
        if (S_eigs.eigenvalues()[0] >= -1e-12) {
            m_tensionState = 2;
            m_wrinkleStrain.setZero();
            return;
        }

        // Handle partial tension
        m_tensionState = 1;
        // In the isotropic case, principal stress and strain directions
        // coincide, and the wrinkling strain must be in the form
        // a n n^T, where n is the eigenvector corresponding to the smallest
        // stress eigenvalue and a > 0 is an unknown.
        // This simplifies the determination of wrinkling strain to a convex
        // 1D optimization problem that we solve with Newton's method.
        V2d n = S_eigs.eigenvectors().col(0);
        using IWSP = IsotropicWrinkleStrainProblem<Psi_C>;
        IWSP isoProb(m_psi, C, n);
        // std::cout << "Solving isotropic wrinkle strain problem" << std::endl;
        isoProb.setVars(typename IWSP::VarType{0.0});
        isoProb.solve();
        Real a = isoProb.getVars()[0];
        if (a < 0) throw std::runtime_error("Invalid wrinkle strain");

        // We use this isotropic assumption to obtain initial guess for the
        // anisotropic case, where the wrinkling strain is in the form
        //      n_tilde n_tilde^T
        // with a 2D vector n_tilde as the unknown.
        m_anisoProb.setC(C);
        // std::cout << "Solving anisotropic wrinkle strain problem" << std::endl;
        m_anisoProb.setVars(std::sqrt(a) * n);
        m_anisoProb.solve();
        auto ntilde = m_anisoProb.getVars();
        m_wrinkleStrain = -ntilde * ntilde.transpose();

        // {
        //     ES S_eigs_new(m_psi.PK2Stress());
        //     std::cout << "new S eigenvalues: " << S_eigs_new.eigenvalues().transpose() << std::endl;
        // }
    }

    Real energy() const {
        if (relaxationEnabled() && fullCompression()) return 0.0;
        return m_psi.energy();
    }

    Matrix PK2Stress() const {
        if (relaxationEnabled() && fullCompression()) return Matrix::Zero();
        // By envelope theorem, the wrinkling strain's perturbation can be
        // neglected, and the stress is simply the stress of the underlying
        // material model evaluated on the "elastic strain".
        return m_psi.PK2Stress();
    }

    template<class Mat_>
    Matrix delta_PK2Stress(const Mat_ &dC) const {
        if (!relaxationEnabled() || fullTension()) return m_psi.delta_PK2Stress(dC);
        if (fullCompression()) return Matrix::Zero();

        // n solves m_anisoProb:
        //     (psi') n = 0
        // delta_n solves:
        //     (psi'' : dC + n delta n) n + (psi') delta_n = 0
        //     (psi'' : n delta_n) n + (psi') delta_n = -(psi'' : dC) n
        //     H delta_n = -(psi'' : dC) n
        auto ntilde = m_anisoProb.getVars();
        V2d delta_n = -m_anisoProb.hessian().inverse() * (m_psi.delta_PK2Stress(dC.matrix()) * ntilde);

        // In the partial tension case, we need to account for the wrinkling
        // strain perturbation.
        return m_psi.delta_PK2Stress(dC.matrix() + ntilde * delta_n.transpose()
                                                 + delta_n * ntilde.transpose());
    }

    template<class Mat_, class Mat2_>
    Matrix delta2_PK2Stress(const Mat_ &/* dF_a */, const Mat2_ &/* dF_b */) const {
        throw std::runtime_error("Unimplemented");
    }

    V2d principalBiotStrains() const {
        ES es(m_C);
        return es.eigenvalues().array().sqrt() - 1.0;
    }

    const Matrix &wrinkleStrain() const { return m_wrinkleStrain; }

    bool fullCompression() const { return m_tensionState == 0; }
    bool partialTension()  const { return m_tensionState == 1; }
    bool fullTension()     const { return m_tensionState == 2; }
    int tensionState()     const { return m_tensionState; }

    // Copying is super dangerous since we m_anisoProb uses a reference to m_psi...
    RelaxedEnergyDensity &operator=(const RelaxedEnergyDensity &) = delete;

    // Direct access to the energy density for debugging or
    // to change the material properties
    Psi_C &psi() { return m_psi; }
    const Psi_C &psi() const { return m_psi; }

    // Turn on or off the tension field theory approximation
    void setRelaxationEnabled(bool enable) {
        m_relaxationEnabled = enable;
        setC(m_C);
    }

    bool relaxationEnabled() const { return m_relaxationEnabled; }

    void copyMaterialProperties(const RelaxedEnergyDensity &other) { psi().copyMaterialProperties(other.psi()); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    // Tension state:
    //      0: compression in all directions
    //      1: partial tension
    //      2: tension in all directions
    int m_tensionState = 2;
    Psi_C m_psi;
    Matrix m_wrinkleStrain = Matrix::Zero(),
           m_C = Matrix::Identity(); // full Cauchy-Green deformation tensor.
    AnisotropicWrinkleStrainProblem<Psi_C> m_anisoProb;
    bool m_relaxationEnabled = true;
};

#endif /* end of include guard: TENSIONFIELDTHEORY_HH */
