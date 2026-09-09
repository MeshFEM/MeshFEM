#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <catch2/catch.hpp>
#include <cmath>

using namespace MeshFEM;

namespace {
struct QuadraticProblem : NewtonProblem {
    QuadraticProblem(const SuiteSparseMatrix &matrix_, const Eigen::Matrix2d &dense_)
        : matrix(matrix_), dense(dense_) { }
    SuiteSparseMatrix matrix;
    Eigen::Matrix2d dense;
    VXd vars = VXd::Zero(2);
    void setVars(const VXd &v) override { vars = v; }
    VXd getVars() const override { return vars; }
    size_t numVars() const override { return 2; }
    Real objective() const override { return 0.5 * vars.dot(gradient()); }
    VXd gradient(bool = false) const override { return dense * vars; }
    NewtonHessian m_getHessianSparsityPattern() const override {
        return NewtonHessian(BlockCSCHessianBase::fromScalar(matrix));
    }
    void m_evalHessian(NewtonHessian &result, bool) const override { result = m_getHessianSparsityPattern(); }
    void m_evalMetric(SuiteSparseMatrix &result) const override { result.setIdentity(true); }
    bool m_updateSparsityPattern() const override { return false; }
};
}

TEST_CASE("Newton factorizer follows precision changes", "[newton_options]") {
    std::vector<CholeskyProvider> providers;
#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
    providers.push_back(CholeskyProvider::Catamari);
#if MESHFEM_WITH_CHOLMOD
    providers.push_back(CholeskyProvider::CatamariNesdisReuse);
#endif
#endif
#ifdef __APPLE__
    providers.push_back(CholeskyProvider::Accelerate);
#endif
    TripletMatrix<> triplets(2, 2);
    triplets.addNZ(0, 0, 4.1);
    triplets.addNZ(0, 1, 0.7);
    triplets.addNZ(1, 1, 3.2);
    SuiteSparseMatrix matrix(triplets);
    matrix.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    Eigen::Matrix2d dense;
    dense << 4.1, 0.7, 0.7, 3.2;
    const Eigen::VectorXd rhs = (Eigen::Vector2d() << 1.2, 2.3).finished();
    const Eigen::VectorXd expected = dense.ldlt().solve(rhs);

    for (auto provider : providers) {
        NewtonOptimizer optimizer(std::make_shared<QuadraticProblem>(matrix, dense));
        auto &options = optimizer.options;
        REQUIRE_FALSE(options.single_precision_factorizer);
        options.factorizer = provider;
        // A precision change must override the Hessian reuse policy.
        options.setHessianUpdateController(HessianUpdateNever());
        auto &factorization = optimizer.hessianFactorization();
        bool hadSolver = false, previousPrecision = false;
        for (bool single : {false, true, true, false}) {
            options.single_precision_factorizer = single;
            Eigen::VectorXd result;
            const Real tau = optimizer.newton_step(result, rhs);
            REQUIRE(std::isnan(tau) == (hadSolver && previousPrecision == single));
            auto &solver = factorization.solver();
            REQUIRE(solver.provider() == provider);
            REQUIRE(solver.hasFactorization());
            REQUIRE((result - expected).norm() < (single ? 1e-6 : 1e-12));
            if (single) {
                REQUIRE((result.array() == result.cast<float>().cast<double>().array()).all());
                REQUIRE((result - expected).norm() > 1e-10);
            }
            hadSolver = true;
            previousPrecision = single;
        }
#if MESHFEM_WITH_CHOLMOD
        // Rejected options must not discard the existing valid solver.
        options.factorizer = CholeskyProvider::CHOLMOD;
        options.single_precision_factorizer = true;
        REQUIRE_THROWS_AS(factorization.solver(), std::invalid_argument);
        options.factorizer = provider;
        options.single_precision_factorizer = false;
        REQUIRE(factorization.solver().hasFactorization());
#endif
    }
}

TEST_CASE("Newton options copies preserve factorizer precision", "[newton_options]") {
    NewtonOptimizerOptions options;
    options.single_precision_factorizer = true;
    NewtonOptimizerOptions copied(options), assigned;
    assigned = options;
    REQUIRE(copied.single_precision_factorizer);
    REQUIRE(assigned.single_precision_factorizer);
    REQUIRE(options.clone()->single_precision_factorizer);
}
