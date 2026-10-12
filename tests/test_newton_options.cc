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

TEST_CASE("Explicit factorization updates refresh shift telemetry", "[newton_options][shift_telemetry]") {
    auto makeMatrix = [](Real firstDiagonal) {
        TripletMatrix<> t(2, 2);
        t.addNZ(0, 0, firstDiagonal);
        t.addNZ(1, 1, 4.0);
        SuiteSparseMatrix matrix(t);
        matrix.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        return matrix;
    };
    Eigen::Matrix2d dense = Eigen::Vector2d(-1.0, 4.0).asDiagonal();
    auto problem = std::make_shared<QuadraticProblem>(makeMatrix(-1.0), dense);
    NewtonOptimizer optimizer(problem);
    optimizer.options.beta = 2.0;
    optimizer.options.hessianScaledBeta = false;
    optimizer.options.useIdentityMetric = true;
    optimizer.options.verboseNonPosDef = false;

    // Use unscaled shifting so this two-variable test does not invoke the
    // spectral estimator (which requires a larger matrix). Check the public
    // no-argument overload against the actual factorization and solution.
    optimizer.update_factorizations();
    REQUIRE(problem->lastFactorizationShiftMagnitude() == Approx(2.0));
    auto shift = optimizer.hessianFactorization().identityShift();
    REQUIRE(shift.has_value());
    REQUIRE(*shift == Approx(2.0));
    const Eigen::Vector2d rhs(1.0, 6.0);
    Eigen::VectorXd solution;
    optimizer.hessianFactorization().solve(rhs, solution);
    REQUIRE((solution - Eigen::Vector2d::Ones()).norm() < 1e-12);

    // A subsequent unshifted factorization must clear the old nonzero tau.
    // Exercise the working-set overload too.
    problem->matrix = makeMatrix(2.0);
    problem->dense(0, 0) = 2.0;
    problem->invalidateCachedHessian();
    optimizer.update_factorizations(WorkingSet(*problem));
    REQUIRE(problem->lastFactorizationShiftMagnitude() == 0.0);
    REQUIRE(*optimizer.hessianFactorization().identityShift() == 0.0);

    // An explicit user shift is not an adaptive tau.
    problem->hessianShift = 0.25;
    problem->useRelativeHessianShift = false;
    optimizer.update_factorizations();
    REQUIRE(problem->lastFactorizationShiftMagnitude() == 0.0);
    REQUIRE(*optimizer.hessianFactorization().identityShift() == Approx(0.25));
}
