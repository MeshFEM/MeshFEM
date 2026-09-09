#if MESHFEM_WITH_CATAMARI && MESHFEM_WITH_CHOLMOD && !defined(MESHFEM_USE_LEGACY_CATAMARI)
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>
#include <MeshFEM/newton_optimizer/NewtonHessianFactorization.hh>
#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>
#include <catch2/catch.hpp>

using namespace MeshFEM;

namespace {
struct PatternTerm : NewtonObjectiveTermBase {
    static constexpr size_t n = 16;
    explicit PatternTerm(SparsityUpdateFrequency frequency_) : frequency(frequency_) { }
    SparsityUpdateFrequency frequency;
    std::vector<std::pair<size_t, size_t>> edges;
    void setEdges(std::vector<std::pair<size_t, size_t>> e) {
        edges = std::move(e);
        sparsityPatternChanged.set();
    }
    SparsityUpdateFrequency sparsityUpdateFrequency() const override { return frequency; }
    size_t numVars() const override { return n; }
    Real objective() const override { return 0; }
    void accumulateGradient(Real, VXd &, bool) const override { }
    void accumulateHessian(Real, NewtonHessian &, bool) const override { }
    NewtonHessian hessianSparsityPattern() const override {
        TripletMatrix<> t(n, n);
        if (frequency == SparsityUpdateFrequency::NEVER) {
            for (size_t v = 0; v < n; ++v) {
                t.addNZ(v, v, 4);
                if (v + 1 < n) t.addNZ(v, v + 1, -1);
            }
        }
        for (auto [a, b] : edges) t.addNZ(a, b, -1);
        SuiteSparseMatrix matrix(t);
        matrix.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        return NewtonHessian(BlockCSCHessianBase::fromScalar(std::move(matrix)));
    }
};

// Also exercise a reset request without any sparsity change, independently of
// SLRU's policy for requesting one.
struct ResettableProblem : NewtonMultiobjectiveProblem {
    using NewtonMultiobjectiveProblem::NewtonMultiobjectiveProblem;
    void requestReset() { m_requestSymbolicFactorizationReset(); }
};

const CholmodParallelNesdis::TemporalReuseStatistics &stats(NewtonHessianFactorization &f) {
    return dynamic_cast<const CatamariFactorizer &>(f.solver()).temporalReuseStatistics();
}
}

TEST_CASE("SLRU expiration resets ND reuse for every factorizer", "[temporal_nd][slru]") {
    using Frequency = NewtonObjectiveTermBase::SparsityUpdateFrequency;
    const auto frequency = GENERATE(Frequency::SOMETIMES, Frequency::ALWAYS);
    auto staticTerm = std::make_shared<PatternTerm>(Frequency::NEVER);
    auto dynamicTerm = std::make_shared<PatternTerm>(frequency);
    dynamicTerm->setEdges({{0, 3}});
    auto problem = std::make_shared<ResettableProblem>(
        std::make_shared<NewtonVars>(Eigen::VectorXd::Zero(PatternTerm::n).eval()),
        NewtonMultiobjectiveProblem::Terms{staticTerm, dynamicTerm});
    auto &slru = *problem->sparsityLRUPtr();
    slru.verbose = false;
    slru.entryCacheBudgetRatio = 1;
    slru.expirationAge = 2;
    slru.hardExpirationAge = 3;

    NewtonOptimizerOptions options;
    options.factorizer = CholeskyProvider::CatamariNesdisReuse;
    NewtonHessianFactorization first(problem, options), second(problem, options);
    first.updateSymbolicFactorization();
    second.updateSymbolicFactorization();
    REQUIRE(stats(first).full_rebuild);
    REQUIRE(stats(second).full_rebuild);
    const size_t resetID = problem->symbolicFactorizationResetID();

    // New connectivity updates symbolic factorization while preserving ND.
    dynamicTerm->setEdges({{0, 3}, {1, 4}});
    first.updateSymbolicFactorization();
    REQUIRE_FALSE(stats(first).full_rebuild);
    REQUIRE(problem->symbolicFactorizationResetID() == resetID);

    SECTION("hard expiration with a nonempty dynamic pattern") {
        dynamicTerm->setEdges({{1, 4}});
        first.updateSymbolicFactorization(); // Retained edge age 1, no rebuild.
        REQUIRE(problem->symbolicFactorizationResetID() == resetID);
        problem->updateSparsityPattern(); // age 2
        problem->updateSparsityPattern(); // age 3, expires
    }
    SECTION("hard expiration with an empty dynamic pattern") {
        slru.entryCacheBudgetRatio = 2; // Exercise update's empty-pattern fast path.
        dynamicTerm->setEdges({});
        first.updateSymbolicFactorization();
        REQUIRE(problem->symbolicFactorizationResetID() == resetID);
        problem->updateSparsityPattern();
        problem->updateSparsityPattern();
    }
    SECTION("budget eviction without new entries") {
        slru.entryCacheBudgetRatio = 0;
        dynamicTerm->setEdges({{1, 4}});
        problem->updateSparsityPattern();
    }
    SECTION("budget eviction with new entries") {
        slru.entryCacheBudgetRatio = 0;
        dynamicTerm->setEdges({{1, 4}, {2, 6}});
        problem->updateSparsityPattern();
    }
    SECTION("reset without a sparsity change") {
        const size_t patternID = problem->sparsityPatternID();
        problem->requestReset();
        REQUIRE(problem->sparsityPatternID() == patternID);
    }

    REQUIRE(problem->symbolicFactorizationResetID() == resetID + 1);
    // Additional pattern queries must not lose the reset before either solver
    // consumes it. A boolean that clears on the next update would fail here.
    problem->updateSparsityPattern();
    first.updateSymbolicFactorization();
    REQUIRE(stats(first).full_rebuild);
    second.updateSymbolicFactorization();
    REQUIRE(stats(second).full_rebuild);

    // An ordinary addition after the reset can reuse the fresh ND tree again.
    auto edges = dynamicTerm->edges;
    edges.emplace_back(5, 9);
    dynamicTerm->setEdges(std::move(edges));
    first.updateSymbolicFactorization();
    second.updateSymbolicFactorization();
    REQUIRE_FALSE(stats(first).full_rebuild);
    REQUIRE_FALSE(stats(second).full_rebuild);
    REQUIRE(problem->symbolicFactorizationResetID() == resetID + 1);
}
#endif
